import torch
import triton
import triton.language as tl


# GEMM + bias + scale + residual fused kernel
# Computes: out = (x @ W.T + bias) * scale + (x @ W.T + bias)
# Which simplifies to: out = linear_out * (1 + scale)
# where scale = 0.5, so out = linear_out * 1.5


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8,  maxnreg=128),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8,  maxnreg=128),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4,  maxnreg=128),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4,  maxnreg=128),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4,  maxnreg=128),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4,  maxnreg=128),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8,  maxnreg=128),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4,  maxnreg=128),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=5, num_warps=8,  maxnreg=128),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4,  maxnreg=128),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_scale_residual_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    scaling_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache optimization: group programs
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Row/col offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Masks
    mask_m = rm < M
    mask_n = rn < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pointers to X and W
    X_ptrs = X_ptr + rm[:, None] * stride_xm + rk[None, :] * stride_xk
    # W is stored as (N, K) -- weight matrix of linear layer
    # Load W as [BLOCK_N, BLOCK_K] so K is the fast-varying axis (contiguous),
    # eliminating shared memory bank conflicts. Use tl.trans() in tl.dot to
    # preserve the x @ W.T matmul semantics.
    W_ptrs = W_ptr + rn[:, None] * stride_wn + rk[None, :] * stride_wk

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        mask_k = (rk + k_offset) < K

        # Load X tile [BLOCK_M, BLOCK_K]
        x_tile = tl.load(
            X_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )

        # Load W tile [BLOCK_N, BLOCK_K] -- K is fast axis, matching physical (N,K) layout
        w_tile = tl.load(
            W_ptrs,
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0
        )

        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=True)

        X_ptrs += BLOCK_K * stride_xk
        W_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias = tl.load(B_ptr + rn, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Fuse: out = acc * scaling_factor + acc = acc * (1 + scaling_factor)
    combined_scale = 1.0 + scaling_factor
    acc = acc * combined_scale

    # Store output
    out_ptrs = Out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc.to(tl.float32),
        mask=mask_m[:, None] & mask_n[None, :]
    )

def kernel_function(p_matmul_weight, p_matmul_bias, x, scaling_factor=0.5):
    M, K = x.shape
    N, K2 = p_matmul_weight.shape
    assert K == K2

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    fused_linear_scale_residual_kernel[grid](
        x, p_matmul_weight, p_matmul_bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_matmul_weight.stride(0), p_matmul_weight.stride(1),
        out.stride(0), out.stride(1),
        scaling_factor,
    )

    return out


def get_inputs():
    p_matmul_weight = torch.randn(4096, 4096, dtype=torch.float32, device='cuda')
    p_matmul_bias = torch.randn(4096, dtype=torch.float32, device='cuda')
    x = torch.rand(16384, 4096, dtype=torch.float32, device='cuda')
    return [p_matmul_weight, p_matmul_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_matmul_weight, p_matmul_bias, x = inputs

    output = kernel_function(p_matmul_weight, p_matmul_bias, x, scaling_factor=0.5)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference computation
        linear_out = x @ p_matmul_weight.T + p_matmul_bias
        original_x = linear_out.clone().detach()
        scaled = linear_out * 0.5
        ref = scaled + original_x

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")