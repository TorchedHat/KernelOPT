import torch
import triton
import triton.language as tl


# GEMM + bias + scale + residual fused kernel
# Computes: out = (x @ W.T + bias) * scale + (x @ W.T + bias)
# Which simplifies to: out = linear_out * (1 + scale)
# where scale = 0.5, so out = linear_out * 1.5


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=4),
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

    # Masks (kept for bias load and output store)
    mask_m = rm < M
    mask_n = rn < N

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Block pointers for TMA-style async loads
    x_block_ptr = tl.make_block_ptr(
        base=X_ptr, shape=(M, K), strides=(stride_xm, stride_xk),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    # W is stored as (N, K) -- weight matrix of linear layer
    # Load W as [BLOCK_N, BLOCK_K] (coalesced: rn rows with stride_wn, rk cols with stride_wk=1)
    w_block_ptr = tl.make_block_ptr(
        base=W_ptr, shape=(N, K), strides=(stride_wn, stride_wk),
        offsets=(pid_n * BLOCK_N, 0), block_shape=(BLOCK_N, BLOCK_K), order=(1, 0))

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Load X tile [BLOCK_M, BLOCK_K] via TMA block pointer
        x_tile = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option='zero')

        # Load W tile [BLOCK_N, BLOCK_K] via TMA block pointer
        w_tile = tl.load(w_block_ptr, boundary_check=(0, 1), padding_option='zero')

        # tl.trans(w_tile) gives [BLOCK_K, BLOCK_N] for the dot product: [M,K] x [K,N]
        # input_precision='tf32x3': Ozaki decomposition using 3 TF32 HMMA ops for FP32-equivalent accuracy
        acc = tl.dot(x_tile, tl.trans(w_tile), acc, input_precision='tf32x3')

        x_block_ptr = tl.advance(x_block_ptr, (0, BLOCK_K))
        w_block_ptr = tl.advance(w_block_ptr, (0, BLOCK_K))

    # Add bias
    bias = tl.load(B_ptr + rn, mask=mask_n, other=0.0,
                   cache_modifier='.cg')  # cache at L2 only: tiny vector, read once per output tile
    acc = acc + bias[None, :]

    # Fuse: out = acc * scaling_factor + acc = acc * (1 + scaling_factor)
    combined_scale = 1.0 + scaling_factor
    acc = acc * combined_scale

    # Store output
    out_ptrs = Out_ptr + rm[:, None] * stride_om + rn[None, :] * stride_on
    tl.store(
        out_ptrs,
        acc.to(tl.float32),
        mask=mask_m[:, None] & mask_n[None, :],
        cache_modifier='.cs'  # evict_first: write-once streaming output, avoid polluting L2
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