import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_mish_mish_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache-friendly reordering
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Row/col offsets
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    X_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is stored as (N, K) i.e. weight matrix, we access W[n, k]
    W_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    # Accumulate
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining

        x_tile = tl.load(
            X_ptrs,
            mask=(offs_m[:, None] < M) & (mask_k[None, :]),
            other=0.0
        )
        w_tile = tl.load(
            W_ptrs,
            mask=(mask_k[:, None]) & (offs_n[None, :] < N),
            other=0.0
        )

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

        X_ptrs += BLOCK_K * stride_xk
        W_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Apply mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)), but for x > 20, softplus(x) ≈ x
    threshold = 20.0

    # First mish
    gt1 = acc > threshold
    exp1 = libdevice.exp(acc)
    log1p1 = libdevice.log1p(exp1)
    sp1 = tl.where(gt1, acc, log1p1)
    tanh1 = libdevice.tanh(sp1)
    mish1 = acc * tanh1

    # Second mish
    gt2 = mish1 > threshold
    exp2 = libdevice.exp(mish1)
    log1p2 = libdevice.log1p(exp2)
    sp2 = tl.where(gt2, mish1, log1p2)
    tanh2 = libdevice.tanh(sp2)
    mish2 = mish1 * tanh2

    # Store result
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    Y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(Y_ptrs, mish2.to(tl.float32), mask=out_mask)


def kernel_function(p_linear_weight, p_linear_bias, x):
    M, K = x.shape
    N = p_linear_weight.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_mish_mish_kernel[grid](
        x, p_linear_weight, p_linear_bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        p_linear_weight.stride(0), p_linear_weight.stride(1),
    )

    return y


def get_inputs():
    p_linear_weight = torch.randn(8192, 8192, dtype=torch.float32, device='cuda')
    p_linear_bias = torch.randn(8192, dtype=torch.float32, device='cuda')
    x = torch.rand(1024, 8192, dtype=torch.float32, device='cuda')
    return [p_linear_weight, p_linear_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_linear_weight, p_linear_bias, x = inputs

        # Reference computation
        import torch.nn.functional as F
        linear_out = F.linear(x, p_linear_weight, p_linear_bias)
        mish1_ref = F.mish(linear_out)
        mish2_ref = F.mish(mish1_ref)

        assert torch.allclose(output, mish2_ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - mish2_ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")