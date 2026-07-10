import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    w_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        mask_x = (offs_m[:, None] < M) & (mask_k[None, :])
        mask_w = (mask_k[:, None]) & (offs_n[None, :] < N)

        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x, w, acc, allow_tf32=False)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_out = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    bias = tl.load(B_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    out_ptrs = Out_ptr + offs_m_out[:, None] * stride_om + offs_n_out[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def kernel_function(p_gemm_weight, p_bias, x):
    # p_gemm_weight: [out_features, in_features] = [8192, 8192]
    # p_bias: [out_features] = [8192]
    # x: [batch_size, in_features] = [1024, 8192]
    # computation: relu(x @ p_gemm_weight.T + p_bias)
    assert p_gemm_weight.ndim == 2, f"Expected 2D weight, got {p_gemm_weight.ndim}D"
    assert p_bias.ndim == 1, f"Expected 1D bias, got {p_bias.ndim}D"
    assert x.ndim == 2, f"Expected 2D input, got {x.ndim}D"

    M, K = x.shape
    N, K2 = p_gemm_weight.shape
    assert K == K2, f"Shape mismatch: x has K={K}, weight has K={K2}"

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    fused_linear_bias_relu_kernel[grid](
        x, p_gemm_weight, p_bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_gemm_weight.stride(0), p_gemm_weight.stride(1),
        out.stride(0), out.stride(1),
    )

    return out


def get_inputs():
    p_gemm_weight = torch.randn(8192, 8192, device='cuda', dtype=torch.float32)
    p_bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    x = torch.rand(1024, 8192, device='cuda', dtype=torch.float32)
    return [p_gemm_weight, p_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_gemm_weight, p_bias, x = inputs

    output = kernel_function(p_gemm_weight, p_bias, x)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference: relu(x @ weight.T + bias)
        ref = torch.relu(x @ p_gemm_weight.T + p_bias)
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")