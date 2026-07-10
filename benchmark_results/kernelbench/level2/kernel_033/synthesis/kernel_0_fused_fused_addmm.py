import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_scale_bn_kernel(
    X_ptr, W_ptr, bias_ptr, scale_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache-friendly ordering
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Pointers for X and W
    X_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # W is stored as [N, K] (weight matrix), we compute X @ W.T
    W_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining

        x_tile = tl.load(X_ptrs,
                         mask=mask_m[:, None] & mask_k[None, :],
                         other=0.0)
        w_tile = tl.load(W_ptrs,
                         mask=mask_k[:, None] & mask_n[None, :],
                         other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

        X_ptrs += BLOCK_K * stride_xk
        W_ptrs += BLOCK_K * stride_wk

    # Load per-column parameters
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0)
    bn_mean = tl.load(bn_mean_ptr + offs_n, mask=mask_n, other=0.0)
    bn_var = tl.load(bn_var_ptr + offs_n, mask=mask_n, other=1.0)
    bn_w = tl.load(bn_weight_ptr + offs_n, mask=mask_n, other=1.0)
    bn_b = tl.load(bn_bias_ptr + offs_n, mask=mask_n, other=0.0)

    # Epilogue: bias add + scale + batch norm (inference)
    # acc shape: [BLOCK_M, BLOCK_N]
    # bias, scale, bn_mean, bn_var, bn_w, bn_b shape: [BLOCK_N]
    result = acc + bias[None, :]          # add linear bias
    result = result * scale[None, :]      # multiply by scale
    # batch norm: (x - mean) / sqrt(var + eps) * bn_weight + bn_bias
    inv_std = 1.0 / tl.sqrt(bn_var + eps)
    result = (result - bn_mean[None, :]) * inv_std[None, :] * bn_w[None, :] + bn_b[None, :]

    # Store output
    Y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(Y_ptrs, result.to(tl.float32),
             mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(p_gemm_weight, p_gemm_bias, p_scale, p_bn_weight, p_bn_bias,
    M, K = x.shape
    N = p_gemm_weight.shape[0]  # weight is [out_features, in_features]

    Y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    fused_gemm_scale_bn_kernel[grid](
        x, p_gemm_weight, p_gemm_bias, p_scale,
        b_bn_running_mean, b_bn_running_var, p_bn_weight, p_bn_bias,
        Y,
        M, N, K,
        x.stride(0), x.stride(1),
        p_gemm_weight.stride(0), p_gemm_weight.stride(1),
        Y.stride(0), Y.stride(1),
        eps=1e-5,
    )

    return Y


def get_inputs():
    p_gemm_weight = torch.randn(8192, 8192, device='cuda', dtype=torch.float32)
    p_gemm_bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    p_scale = torch.randn(8192, device='cuda', dtype=torch.float32)
    p_bn_weight = torch.randn(8192, device='cuda', dtype=torch.float32)
    p_bn_bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    b_bn_running_mean = torch.randn(8192, device='cuda', dtype=torch.float32)
    b_bn_running_var = torch.abs(torch.randn(8192, device='cuda', dtype=torch.float32)) + 0.1
    x = torch.randn(1024, 8192, device='cuda', dtype=torch.float32)
    return [p_gemm_weight, p_gemm_bias, p_scale, p_bn_weight, p_bn_bias,


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_gemm_weight, p_gemm_bias, p_scale, p_bn_weight, p_bn_bias, \
            b_bn_running_mean, b_bn_running_var, x = inputs

        # Reference computation
        ref = x @ p_gemm_weight.T + p_gemm_bias
        ref = ref * p_scale
        # Batch norm inference: (x - mean) / sqrt(var + eps) * weight + bias
        inv_std = 1.0 / torch.sqrt(b_bn_running_var + 1e-5)
        ref = (ref - b_bn_running_mean) * inv_std * p_bn_weight + p_bn_bias

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")