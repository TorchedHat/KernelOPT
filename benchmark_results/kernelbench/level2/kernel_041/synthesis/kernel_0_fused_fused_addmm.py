import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_gemm_bn_gelu_relu_kernel(
    # Input matrix
    X_ptr, stride_xm, stride_xk,
    # Weight matrix (transposed: shape [N, K])
    W_ptr, stride_wn, stride_wk,
    # Bias
    bias_ptr,
    # BatchNorm params
    bn_weight_ptr, bn_bias_ptr,
    bn_running_mean_ptr, bn_running_var_ptr,
    # Output
    out_ptr, stride_om, stride_on,
    # Dimensions
    M, N, K,
    # BatchNorm epsilon
    eps,
    # Tile sizes
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
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    X_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    W_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main GEMM loop
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining

        x_tile = tl.load(X_ptrs,
                         mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining),
                         other=0.0)
        w_tile = tl.load(W_ptrs,
                         mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N),
                         other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

        X_ptrs += BLOCK_K * stride_xk
        W_ptrs += BLOCK_K * stride_wk

    # Epilogue: bias + batchnorm + gelu + relu
    # Load per-column parameters
    mask_n = offs_n < N

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    bn_mean = tl.load(bn_running_mean_ptr + offs_n, mask=mask_n, other=0.0)
    bn_var = tl.load(bn_running_var_ptr + offs_n, mask=mask_n, other=1.0)
    bn_w = tl.load(bn_weight_ptr + offs_n, mask=mask_n, other=1.0)
    bn_b = tl.load(bn_bias_ptr + offs_n, mask=mask_n, other=0.0)

    # Add linear bias
    acc = acc + bias[None, :]

    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(bn_var + eps)
    acc = (acc - bn_mean[None, :]) * inv_std[None, :] * bn_w[None, :] + bn_b[None, :]

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    sqrt2_inv = 0.7071067811865476
    gelu_out = 0.5 * acc * (1.0 + libdevice.erf(acc * sqrt2_inv))

    # ReLU
    result = tl.maximum(gelu_out, 0.0)

    # Store output
    mask_m = offs_m < M
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result.to(tl.float32),
             mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(p_gemm_weight, p_gemm_bias, p_batch_norm_weight, p_batch_norm_bias,
    M = x.shape[0]
    K = x.shape[1]
    N = p_gemm_weight.shape[0]  # weight shape is [out_features, in_features]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    eps = 1e-5

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    fused_gemm_bn_gelu_relu_kernel[grid](
        x, x.stride(0), x.stride(1),
        p_gemm_weight, p_gemm_weight.stride(0), p_gemm_weight.stride(1),
        p_gemm_bias,
        p_batch_norm_weight, p_batch_norm_bias,
        b_batch_norm_running_mean, b_batch_norm_running_var,
        out, out.stride(0), out.stride(1),
        M, N, K,
        eps,
    )

    return out


def get_inputs():
    p_gemm_weight = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
    p_gemm_bias = torch.randn(4096, device='cuda', dtype=torch.float32)
    p_batch_norm_weight = torch.randn(4096, device='cuda', dtype=torch.float32)
    p_batch_norm_bias = torch.randn(4096, device='cuda', dtype=torch.float32)
    b_batch_norm_running_mean = torch.randn(4096, device='cuda', dtype=torch.float32)
    b_batch_norm_running_var = torch.abs(torch.randn(4096, device='cuda', dtype=torch.float32)) + 0.1
    x = torch.randn(16384, 4096, device='cuda', dtype=torch.float32)
    return [p_gemm_weight, p_gemm_bias, p_batch_norm_weight, p_batch_norm_bias,


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_gemm_weight, p_gemm_bias, p_batch_norm_weight, p_batch_norm_bias, \
            b_batch_norm_running_mean, b_batch_norm_running_var, x = inputs

        # Reference computation
        linear_out = x @ p_gemm_weight.T + p_gemm_bias
        eps = 1e-5
        inv_std = 1.0 / torch.sqrt(b_batch_norm_running_var + eps)
        bn_out = (linear_out - b_batch_norm_running_mean) * inv_std * p_batch_norm_weight + p_batch_norm_bias
        gelu_out = 0.5 * bn_out * (1.0 + torch.erf(bn_out / (2.0 ** 0.5)))
        ref = torch.relu(gelu_out)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")