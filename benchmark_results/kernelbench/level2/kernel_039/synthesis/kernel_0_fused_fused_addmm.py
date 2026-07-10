import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_scale_bn_kernel(
    X_ptr, W_ptr, bias_ptr, scale_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    eps,
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    scale = tl.load(scale_ptr + offs_n, mask=mask_n, other=1.0)
    acc = acc * scale[None, :]

    bn_mean = tl.load(bn_mean_ptr + offs_n, mask=mask_n, other=0.0)
    bn_var = tl.load(bn_var_ptr + offs_n, mask=mask_n, other=1.0)
    bn_w = tl.load(bn_weight_ptr + offs_n, mask=mask_n, other=1.0)
    bn_b = tl.load(bn_bias_ptr + offs_n, mask=mask_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    acc = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(p_gemm_weight, p_gemm_bias, p_scale, p_bn_weight, p_bn_bias,
    M = x.shape[0]
    K = x.shape[1]
    N = p_gemm_weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    eps = 1e-5

    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    fused_linear_scale_bn_kernel[grid](
        x, p_gemm_weight, p_gemm_bias, p_scale,
        b_bn_running_mean, b_bn_running_var, p_bn_weight, p_bn_bias,
        out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_gemm_weight.stride(0), p_gemm_weight.stride(1),
        eps,
    )

    return out


def get_inputs():
    p_gemm_weight = torch.randn(4096, 4096, device='cuda', dtype=torch.float32)
    p_gemm_bias = torch.randn(4096, device='cuda', dtype=torch.float32)
    p_scale = torch.randn(4096, device='cuda', dtype=torch.float32)
    p_bn_weight = torch.randn(4096, device='cuda', dtype=torch.float32)
    p_bn_bias = torch.randn(4096, device='cuda', dtype=torch.float32)
    b_bn_running_mean = torch.randn(4096, device='cuda', dtype=torch.float32)
    b_bn_running_var = torch.abs(torch.randn(4096, device='cuda', dtype=torch.float32)) + 0.1
    x = torch.randn(16384, 4096, device='cuda', dtype=torch.float32)
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

        ref_linear = x @ p_gemm_weight.t() + p_gemm_bias
        ref_scaled = ref_linear * p_scale
        eps = 1e-5
        inv_std = 1.0 / torch.sqrt(b_bn_running_var + eps)
        ref_bn = (ref_scaled - b_bn_running_mean) * inv_std * p_bn_weight + p_bn_bias

        assert torch.allclose(output, ref_bn, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref_bn).abs().max().item()):.4e}"
        print("Kernel executed successfully.")