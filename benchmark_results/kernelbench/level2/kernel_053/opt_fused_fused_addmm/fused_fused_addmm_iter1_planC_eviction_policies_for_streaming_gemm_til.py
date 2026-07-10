import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fused_linear_mul_hardtanh_gelu_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    scaling_factor: tl.constexpr,
    hardtanh_min: tl.constexpr,
    hardtanh_max: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program IDs
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache-friendly swizzling
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
    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    # Weight is stored as [N, K] (out_features x in_features), we access w[n, k]
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    # Accumulate
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining

        x_tile = tl.load(
            x_ptrs,
            mask=(offs_m[:, None] < M) & (mask_k[None, :]),
            other=0.0,
            cache_modifier=".ca"
        )
        w_tile = tl.load(
            w_ptrs,
            mask=(mask_k[:, None]) & (offs_n[None, :] < N),
            other=0.0,
            cache_modifier=".ca"
        )

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Load bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    # Add bias
    acc = acc + bias[None, :]

    # Scale (mul)
    acc = acc * scaling_factor

    # Hardtanh (clamp)
    acc = tl.maximum(acc, hardtanh_min)
    acc = tl.minimum(acc, hardtanh_max)

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    sqrt2_inv = 0.7071067811865476
    gelu_out = acc * 0.5 * (1.0 + libdevice.erf(acc * sqrt2_inv))

    # Store
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, gelu_out.to(tl.float32), mask=mask_out, cache_modifier=".cs")

def kernel_function(p_gemm_weight, p_gemm_bias, x):
    # p_gemm_weight: [out_features, in_features] = [8192, 8192]
    # p_gemm_bias: [out_features] = [8192]
    # x: [batch, in_features] = [2048, 8192]
    M, K = x.shape
    N = p_gemm_weight.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Weight layout: [N, K] where N=out_features, K=in_features
    # stride_wn = K (stride along output feature dim)
    # stride_wk = 1 (stride along input feature dim)
    stride_wn = p_gemm_weight.stride(0)  # K
    stride_wk = p_gemm_weight.stride(1)  # 1

    GROUP_M = 8

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    fused_linear_mul_hardtanh_gelu_kernel[grid](
        x, p_gemm_weight, p_gemm_bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        stride_wn, stride_wk,
        out.stride(0), out.stride(1),
        scaling_factor=0.5,
        hardtanh_min=-2.0,
        hardtanh_max=2.0,
        GROUP_M=GROUP_M,
    )

    return out


def get_inputs():
    p_gemm_weight = torch.randn(8192, 8192, dtype=torch.float32, device='cuda')
    p_gemm_bias = torch.randn(8192, dtype=torch.float32, device='cuda')
    x = torch.rand(2048, 8192, dtype=torch.float32, device='cuda')
    return [p_gemm_weight, p_gemm_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F

        p_gemm_weight, p_gemm_bias, x = inputs

        # Reference computation
        linear_out = F.linear(x, p_gemm_weight, p_gemm_bias)
        scaled = linear_out * 0.5
        clamped = torch.clamp(scaled, min=-2.0, max=2.0)
        ref = clamped * 0.5 * (1.0 + torch.erf(clamped * 0.7071067811865476))

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")