import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(a, b, acc, allow_tf32=False)

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 4096}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_N': 8192}, num_stages=2, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def bn_scale_softmax_kernel(
    X_ptr, Out_ptr,
    bn_mean_ptr, bn_var_ptr,
    bn_weight_ptr, bn_bias_ptr,
    scale_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    bn_eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    scale_val = tl.load(scale_ptr)

    # Pass 1: compute BN+scale and find row max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N

        x = tl.load(X_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0)
        mean = tl.load(bn_mean_ptr + cols, mask=mask, other=0.0)
        var = tl.load(bn_var_ptr + cols, mask=mask, other=0.0)
        w = tl.load(bn_weight_ptr + cols, mask=mask, other=0.0)
        b = tl.load(bn_bias_ptr + cols, mask=mask, other=0.0)

        x_norm = (x - mean) / tl.sqrt(var + bn_eps)
        x_bn = x_norm * w + b
        x_scaled = x_bn * scale_val

        # Store intermediate result
        tl.store(Out_ptr + row * stride_om + cols * stride_on, x_scaled, mask=mask)

        col_max = tl.max(tl.where(mask, x_scaled, float('-inf')), axis=0)
        row_max = tl.maximum(row_max, col_max)

    # Pass 2: compute exp sum
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N

        x_scaled = tl.load(Out_ptr + row * stride_om + cols * stride_on, mask=mask, other=0.0)
        exp_val = tl.exp(x_scaled - row_max)
        row_sum += tl.sum(tl.where(mask, exp_val, 0.0), axis=0)

    # Pass 3: normalize
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N

        x_scaled = tl.load(Out_ptr + row * stride_om + cols * stride_on, mask=mask, other=0.0)
        exp_val = tl.exp(x_scaled - row_max)
        softmax_val = exp_val / row_sum
        tl.store(Out_ptr + row * stride_om + cols * stride_on, softmax_val, mask=mask)


def kernel_function(p_gemm_weight, p_gemm_bias, p_bn_weight, p_bn_bias,
    M, K = x.shape
    N = p_gemm_weight.shape[0]

    # Allocate output for GEMM
    gemm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Launch GEMM kernel
    grid_gemm = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    gemm_kernel[grid_gemm](
        x, p_gemm_weight, p_gemm_bias, gemm_out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_gemm_weight.stride(1), p_gemm_weight.stride(0),
        gemm_out.stride(0), gemm_out.stride(1),
    )

    # Allocate output for BN+scale+softmax
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Launch BN+scale+softmax kernel
    grid_softmax = (M,)
    bn_scale_softmax_kernel[grid_softmax](
        gemm_out, out,
        b_bn_running_mean, b_bn_running_var,
        p_bn_weight, p_bn_bias,
        p_scale,
        M, N,
        gemm_out.stride(0), gemm_out.stride(1),
        out.stride(0), out.stride(1),
        bn_eps=1e-5,
    )

    return out


def get_inputs():
    p_gemm_weight = torch.randn(8192, 8192, dtype=torch.float32, device='cuda')
    p_gemm_bias = torch.randn(8192, dtype=torch.float32, device='cuda')
    p_bn_weight = torch.randn(8192, dtype=torch.float32, device='cuda')
    p_bn_bias = torch.randn(8192, dtype=torch.float32, device='cuda')
    p_scale = torch.ones(1, dtype=torch.float32, device='cuda')
    b_bn_running_mean = torch.zeros(8192, dtype=torch.float32, device='cuda')
    b_bn_running_var = torch.ones(8192, dtype=torch.float32, device='cuda')
    x = torch.randn(1024, 8192, dtype=torch.float32, device='cuda')
    return [p_gemm_weight, p_gemm_bias, p_bn_weight, p_bn_bias,


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
        p_gemm_weight, p_gemm_bias, p_bn_weight, p_bn_bias, p_scale, b_bn_running_mean, b_bn_running_var, x = inputs

        # Reference computation
        linear_out = x @ p_gemm_weight.T + p_gemm_bias
        bn_out = F.batch_norm(linear_out, b_bn_running_mean, b_bn_running_var,
                              p_bn_weight, p_bn_bias, training=False, eps=1e-5)
        scaled = p_scale * bn_out
        ref = torch.softmax(scaled, dim=1)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")