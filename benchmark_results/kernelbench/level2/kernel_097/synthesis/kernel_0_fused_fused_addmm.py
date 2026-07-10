import torch
import triton
import triton.language as tl

# GEMM kernel with fused BN + bias + div + swish epilogue
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
def fused_linear_bn_swish_kernel(
    # Input pointers
    X_ptr, W_ptr, linear_bias_ptr,
    bn_weight_ptr, bn_bias_ptr,
    p_bias_ptr,
    bn_mean_ptr, bn_var_ptr,
    # Output pointer
    Out_ptr,
    # Dimensions
    M, N, K,
    # Strides
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    # BN params
    bn_eps: tl.constexpr,
    divide_value: tl.constexpr,
    # Block sizes
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

    # Pointers for A (X) and B (W^T, so we load W with transposed strides)
    X_ptrs = X_ptr + (offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    W_ptrs = W_ptr + (offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        mask_x = (offs_m[:, None] < M) & (mask_k[None, :])
        mask_w = (mask_k[:, None]) & (offs_n[None, :] < N)

        x_tile = tl.load(X_ptrs, mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

        X_ptrs += BLOCK_K * stride_xk
        W_ptrs += BLOCK_K * stride_wk

    # Load linear bias (per output feature)
    lin_bias = tl.load(linear_bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    # Load BN params
    bn_mean = tl.load(bn_mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    bn_var  = tl.load(bn_var_ptr  + offs_n, mask=offs_n < N, other=1.0)
    bn_w    = tl.load(bn_weight_ptr + offs_n, mask=offs_n < N, other=1.0)
    bn_b    = tl.load(bn_bias_ptr   + offs_n, mask=offs_n < N, other=0.0)

    # Load p_bias (scalar broadcast)
    p_bias = tl.load(p_bias_ptr)

    # Epilogue: linear bias add
    acc = acc + lin_bias[None, :]

    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(bn_var + bn_eps)
    acc = (acc - bn_mean[None, :]) * inv_std[None, :] * bn_w[None, :] + bn_b[None, :]

    # Add p_bias
    acc = acc + p_bias

    # Divide
    acc = acc / divide_value

    # Swish: x * sigmoid(x)
    sig = tl.sigmoid(acc)
    acc = acc * sig

    # Store
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    Out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(Out_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(p_matmul_weight, p_matmul_bias, p_bn_weight, p_bn_bias,
    M, K = x.shape
    N = p_matmul_weight.shape[0]  # weight is [out_features, in_features]

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    bn_eps = 1e-5
    divide_value = 1.0

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    fused_linear_bn_swish_kernel[grid](
        x, p_matmul_weight, p_matmul_bias,
        p_bn_weight, p_bn_bias,
        p_bias,
        b_bn_running_mean, b_bn_running_var,
        out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_matmul_weight.stride(0), p_matmul_weight.stride(1),
        out.stride(0), out.stride(1),
        bn_eps=bn_eps,
        divide_value=divide_value,
    )

    return out


def get_inputs():
    batch_size = 1024
    in_features = 8192
    out_features = 8192

    p_matmul_weight = torch.randn(out_features, in_features, device='cuda', dtype=torch.float32)
    p_matmul_bias   = torch.randn(out_features, device='cuda', dtype=torch.float32)
    p_bn_weight     = torch.randn(out_features, device='cuda', dtype=torch.float32)
    p_bn_bias       = torch.randn(out_features, device='cuda', dtype=torch.float32)
    p_bias          = torch.randn(1, device='cuda', dtype=torch.float32)
    b_bn_running_mean = torch.zeros(out_features, device='cuda', dtype=torch.float32)
    b_bn_running_var  = torch.ones(out_features, device='cuda', dtype=torch.float32)
    x = torch.randn(batch_size, in_features, device='cuda', dtype=torch.float32)

    return [p_matmul_weight, p_matmul_bias, p_bn_weight, p_bn_bias,


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_matmul_weight, p_matmul_bias, p_bn_weight, p_bn_bias, \
            p_bias, b_bn_running_mean, b_bn_running_var, x = inputs

        # Reference computation
        import torch.nn.functional as F
        # Linear
        ref = F.linear(x, p_matmul_weight, p_matmul_bias)
        # BatchNorm (inference)
        ref = F.batch_norm(ref, b_bn_running_mean, b_bn_running_var,
                           p_bn_weight, p_bn_bias, training=False, eps=1e-5)
        # Add bias
        ref = ref + p_bias
        # Divide
        ref = ref / 1.0
        # Swish
        ref = ref * torch.sigmoid(ref)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")