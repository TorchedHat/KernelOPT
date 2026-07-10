import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

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
def fused_linear_swish_tanh_gelu_hardtanh_kernel(
    X_ptr, W_ptr, bias_ptr, add_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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

        # W is (N, K), we want W[n, k] for output[m, n] = sum_k X[m,k] * W[n,k]
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Load bias and add_value
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    add_val = tl.load(add_ptr + offs_n, mask=mask_n, other=0.0)

    # Add bias and add_value
    x = acc + bias[None, :] + add_val[None, :]

    # Swish: sigmoid(x) * x
    sig = tl.sigmoid(x)
    x = sig * x

    # Tanh
    x = libdevice.tanh(x)

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    x_gelu = x * 0.5 * (libdevice.erf(x * 0.7071067811865476) + 1.0)

    # Hardtanh: clamp to [-1, 1]
    x_out = tl.minimum(tl.maximum(x_gelu, -1.0), 1.0)

    # Store
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, x_out.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(p_matmul_weight, p_matmul_bias, p_add_value, x):
    # p_matmul_weight: (N, K) = (8192, 8192)
    # p_matmul_bias: (N,) = (8192,)
    # p_add_value: (N,) = (8192,)
    # x: (M, K) = (1024, 8192)
    assert x.ndim == 2, f"x must be 2D, got shape {x.shape}"
    assert p_matmul_weight.ndim == 2, f"Weight must be 2D, got shape {p_matmul_weight.shape}"

    M, K = x.shape
    N_out, K2 = p_matmul_weight.shape
    assert K == K2, f"K mismatch: {K} vs {K2}"

    out = torch.empty((M, N_out), device=x.device, dtype=torch.float32)

    # Ensure contiguous
    x = x.contiguous()
    w = p_matmul_weight.contiguous()

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N_out, meta['BLOCK_N']),)

    fused_linear_swish_tanh_gelu_hardtanh_kernel[grid](
        x, w, p_matmul_bias, p_add_value, out,
        M, N_out, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
    )

    return out


def get_inputs():
    p_matmul_weight = torch.randn(8192, 8192, device='cuda', dtype=torch.float32)
    p_matmul_bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    p_add_value = torch.randn(8192, device='cuda', dtype=torch.float32)
    x = torch.rand(1024, 8192, device='cuda', dtype=torch.float32)
    return [p_matmul_weight, p_matmul_bias, p_add_value, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_matmul_weight, p_matmul_bias, p_add_value, x = inputs

        # Reference computation
        import torch.nn.functional as F
        ref = F.linear(x, p_matmul_weight, p_matmul_bias)
        ref = ref + p_add_value
        ref = torch.sigmoid(ref) * ref  # Swish
        ref = torch.tanh(ref)
        ref = F.gelu(ref)
        ref = F.hardtanh(ref, min_val=-1, max_val=1)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")