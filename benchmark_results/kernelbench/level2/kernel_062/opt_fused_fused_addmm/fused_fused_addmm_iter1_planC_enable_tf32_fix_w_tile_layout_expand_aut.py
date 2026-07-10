import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=16),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
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
    group_size_m = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size_m)
    pid_n = (pid % width) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # W has shape [N, K]: load [BLOCK_N, BLOCK_K] tile (contiguous along K)
        w_ptrs = W_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w_mask = (offs_n[:, None] < N) & (k_offs[None, :] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # shape [BLOCK_N, BLOCK_K]

        # tl.dot(A[M,K], B[K,N]) — transpose w_tile from [BLOCK_N, BLOCK_K] to [BLOCK_K, BLOCK_N]
        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=True)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
    ],
    key=['group_size'],
)
@triton.jit

def groupnorm_leakyrelu_add_kernel(
    X_ptr, GN_W_ptr, GN_B_ptr, Out_ptr,
    num_rows, N, num_groups, group_size,
    eps, negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    group = tl.program_id(1)

    base = row * N + group * group_size
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < group_size

    x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)

    mean = tl.sum(x, axis=0) / group_size
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / group_size
    inv_std = libdevice.rsqrt(var + eps)
    x_norm = diff * inv_std

    gn_w = tl.load(GN_W_ptr + group * group_size + offs, mask=mask, other=1.0).to(tl.float32)
    gn_b = tl.load(GN_B_ptr + group * group_size + offs, mask=mask, other=0.0).to(tl.float32)

    x_scaled = x_norm * gn_w + gn_b
    x_relu = tl.where(x_scaled > 0.0, x_scaled, x_scaled * negative_slope)
    x_out = x_relu + x_relu

    tl.store(Out_ptr + base + offs, x_out.to(tl.float32), mask=mask)


def kernel_function(p_fc_weight, p_fc_bias, p_gn_weight, p_gn_bias, x):
    M, K = x.shape
    N = p_fc_weight.shape[0]

    linear_out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid_gemm = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    gemm_bias_kernel[grid_gemm](
        x, p_fc_weight, p_fc_bias, linear_out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_fc_weight.stride(0), p_fc_weight.stride(1),
    )

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    num_groups = 512
    group_size = N // num_groups  # 16

    grid_gn = (M, num_groups)

    groupnorm_leakyrelu_add_kernel[grid_gn](
        linear_out, p_gn_weight, p_gn_bias, out,
        M, N, num_groups, group_size,
        1e-5, 0.01,
    )

    return out


def get_inputs():
    p_fc_weight = torch.randn(8192, 8192, device='cuda', dtype=torch.float32)
    p_fc_bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    p_gn_weight = torch.randn(8192, device='cuda', dtype=torch.float32)
    p_gn_bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    x = torch.randn(1024, 8192, device='cuda', dtype=torch.float32)
    return [p_fc_weight, p_fc_bias, p_gn_weight, p_gn_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(0)
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn
        p_fc_weight, p_fc_bias, p_gn_weight, p_gn_bias, x = inputs

        fc = nn.Linear(8192, 8192).cuda()
        fc.weight.data = p_fc_weight
        fc.bias.data = p_fc_bias
        gn = nn.GroupNorm(num_groups=512, num_channels=8192, eps=1e-5).cuda()
        gn.weight.data = p_gn_weight
        gn.bias.data = p_gn_bias
        leaky_relu = nn.LeakyReLU(negative_slope=0.01)

        with torch.no_grad():
            ref = fc(x)
            ref = gn(ref)
            ref = leaky_relu(ref)
            ref = ref + ref

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")