import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math

# GEMM kernel: computes x @ W^T + bias, output shape [M, N]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=5, num_warps=8),
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
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(M, BLOCK_M) * tl.cdiv(N, BLOCK_N)
    if pid >= num_tiles:
        return  # padded CTA beyond real tile grid, exit immediately
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    A = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a = tl.load(A, mask=(rm[:, None] < M) & (rk[None, :] < k_remaining), other=0.0)
        b = tl.load(B, mask=(rk[:, None] < k_remaining) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=True)
        A += BLOCK_K * stride_ak
        B += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + rn, mask=rn < N, other=0.0)
    acc = acc + bias[None, :]

    # Store result
    C = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(C, acc.to(tl.float32), mask=mask)


# Reduction kernel: logsumexp + leaky_relu x2 + gelu x2
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 4096}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_N': 8192}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_N': 512}, num_stages=3, num_warps=4),
    ],
    key=['N'],
)
@triton.jit

def logsumexp_leakyrelu_gelu_kernel(
    input_ptr, output_ptr,
    M, N,
    stride_m, stride_n,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_ptr = input_ptr + row_idx * stride_m

    # Pass 1: find max for numerical stability
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_ptr + cols * stride_n, mask=mask, other=float('-inf'))
        x = x.to(tl.float32)
        local_max = tl.max(x, axis=0)
        row_max = tl.maximum(row_max, local_max)

    # Handle inf max: if abs(max) == inf, use 0 instead
    abs_max = tl.abs(row_max)
    is_inf = abs_max == float('inf')
    safe_max = tl.where(is_inf, 0.0, row_max)

    # Pass 2: compute sum of exp(x - safe_max)
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_ptr + cols * stride_n, mask=mask, other=float('-inf'))
        x = x.to(tl.float32)
        x_shifted = x - safe_max
        exp_x = libdevice.exp(x_shifted)
        exp_x = tl.where(mask, exp_x, 0.0)
        row_sum += tl.sum(exp_x, axis=0)

    # logsumexp = log(sum) + safe_max
    lse = libdevice.log(row_sum) + safe_max

    # leaky_relu 1
    neg_slope = 0.01
    lr1 = tl.where(lse > 0.0, lse, lse * neg_slope)

    # leaky_relu 2
    lr2 = tl.where(lr1 > 0.0, lr1, lr1 * neg_slope)

    # gelu 1: x * 0.5 * (1 + erf(x * 0.7071067811865476))
    sqrt2_inv = 0.7071067811865476
    g1 = lr2 * 0.5 * (libdevice.erf(lr2 * sqrt2_inv) + 1.0)

    # gelu 2
    g2 = g1 * 0.5 * (libdevice.erf(g1 * sqrt2_inv) + 1.0)

    # Store output
    tl.store(output_ptr + row_idx, g2)


def kernel_function(weight, bias, x):
    # weight: [out_features, in_features] = [8192, 8192]
    # bias: [out_features] = [8192]
    # x: [batch, in_features] = [1024, 8192]
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2

    # Allocate GEMM output
    gemm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Launch GEMM kernel: computes x @ weight^T + bias
    # x: [M, K], weight: [N, K] -> x @ weight^T = [M, N]
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid_gemm](
        x, weight, bias, gemm_out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # B is weight^T: stride_bk=weight.stride(1), stride_bn=weight.stride(0)
        gemm_out.stride(0), gemm_out.stride(1),
    )

    # Allocate output for reduction + activations: [M, 1]
    output = torch.empty((M, 1), device=x.device, dtype=torch.float32)

    # Launch reduction kernel
    grid_red = (M,)
    logsumexp_leakyrelu_gelu_kernel[grid_red](
        gemm_out, output,
        M, N,
        gemm_out.stride(0), gemm_out.stride(1),
    )

    return output


def get_inputs():
    weight = torch.randn(8192, 8192, device='cuda', dtype=torch.float32)
    bias = torch.randn(8192, device='cuda', dtype=torch.float32)
    x = torch.rand(1024, 8192, device='cuda', dtype=torch.float32)
    return [weight, bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, bias, x = inputs

    # Verify correctness
    if not args.profile_only:
        import torch.nn.functional as F

        # Reference computation
        linear_out = F.linear(x, weight, bias)
        lse = torch.logsumexp(linear_out, dim=1, keepdim=True)
        lr1 = F.leaky_relu(lse, negative_slope=0.01)
        lr2 = F.leaky_relu(lr1, negative_slope=0.01)
        g1 = F.gelu(lr2)
        ref_out = F.gelu(g1)

        # Triton computation
        triton_out = kernel_function(weight, bias, x)

        print(f"Max diff: {(triton_out - ref_out).abs().max().item()}")
        print(f"Mean diff: {(triton_out - ref_out).abs().mean().item()}")
        assert torch.allclose(triton_out, ref_out, atol=1e-3), "Results don't match!"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")