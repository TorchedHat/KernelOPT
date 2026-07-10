import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# GEMM kernel: computes X @ W^T (no bias) -> [M, N]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(C_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, acc, mask=mask)


# Fused kernel: bias_add + GELU + Softmax (two-pass over columns)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64},  num_stages=2, num_warps=2),
        triton.Config({'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 64},  num_stages=1, num_warps=2),
        triton.Config({'BLOCK_N': 128}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_stages=1, num_warps=8),
    ],
    key=['N'],
)
@triton.jit

def bias_gelu_softmax_kernel(
    X_ptr, bias_ptr, out_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= M:
        return

    row_start_x = X_ptr + row_idx * stride_xm
    row_start_o = out_ptr + row_idx * stride_om

    # Pass 1: compute GELU(x + bias) and find max + sum for softmax
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((), dtype=tl.float32)

    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_start_x + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        val = x + b
        # GELU: 0.5 * val * (1 + erf(val * 0.7071067811865476))
        gelu_val = 0.5 * val * (1.0 + libdevice.erf(val * 0.7071067811865476))
        # Online softmax update
        block_max = tl.max(tl.where(mask, gelu_val, float('-inf')), axis=0)
        new_max = tl.maximum(row_max, block_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.where(mask, tl.exp(gelu_val - new_max), 0.0), axis=0)
        row_max = new_max

    # Pass 2: compute softmax output
    for col_off in range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_start_x + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        val = x + b
        gelu_val = 0.5 * val * (1.0 + libdevice.erf(val * 0.7071067811865476))
        softmax_val = tl.exp(gelu_val - row_max) / row_sum
        tl.store(row_start_o + cols * stride_on, softmax_val.to(tl.float32), mask=mask)


def kernel_function(weight, bias, x):
    # weight: [out_features, in_features] = [8192, 8192]
    # bias: [out_features] = [8192]
    # x: [batch, in_features] = [1024, 8192]
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2

    # Step 1: GEMM: x @ weight^T -> [M, N]
    mm_out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid_gemm](
        x, weight, mm_out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # B is weight^T: stride_bk=weight.stride(1), stride_bn=weight.stride(0)
        mm_out.stride(0), mm_out.stride(1),
    )

    # Step 2: Fused bias_add + GELU + Softmax
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid_softmax = (M,)
    bias_gelu_softmax_kernel[grid_softmax](
        mm_out, bias, out,
        M, N,
        mm_out.stride(0), mm_out.stride(1),
        out.stride(0), out.stride(1),
    )

    return out


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
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        weight, bias, x = inputs
        # Reference: linear -> gelu -> softmax
        import torch.nn.functional as F
        linear_out = F.linear(x, weight, bias)
        gelu_out = F.gelu(linear_out)
        ref = F.softmax(gelu_out, dim=1)

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")