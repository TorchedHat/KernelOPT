import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
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
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < k_remaining), other=0.0)
        b = tl.load(B_ptrs, mask=(offs_k[:, None] < k_remaining) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_stages=2, num_warps=16),
        triton.Config({'BLOCK_N': 256}, num_stages=3, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def fused_reduction_kernel(
    mm_ptr, bias_ptr, out_ptr,
    M, N,
    scale_factor,
    clamp_min, clamp_max,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    # Pass 1: compute max for numerically stable logsumexp
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        mm_val = tl.load(mm_ptr + row * N + cols, mask=mask, other=0.0)
        bias_val = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        # linear + bias, then *2.0, then +self (= *4.0 total)
        val = (mm_val + bias_val) * scale_factor
        val = val + val  # x + x
        val = tl.maximum(val, clamp_min)
        val = tl.minimum(val, clamp_max)
        local_max = tl.max(tl.where(mask, val, float('-inf')), axis=0)
        row_max = tl.maximum(row_max, local_max)

    # Handle inf max
    abs_max = tl.abs(row_max)
    row_max_safe = tl.where(abs_max == float('inf'), 0.0, row_max)

    # Pass 2: compute sum of exp(x - max)
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        mm_val = tl.load(mm_ptr + row * N + cols, mask=mask, other=0.0)
        bias_val = tl.load(bias_ptr + cols, mask=mask, other=0.0)
        val = (mm_val + bias_val) * scale_factor
        val = val + val
        val = tl.maximum(val, clamp_min)
        val = tl.minimum(val, clamp_max)
        exp_val = libdevice.exp(val - row_max_safe)
        exp_val = tl.where(mask, exp_val, 0.0)
        row_sum += tl.sum(exp_val, axis=0)

    # logsumexp = log(sum) + max_safe
    lse = libdevice.log(row_sum) + row_max_safe

    # Mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)) if x <= 20, else x
    gt20 = lse > 20.0
    exp_lse = libdevice.exp(lse)
    softplus = tl.where(gt20, lse, libdevice.log1p(exp_lse))
    tanh_sp = libdevice.tanh(softplus)
    mish_val = lse * tanh_sp

    # Final: lse * mish(lse)
    result = lse * mish_val

    tl.store(out_ptr + row, result)


def kernel_function(weight, bias, x):
    # weight: [8192, 8192], bias: [8192], x: [1024, 8192]
    M, K = x.shape
    N = weight.shape[0]  # output features

    # Allocate matmul output
    mm_out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Launch matmul: x @ weight.T
    # A = x (M, K), B = weight (N, K) -> need B transposed as (K, N)
    grid_mm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_kernel[grid_mm](
        x, weight, mm_out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight is (N, K), we want (K, N) access
        mm_out.stride(0), mm_out.stride(1),
    )

    # Allocate output: (M, 1)
    out = torch.empty((M, 1), device=x.device, dtype=x.dtype)

    # Launch fused reduction
    scale_factor = 2.0
    clamp_min = -10.0
    clamp_max = 10.0

    grid_red = (M,)
    fused_reduction_kernel[grid_red](
        mm_out, bias, out,
        M, N,
        scale_factor,
        clamp_min, clamp_max,
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
        # Reference computation
        import torch.nn.functional as F
        mm = x @ weight.T + bias
        mm = mm * 2.0
        mm = mm + mm
        mm = torch.clamp(mm, -10.0, 10.0)
        lse = torch.logsumexp(mm, dim=1, keepdim=True)
        mish_val = lse * F.mish(lse)
        ref = lse * mish_val

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")