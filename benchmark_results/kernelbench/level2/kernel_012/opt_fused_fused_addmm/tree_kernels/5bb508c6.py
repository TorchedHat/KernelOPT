import torch
import triton
import triton.language as tl


# Epilogue kernel: fuses bias_add + mul(2.0) + leaky_relu(0.1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['n_cols'],
)
@triton.jit
def epilogue_kernel(
    M,
    N,
    n_cols,        # = N (used as autotune key)
    multiplier,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N

    # Load mm result
    mm_val = tl.load(out_ptr + offsets, mask=mask, other=0.0)

    # Load bias (broadcast over rows)
    col_idx = offsets % N
    bias_val = tl.load(bias_ptr + col_idx, mask=mask, other=0.0)

    # bias add
    x = mm_val + bias_val

    # multiply
    x = x * multiplier

    # leaky relu
    pos = x > 0.0
    x_neg = x * negative_slope
    x = tl.where(pos, x, x_neg)

    tl.store(out_ptr + offsets, x, mask=mask)


# GEMM kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit

def matmul_epilogue_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    multiplier,
    negative_slope,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache-friendly swizzling
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Pointers
    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)

        a = tl.load(A_ptrs, mask=a_mask, other=0.0)
        b = tl.load(B_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=False)

        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Load bias and apply epilogue
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    # bias add
    acc = acc + bias[None, :]

    # multiply
    acc = acc * multiplier

    # leaky relu
    pos = acc > 0.0
    acc_neg = acc * negative_slope
    acc = tl.where(pos, acc, acc_neg)

    # Store
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float32), mask=c_mask)


def kernel_function(p_gemm_weight, p_gemm_bias, x):
    # x: [M, K], weight: [N, K] (linear weight is transposed), bias: [N]
    M, K = x.shape
    N = p_gemm_weight.shape[0]

    # Weight needs to be transposed for matmul: x @ weight.T
    # A = x [M, K], B = weight.T [K, N]
    # We pass weight as [N, K] and use strides to treat it as [K, N]
    C = torch.empty((M, N), device=x.device, dtype=torch.float32)

    multiplier = 2.0
    negative_slope = 0.1

    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    matmul_epilogue_kernel[grid](
        x, p_gemm_weight, p_gemm_bias, C,
        M, N, K,
        x.stride(0), x.stride(1),
        p_gemm_weight.stride(1), p_gemm_weight.stride(0),  # treat weight as transposed
        C.stride(0), C.stride(1),
        multiplier,
        negative_slope,
    )

    return C


def get_inputs():
    p_gemm_weight = torch.randn(8192, 8192, dtype=torch.float32, device='cuda')
    p_gemm_bias = torch.randn(8192, dtype=torch.float32, device='cuda')
    x = torch.rand(1024, 8192, dtype=torch.float32, device='cuda')
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
        import torch.nn as nn

        p_gemm_weight, p_gemm_bias, x = inputs

        # Reference computation
        linear = nn.Linear(8192, 8192)
        linear.weight = nn.Parameter(p_gemm_weight)
        linear.bias = nn.Parameter(p_gemm_bias)
        linear = linear.cuda()

        with torch.no_grad():
            ref = linear(x)
            ref = ref * 2.0
            ref = torch.where(ref > 0, ref, ref * 0.1)

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.6e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")