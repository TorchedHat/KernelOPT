import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M':  32, 'BLOCK_N': 128, 'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  32, 'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
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

    # L2 cache-friendly reordering
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Compute row/col offsets
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rk = tl.arange(0, BLOCK_K)

    # Boundary-safe offsets for A and B
    offs_am = tl.max_contiguous(tl.multiple_of(rm % M, BLOCK_M), BLOCK_M)
    offs_bn = tl.max_contiguous(tl.multiple_of(rn % N, BLOCK_N), BLOCK_N)

    # Pointers to A and B tiles
    A_ptrs = A_ptr + (offs_am[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptrs = B_ptr + (rk[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Accumulate in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        # Load A tile
        a = tl.load(A_ptrs, mask=rk[None, :] < k_remaining, other=0.0)
        # Load B tile
        b = tl.load(B_ptrs, mask=rk[:, None] < k_remaining, other=0.0)
        # Cast to FP16 for HMMA Tensor Core path
        a = a.to(tl.float16)
        b = b.to(tl.float16)
        # Accumulate: FP16 inputs + FP32 acc -> HMMA.16816
        acc = tl.dot(a, b, acc=acc)
        # Advance pointers
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Write output
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (rm[:, None] < M) & (rn[None, :] < N)
    C_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)

def kernel_function(a, b):
    # Ensure inputs are on CUDA and float32
    assert a.is_cuda and b.is_cuda
    assert a.dtype == torch.float32 and b.dtype == torch.float32

    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    # Grid: one program per output tile
    grid = lambda META: (
        triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),
    )

    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


def get_inputs():
    M = 2048
    K = 8192
    N = 4096
    A = torch.rand(M, K, device='cuda', dtype=torch.float32)
    B = torch.rand(K, N, device='cuda', dtype=torch.float32)
    return [A, B]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify correctness
        ref = torch.matmul(inputs[0], inputs[1])
        max_diff = (output - ref).abs().max().item()
        print(f"Max absolute difference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match PyTorch.")