import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
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
        # Load A tile (FP32)
        a = tl.load(A_ptrs, mask=rk[None, :] < k_remaining, other=0.0)
        # Load B tile (FP32)
        b = tl.load(B_ptrs, mask=rk[:, None] < k_remaining, other=0.0)

        # Per-tile block scaling for FP8 precision:
        # Scale each tile to use the full FP8 E4M3 dynamic range,
        # then compensate by multiplying the partial result by scale_a * scale_b.
        # This reduces quantization error from O(K*eps) to O(BLOCK_K*eps) per tile.
        # FP8 E4M3 max representable value is 448.0
        FP8_MAX: tl.constexpr = 448.0

        # Compute per-tile max absolute values for scaling
        abs_a = tl.abs(a)
        abs_b = tl.abs(b)
        max_a = tl.max(abs_a)
        max_b = tl.max(abs_b)

        # Avoid division by zero; if tile is all zeros, scale = 1
        scale_a = tl.where(max_a > 0.0, FP8_MAX / max_a, 1.0)
        scale_b = tl.where(max_b > 0.0, FP8_MAX / max_b, 1.0)

        # Scale tiles to fill FP8 range, clamp for safety
        a_scaled = tl.clamp(a * scale_a, -FP8_MAX, FP8_MAX)
        b_scaled = tl.clamp(b * scale_b, -FP8_MAX, FP8_MAX)

        # Cast to FP8 E4M3
        a_fp8 = a_scaled.to(tl.float8e4nv)
        b_fp8 = b_scaled.to(tl.float8e4nv)

        # FP8 dot product; accumulator stays in FP32
        # Compensate for the scaling: result = dot(a/scale_a, b/scale_b)
        #                                     = dot(a_scaled, b_scaled) / (scale_a * scale_b)
        partial = tl.dot(a_fp8, b_fp8, acc=tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))
        acc = acc + partial * (1.0 / (scale_a * scale_b))

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