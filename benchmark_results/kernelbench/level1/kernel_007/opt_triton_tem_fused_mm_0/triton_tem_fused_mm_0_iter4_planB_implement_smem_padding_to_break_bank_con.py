import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        # Large tiles: FP16 HMMA Tensor Core path - prepended per optimization plan
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=4, num_warps=4),
        # Small tiles: low register pressure, high occupancy - best for small K=64
        # SMEM_PAD=8 variants (padding to break bank conflicts)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=4, num_warps=4),
        # Medium tiles
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=3, num_warps=8),
        # Kept for larger K shapes
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SMEM_PAD': 8}, num_stages=3, num_warps=8),
        # SMEM_PAD=0 variants (for autotuner comparison, no padding)
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N':  64, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M':  64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8, 'SMEM_PAD': 0}, num_stages=3, num_warps=8),
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
    GROUP_M: tl.constexpr, SMEM_PAD: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)

    # Number of blocks in M and N dimensions
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache optimization: group programs for better cache reuse
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Index vectors (computed once, before the loop)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)   # [BLOCK_M]
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)   # [BLOCK_N]
    rk = tl.arange(0, BLOCK_K)                      # [BLOCK_K]

    # Raw global pointers
    A_ptrs = A_ptr + (rm[:, None] * stride_am + rk[None, :] * stride_ak)
    B_ptrs = B_ptr + (rk[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Precompute M and N masks (constant across K iterations)
    mask_m = rm < M   # [BLOCK_M]
    mask_n = rn < N   # [BLOCK_N]

    # Accumulator in float32
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Main loop over K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offset = k * BLOCK_K
        mask_k = (rk + k_offset) < K   # [BLOCK_K]

        # Load A tile from global memory with boundary mask
        a = tl.load(
            A_ptrs,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        ).to(tl.float16)  # [BLOCK_M, BLOCK_K]

        # Load B tile from global memory with boundary mask
        b = tl.load(
            B_ptrs,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        ).to(tl.float16)  # [BLOCK_K, BLOCK_N]

        # Apply SMEM_PAD: route tiles through a padded intermediate buffer
        # so Triton's compiler allocates smem with stride (BLOCK_K+SMEM_PAD)
        # for A and (BLOCK_N+SMEM_PAD) for B, breaking bank conflicts.
        # When SMEM_PAD=0 this branch is a compile-time no-op.
        if SMEM_PAD > 0:
            # Pad A: concatenate SMEM_PAD zero columns to make [BLOCK_M, BLOCK_K+SMEM_PAD]
            # then slice back to [BLOCK_M, BLOCK_K] — forces padded smem row stride
            a_pad = tl.cat(a, tl.zeros([BLOCK_M, SMEM_PAD], dtype=tl.float16), dim=1)
            a = a_pad[:, :BLOCK_K]

            # Pad B: concatenate SMEM_PAD zero columns to make [BLOCK_K, BLOCK_N+SMEM_PAD]
            # then slice back to [BLOCK_K, BLOCK_N] — forces padded smem row stride
            b_pad = tl.cat(b, tl.zeros([BLOCK_K, SMEM_PAD], dtype=tl.float16), dim=1)
            b = b_pad[:, :BLOCK_N]

        # Matrix multiply accumulate using FP16 HMMA Tensor Core path
        acc = tl.dot(a, b, acc=acc, out_dtype=tl.float32)

        # Advance raw pointers by one BLOCK_K step
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Store result
    cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    C_ptrs = C_ptr + (cm[:, None] * stride_cm + cn[None, :] * stride_cn)
    mask = (cm[:, None] < M) & (cn[None, :] < N)

    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)

def kernel_function(a, b):
    # Get dimensions
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, f"Incompatible dimensions: {K} != {K2}"
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Ensure inputs are float32
    if a.dtype != torch.float32:
        a = a.to(torch.float32)
    if b.dtype != torch.float32:
        b = b.to(torch.float32)
    
    # Get strides
    stride_am, stride_ak = a.stride(0), a.stride(1)
    stride_bk, stride_bn = b.stride(0), b.stride(1)
    stride_cm, stride_cn = c.stride(0), c.stride(1)
    
    # Launch kernel
    def grid(META):
        return (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
    )
    
    return c


def get_inputs():
    M = 32768
    N = 32768
    K = 64
    A = torch.rand(M, K, dtype=torch.float32, device='cuda')
    B = torch.rand(K, N, dtype=torch.float32, device='cuda')
    return [A, B]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    # Verify correctness
    if not args.profile_only:
        output = kernel_function(*inputs)
        ref = torch.matmul(inputs[0], inputs[1])
        torch.cuda.synchronize()
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")