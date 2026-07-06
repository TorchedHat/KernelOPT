import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        # 256x256 REMOVED - it maximizes register pressure to 255 regs/thread
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_transpose_kernel(
    # Pointers to matrices
    A_ptr, B_ptr, C_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides for A (shape [K, M], stride_ak=M, stride_am=1 after transpose gives A.T shape [M, K])
    stride_a_row, stride_a_col,
    # Strides for B (shape [N, K], stride_bn=K, stride_bk=1 after transpose gives B.T shape [K, N])
    stride_b_row, stride_b_col,
    # Strides for C (shape [M, N])
    stride_cm, stride_cn,
    # Tile sizes
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Computes C = A.T @ B.T = (B @ A).T using algebraic identity.

    Instead of loading transposed tiles (causing bank conflicts), we compute
    B @ A with row-major tile access, then store the result transposed into C.

    pid_m indexes N-dimension (rows of B), pid_n indexes M-dimension (cols of A).
    acc shape is [BLOCK_M, BLOCK_N] = tile of (B @ A) = tile of C.T
    Store: C[offs_rhs, offs_lhs] = acc, achieved via swapped strides.
    """
    pid = tl.program_id(0)

    # STEP 1: grid_m covers N dimension, grid_n covers M dimension
    grid_m = tl.cdiv(N, BLOCK_M)
    grid_n = tl.cdiv(M, BLOCK_N)

    # L2 cache optimization: group programs
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # STEP 2: offs_lhs indexes N dim of B, offs_rhs indexes M dim of A
    offs_lhs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # indexes N dim of B
    offs_rhs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # indexes M dim of A
    offs_k = tl.arange(0, BLOCK_K)

    # Initialize accumulator: shape [BLOCK_M, BLOCK_N] = tile of (B @ A)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        # STEP 4: Load B tile row-major: shape [BLOCK_M, BLOCK_K]
        # B[offs_lhs, k_offs] — row-major, stride_b_col=1 is innermost along K
        b_ptrs = B_ptr + offs_lhs[:, None] * stride_b_row + k_offs[None, :] * stride_b_col
        mask_b = (offs_lhs[:, None] < N) & (k_offs[None, :] < K)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0, eviction_policy='evict_first')

        # STEP 3: Load A tile row-major: shape [BLOCK_K, BLOCK_N]
        # A[k_offs, offs_rhs] — row-major, stride_a_col=1 is innermost along M
        a_ptrs = A_ptr + k_offs[:, None] * stride_a_row + offs_rhs[None, :] * stride_a_col
        mask_a = (k_offs[:, None] < K) & (offs_rhs[None, :] < M)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0, eviction_policy='evict_first')

        # STEP 6: b shape [BLOCK_M, BLOCK_K], a shape [BLOCK_K, BLOCK_N]
        # acc shape [BLOCK_M, BLOCK_N] = tile of (B @ A) = tile of C.T
        # allow_tf32=True enables Tensor Core MMA units, reducing register pressure
        acc = tl.dot(b, a, acc=acc, allow_tf32=True)

    # STEP 7 & 8: Store result — swap strides to write C.T into C's memory layout
    # C[offs_rhs, offs_lhs] = acc  =>  c_ptrs uses stride_cn for offs_lhs, stride_cm for offs_rhs
    offs_lhs = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_rhs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_lhs[:, None] * stride_cn + offs_rhs[None, :] * stride_cm
    mask_c = (offs_lhs[:, None] < N) & (offs_rhs[None, :] < M)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask_c, eviction_policy='evict_first')

def kernel_function(a, b):
    """
    Computes C = a.T @ b.T
    """
    # a has shape [K, M], b has shape [N, K]
    K_a, M = a.shape
    N, K_b = b.shape
    assert K_a == K_b, f"K dimensions must match: {K_a} vs {K_b}"
    K = K_a
    
    # Output shape: [M, N]
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Grid
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )
    
    matmul_transpose_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),  # stride_a_row=M (stride along K dim), stride_a_col=1 (stride along M dim)
        b.stride(0), b.stride(1),  # stride_b_row=K (stride along N dim), stride_b_col=1 (stride along K dim)
        c.stride(0), c.stride(1),
    )
    
    return c


def get_inputs():
    # a: shape=[8192, 2048], b: shape=[4096, 8192]
    a = torch.rand(8192, 2048, dtype=torch.float32, device='cuda')
    b = torch.rand(4096, 8192, dtype=torch.float32, device='cuda')
    return [a, b]


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
        a, b = inputs
        expected = torch.matmul(a.T, b.T)
        max_diff = (output - expected).abs().max().item()
        print(f"Max difference from reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")
        print(f"Output shape: {output.shape}")