import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
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
    Computes C = A.T @ B.T where:
    
    A.T[m, k] = A[k, m] -> access A at row=k, col=m
    B.T[k, n] = B[n, k] -> access B at row=n, col=k
    """
    pid = tl.program_id(0)
    
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 cache optimization: group programs
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    # Row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # A has shape [K, M]: A.T[m, k] = A[k, m]
    # For a tile: A_tile[BLOCK_M, BLOCK_K] where A_tile[i, j] = A[offs_k[j], offs_m[i]]
    # Pointer: A_ptr + offs_k[j] * stride_a_row + offs_m[i] * stride_a_col
    
    # B has shape [N, K]: B.T[k, n] = B[n, k]
    # For a tile: B_tile[BLOCK_K, BLOCK_N] where B_tile[j, l] = B[offs_n[l], offs_k[j]]
    # Pointer: B_ptr + offs_n[l] * stride_b_row + offs_k[j] * stride_b_col
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        # Load A tile: shape [BLOCK_M, BLOCK_K]
        # A.T[offs_m, k_offs] = A[k_offs, offs_m]
        a_ptrs = A_ptr + k_offs[None, :] * stride_a_row + offs_m[:, None] * stride_a_col
        mask_a = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        
        # Load B tile: shape [BLOCK_K, BLOCK_N]
        # B.T[k_offs, offs_n] = B[offs_n, k_offs]
        b_ptrs = B_ptr + offs_n[None, :] * stride_b_row + k_offs[:, None] * stride_b_col
        mask_b = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc = tl.dot(a, b, acc=acc, allow_tf32=False)
    
    # Store result
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask_c)


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