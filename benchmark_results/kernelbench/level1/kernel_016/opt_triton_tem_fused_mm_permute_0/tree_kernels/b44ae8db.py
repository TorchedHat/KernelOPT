import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64,  'GROUP_M': 8}, num_stages=5, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32,  'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_transpose_a_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    # We compute C = A^T @ B
    # A is (K, M) stored as (K, M), so A^T is (M, K)
    # B is (K, N)
    # C is (M, N)
    M, N, K,
    # Strides of A (shape K x M, stride_ak=M, stride_am=1)
    stride_ak, stride_am,
    # Strides of B (shape K x N, stride_bk=N, stride_bn=1)
    stride_bk, stride_bn,
    # Strides of C (shape M x N, stride_cm=N, stride_cn=1)
    stride_cm, stride_cn,
    # Tile sizes (autotuned)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    """
    Computes C = A^T @ B where:
    - A has shape (K, M) with strides (stride_ak=M, stride_am=1)
    - B has shape (K, N) with strides (stride_bk=N, stride_bn=1)
    - C has shape (M, N) with strides (stride_cm=N, stride_cn=1)
    
      - We load A^T[pid_m*BM:(pid_m+1)*BM, k:k+BK] = A[k:k+BK, pid_m*BM:(pid_m+1)*BM]
    """
    pid = tl.program_id(0)
    
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)
    
    # L2 cache-friendly swizzling
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size
    
    # Row/col offsets for this tile
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    # Pointers into A: A has shape (K, M), stride (stride_ak, stride_am)
    # A^T[m, k] = A[k, m]
    # For tile: A^T[offs_m, offs_k] = A[offs_k, offs_m]
    # a_ptrs[bm, bk] = a_ptr + offs_k[bk] * stride_ak + offs_m[bm] * stride_am
    a_ptrs = a_ptr + (offs_k[None, :] * stride_ak + offs_m[:, None] * stride_am)
    
    # Pointers into B: B has shape (K, N), stride (stride_bk, stride_bn)
    # b_ptrs[bk, bn] = b_ptr + offs_k[bk] * stride_bk + offs_n[bn] * stride_bn
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        # Boundary masks
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K
        
        # Load A tile: shape (BLOCK_M, BLOCK_K)
        a_mask = (offs_m[:, None] < M) & (mask_k[None, :])
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        
        # Load B tile: shape (BLOCK_K, BLOCK_N)
        b_mask = (mask_k[:, None]) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        # Accumulate
        acc = tl.dot(a, b, acc=acc, allow_tf32=False)
        
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Write output
    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)

def kernel_function(a, b):
    """
    Computes C = A^T @ B
    a: shape (K, M) = (8192, 2048)
    b: shape (K, N) = (8192, 4096)
    returns c: shape (M, N) = (2048, 4096)
    """
    # A is (K, M), B is (K, N)
    # C = A^T @ B has shape (M, N)
    K_a, M = a.shape
    K_b, N = b.shape
    assert K_a == K_b, f"K dimensions must match: {K_a} != {K_b}"
    K = K_a
    
    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    # Grid: one program per (BLOCK_M, BLOCK_N) tile of C
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    
    matmul_transpose_a_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),  # stride_ak, stride_am
        b.stride(0), b.stride(1),  # stride_bk, stride_bn
        c.stride(0), c.stride(1),  # stride_cm, stride_cn
    )
    
    return c


def get_inputs():
    # a: shape (K, M) = (8192, 2048)
    # b: shape (K, N) = (8192, 4096)
    a = torch.rand(8192, 2048, device='cuda', dtype=torch.float32)
    b = torch.rand(8192, 4096, device='cuda', dtype=torch.float32)
    return [a, b]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    a, b = inputs
    
    # Run our kernel
    output = kernel_function(a, b)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        # Verify correctness
        expected = torch.matmul(a.T, b)
        max_diff = (output - expected).abs().max().item()
        print(f"Max absolute difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print(f"Output shape: {output.shape}")
        print(f"Expected shape: {expected.shape}")
        print("Kernel executed successfully.")
        
        # Simple benchmark
        import time
        torch.cuda.synchronize()
        N_WARMUP = 5
        N_ITER = 20
        for _ in range(N_WARMUP):
            kernel_function(a, b)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(N_ITER):
            kernel_function(a, b)
        torch.cuda.synchronize()
        end = time.time()
        
        elapsed_ms = (end - start) / N_ITER * 1000
        print(f"Triton kernel: {elapsed_ms:.3f} ms per iteration")
        
        # Compare with PyTorch
        for _ in range(N_WARMUP):
            torch.matmul(a.T, b)
        torch.cuda.synchronize()
        
        start = time.time()
        for _ in range(N_ITER):
            torch.matmul(a.T, b)
        torch.cuda.synchronize()
        end = time.time()
        
        pt_elapsed_ms = (end - start) / N_ITER * 1000
        print(f"PyTorch matmul: {pt_elapsed_ms:.3f} ms per iteration")
        print(f"Speedup: {pt_elapsed_ms / elapsed_ms:.2f}x")