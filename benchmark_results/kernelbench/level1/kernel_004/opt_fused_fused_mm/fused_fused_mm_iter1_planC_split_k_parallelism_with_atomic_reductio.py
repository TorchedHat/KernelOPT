import torch
import triton
import triton.language as tl


# Matrix-vector multiplication: C[M,1] = A[M,K] @ B[K,1]
# Strategy: Each program handles one row of A (one output element).
# We tile over K dimension and accumulate dot product.
# For large K (1048576), we need to be careful about register pressure.


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 2048, 'K_SPLITS': 8},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 4096, 'K_SPLITS': 8},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 2048, 'K_SPLITS': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 4096, 'K_SPLITS': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 2048, 'K_SPLITS': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 4096, 'K_SPLITS': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 2048, 'K_SPLITS': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 1, 'BLOCK_K': 4096, 'K_SPLITS': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 2, 'BLOCK_K': 2048, 'K_SPLITS': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 4, 'BLOCK_K': 2048, 'K_SPLITS': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'K'],
    reset_to_zero=['C_ptr'],
)
@triton.jit
def matvec_kernel(
    A_ptr, B_ptr, C_ptr,
    M, K,
    stride_am, stride_ak,
    stride_bk,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_SPLITS: tl.constexpr,
):
    # Each program handles BLOCK_M rows and a K-slice
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    row_start = pid_m * BLOCK_M

    # Row offsets for this block
    row_offsets = row_start + tl.arange(0, BLOCK_M)
    row_mask = row_offsets < M

    # Compute K-slice bounds for this pid_k
    k_split_size = tl.cdiv(K, K_SPLITS)
    k_begin = pid_k * k_split_size
    k_end = tl.minimum(k_begin + k_split_size, K)

    # Accumulator: one per row in BLOCK_M
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Iterate over this K-slice in tiles
    for k_start in tl.range(k_begin, k_end, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        k_mask = k_offsets < k_end

        # Load B tile: shape [BLOCK_K]
        b_ptrs = B_ptr + k_offsets * stride_bk
        b = tl.load(b_ptrs, mask=k_mask, other=0.0)

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_ptr + row_offsets[:, None] * stride_am + k_offsets[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=row_mask[:, None] & k_mask[None, :], other=0.0)

        # Accumulate dot product
        acc += tl.sum(a * b[None, :], axis=1)

    # Atomically accumulate partial results into C
    tl.atomic_add(C_ptr + row_offsets, acc, mask=row_mask)

def kernel_function(a, b):
    M, K = a.shape
    assert b.shape == (K, 1), f"Expected b shape ({K}, 1), got {b.shape}"
    
    # Output tensor
    c = torch.empty((M, 1), device=a.device, dtype=torch.float32)
    
    # B is (K, 1), we treat it as a vector of length K
    b_vec = b.view(K)
    c_vec = c.view(M)
    
    # Grid: number of row blocks
    def grid(meta):
        return (triton.cdiv(M, meta['BLOCK_M']),)
    
    matvec_kernel[grid](
        a, b_vec, c_vec,
        M, K,
        a.stride(0), a.stride(1),
        b_vec.stride(0),
    )
    
    return c


def get_inputs():
    M = 256 * 8   # 2048
    K = 131072 * 8  # 1048576
    A = torch.rand(M, K, device='cuda', dtype=torch.float32)
    B = torch.rand(K, 1, device='cuda', dtype=torch.float32)
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
        print(f"Max difference from reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")
    else:
        # Warmup
        for _ in range(3):
            output = kernel_function(*inputs)
        torch.cuda.synchronize()
        
        import time
        N = 10
        start = time.time()
        for _ in range(N):
            output = kernel_function(*inputs)
        torch.cuda.synchronize()
        elapsed = (time.time() - start) / N * 1000
        print(f"Kernel time: {elapsed:.3f} ms")