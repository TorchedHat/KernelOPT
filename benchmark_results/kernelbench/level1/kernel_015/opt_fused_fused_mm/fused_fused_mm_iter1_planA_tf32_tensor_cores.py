import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_tril_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # L2 cache optimization: group programs
    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    # Row and column ranges for this tile
    row_start = pid_m * BLOCK_M
    col_start = pid_n * BLOCK_N

    # Early exit: if the entire tile is above the diagonal, output is zero
    # The tile covers rows [row_start, row_start+BLOCK_M) and cols [col_start, col_start+BLOCK_N)
    # A tile is entirely above diagonal if col_start > row_start + BLOCK_M - 1
    # i.e., the minimum row in the tile < maximum col in the tile means some elements are above diagonal
    # We only skip if col_start > row_start + BLOCK_M - 1 (entire tile above diagonal)
    rm = row_start + tl.arange(0, BLOCK_M)
    rn = col_start + tl.arange(0, BLOCK_N)

    # Compute matmul for this tile
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For lower triangular result: C[i,j] = sum_k A[i,k]*B[k,j] where i >= j
    # Since A and B are lower triangular:
    # A[i,k] = 0 if k > i
    # B[k,j] = 0 if j > k
    # So C[i,j] = sum_{k=j}^{i} A[i,k]*B[k,j]
    # But we compute the full matmul and apply tril at the end

    offs_am = rm
    offs_bn = rn
    offs_k = tl.arange(0, BLOCK_K)

    A_base = A_ptr + offs_am[:, None] * stride_am
    B_base = B_ptr + offs_bn[None, :] * stride_bn

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_start = k * BLOCK_K
        k_offs = k_start + offs_k

        # Load A tile: shape [BLOCK_M, BLOCK_K]
        a_ptrs = A_base + k_offs[None, :] * stride_ak
        mask_a = (offs_am[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)

        # Load B tile: shape [BLOCK_K, BLOCK_N]
        b_ptrs = B_base + k_offs[:, None] * stride_bk
        mask_b = (k_offs[:, None] < K) & (offs_bn[None, :] < N)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=True)

    # Apply tril: zero out elements where col > row
    row_idx = rm[:, None]  # [BLOCK_M, 1]
    col_idx = rn[None, :]  # [1, BLOCK_N]
    tril_mask = col_idx <= row_idx

    # Zero out upper triangular part
    acc = tl.where(tril_mask, acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store result
    c_ptrs = C_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    mask_c = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask_c)

def kernel_function(a, b):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "Inner dimensions must match"

    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),
    )

    matmul_tril_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )

    return c


def get_inputs():
    M = 4096
    A = torch.rand(M, M, device='cuda', dtype=torch.float32)
    B = torch.rand(M, M, device='cuda', dtype=torch.float32)
    A = torch.tril(A)
    B = torch.tril(B)
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
        a, b = inputs
        expected = torch.tril(torch.matmul(a, b))
        max_diff = (output - expected).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match.")