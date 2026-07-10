import torch
import triton
import triton.language as tl

# GEMM kernel: computes out = x @ W.T + bias
# x: [M, K], W: [N, K], bias: [N], out: [M, N]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128, 'GROUP_M': 8}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
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

    x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(x, w, acc, allow_tf32=True)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Fused epilogue: pairwise max-pool (kernel_size=2) + sum + scale
    # acc shape: [BLOCK_M, BLOCK_N], BLOCK_N is always even in all configs
    # Reshape to [BLOCK_M, BLOCK_N//2, 2] to extract even/odd pairs
    BLOCK_N_HALF: tl.constexpr = BLOCK_N // 2
    acc_pairs = tl.reshape(acc, (BLOCK_M, BLOCK_N_HALF, 2))
    # Extract even (index 0) and odd (index 1) along last axis
    acc_even = tl.reshape(acc_pairs[:, :, 0], (BLOCK_M, BLOCK_N_HALF))
    acc_odd  = tl.reshape(acc_pairs[:, :, 1], (BLOCK_M, BLOCK_N_HALF))
    pooled = tl.maximum(acc_even, acc_odd)  # [BLOCK_M, BLOCK_N_HALF]

    # Mask out-of-bounds entries
    offs_n_half = pid_n * BLOCK_N_HALF + tl.arange(0, BLOCK_N_HALF)
    N_HALF = N // 2
    mask_n_half = offs_n_half[None, :] < N_HALF
    mask_m = offs_m[:, None] < M
    pooled = tl.where(mask_m & mask_n_half, pooled, 0.0)

    # Sum over pooled columns -> [BLOCK_M] partial row sums, then scale by 0.5
    row_partial = tl.sum(pooled, axis=1) * 0.5

    # Atomic accumulate into output (multiple pid_n tiles contribute to same row)
    tl.atomic_add(Out_ptr + offs_m, row_partial, mask=(offs_m < M))

def maxpool_sum_scale_kernel(
    In_ptr, Out_ptr,
    M, N, N_HALF,
    scale_factor,
    BLOCK_N: tl.constexpr,
):
    # One program per row
    row = tl.program_id(0)
    if row >= M:
        return

    row_ptr = In_ptr + row * N

    acc = tl.zeros((), dtype=tl.float32)

    # Process N_HALF pooled elements in blocks of BLOCK_N
    for start in range(0, N_HALF, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        mask = offs < N_HALF
        # Load pairs: element at 2*offs and 2*offs+1
        even_offs = 2 * offs
        odd_offs = 2 * offs + 1
        v0 = tl.load(row_ptr + even_offs, mask=mask, other=float('-inf'))
        v1 = tl.load(row_ptr + odd_offs, mask=mask, other=float('-inf'))
        pool_val = tl.maximum(v0, v1)
        # Mask out-of-bounds to 0 for sum
        pool_val = tl.where(mask, pool_val, 0.0)
        acc += tl.sum(pool_val, axis=0)

    acc = acc * scale_factor
    tl.store(Out_ptr + row, acc)


def kernel_function(p_matmul_weight, p_matmul_bias, x):
    # x: [128, 32768], W: [32768, 32768], b: [32768]
    M = x.shape[0]
    K = x.shape[1]
    N = p_matmul_weight.shape[0]  # out_features

    # Allocate GEMM output
    linear_out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    # Launch GEMM kernel
    grid_gemm = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid_gemm](
        x, p_matmul_weight, p_matmul_bias, linear_out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_matmul_weight.stride(0), p_matmul_weight.stride(1),
        linear_out.stride(0), linear_out.stride(1),
    )

    # Allocate output for final result
    out = torch.empty((M,), device=x.device, dtype=x.dtype)

    N_HALF = N // 2
    scale_factor = 0.5

    grid_pool = (M,)
    maxpool_sum_scale_kernel[grid_pool](
        linear_out, out,
        M, N, N_HALF,
        scale_factor,
    )

    return out


def get_inputs():
    p_matmul_weight = torch.randn(32768, 32768, device='cuda', dtype=torch.float32)
    p_matmul_bias = torch.randn(32768, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 32768, device='cuda', dtype=torch.float32)
    return [p_matmul_weight, p_matmul_bias, x]


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
        import torch.nn as nn
        W, b, x = inputs
        # Reference computation
        linear_ref = x @ W.T + b
        pool_ref = torch.nn.functional.max_pool1d(linear_ref.unsqueeze(1), kernel_size=2).squeeze(1)
        sum_ref = pool_ref.sum(dim=1)
        ref_out = sum_ref * 0.5

        max_diff = (output - ref_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        if max_diff < 1e-3:
            print("Correctness PASSED")
        else:
            print("Correctness FAILED")
        print("Kernel executed successfully.")