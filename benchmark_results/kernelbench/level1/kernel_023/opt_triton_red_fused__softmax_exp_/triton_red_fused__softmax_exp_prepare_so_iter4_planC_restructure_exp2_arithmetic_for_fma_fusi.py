import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024, 'num_stages': 2}, num_warps=32),
        triton.Config({'BLOCK_N': 2048, 'num_stages': 2}, num_warps=32),
        triton.Config({'BLOCK_N': 4096, 'num_stages': 2}, num_warps=32),
        triton.Config({'BLOCK_N': 512, 'num_stages': 2}, num_warps=16),
        triton.Config({'BLOCK_N': 1024, 'num_stages': 3}, num_warps=16),
        triton.Config({'BLOCK_N': 2048, 'num_stages': 3}, num_warps=16),
        triton.Config({'BLOCK_N': 4096, 'num_stages': 3}, num_warps=16),
        triton.Config({'BLOCK_N': 8192, 'num_stages': 2}, num_warps=32),
        triton.Config({'BLOCK_N': 1024, 'num_stages': 4}, num_warps=32),
        triton.Config({'BLOCK_N': 2048, 'num_stages': 4}, num_warps=32),
        triton.Config({'BLOCK_N': 4096, 'num_stages': 4}, num_warps=32),
        triton.Config({'BLOCK_N': 8192, 'num_stages': 4}, num_warps=32),
        triton.Config({'BLOCK_N': 1024, 'num_stages': 4}, num_warps=16),
        triton.Config({'BLOCK_N': 2048, 'num_stages': 4}, num_warps=16),
        triton.Config({'BLOCK_N': 4096, 'num_stages': 4}, num_warps=16),
        triton.Config({'BLOCK_N': 8192, 'num_stages': 4}, num_warps=16),
        triton.Config({'BLOCK_N': 1024, 'num_stages': 4}, num_warps=8),
        triton.Config({'BLOCK_N': 2048, 'num_stages': 4}, num_warps=8),
        triton.Config({'BLOCK_N': 4096, 'num_stages': 4}, num_warps=8),
        triton.Config({'BLOCK_N': 8192, 'num_stages': 4}, num_warps=8),
        # New num_warps=4 configs
        triton.Config({'BLOCK_N': 1024, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_N': 2048, 'num_stages': 2}, num_warps=4),
        triton.Config({'BLOCK_N': 4096, 'num_stages': 2}, num_warps=4),
        # New num_warps=8 configs
        triton.Config({'BLOCK_N': 2048, 'num_stages': 3}, num_warps=8),
        triton.Config({'BLOCK_N': 4096, 'num_stages': 3}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(
    output_ptr,
    input_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
    num_stages: tl.constexpr,
):
    # log2(e) constant for fast exp2-based approximation
    LOG2E: tl.constexpr = 1.4426950408889634

    # Each program handles one row
    row_idx = tl.program_id(0)
    
    row_start = input_ptr + row_idx * N
    
    # Pass 1: find max and compute sum of exp(x - max) using online algorithm
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((), dtype=tl.float32)
    
    for col_off in tl.range(0, N, BLOCK_N, num_stages=num_stages):
        col_idx = col_off + tl.arange(0, BLOCK_N)
        mask = col_idx < N
        x = tl.load(row_start + col_idx, mask=mask, other=float('-inf')).to(tl.float32)
        
        # Online softmax update
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        
        # Update sum: scale old sum by exp(old_max - new_max), add new block's contribution
        # Use tl.exp2(x * log2(e)) = exp(x), maps to fast MUFU.EX2 instruction
        scale = tl.exp2((row_max - new_max) * LOG2E)
        # Pre-compute scalar offset for FMA fusion: tl.fma(x, LOG2E, neg_new_max_log2e)
        # is mathematically identical to (x - new_max) * LOG2E but allows FMA fusion
        neg_new_max_log2e = (-new_max) * LOG2E
        block_sum = tl.sum(tl.exp2(tl.fma(x, LOG2E, neg_new_max_log2e)), axis=0)
        row_sum = tl.fma(row_sum, scale, block_sum)
        row_max = new_max
    
    # Pass 2: normalize and store
    out_row_start = output_ptr + row_idx * N
    
    # Hoist reciprocal computation outside the loop to replace per-iteration FDIV with FMUL
    inv_sum = 1.0 / row_sum
    
    # Hoist scalar offset for FMA fusion outside the loop — row_max is fixed after pass 1
    neg_row_max_log2e = (-row_max) * LOG2E
    
    for col_off in tl.range(0, N, BLOCK_N, num_stages=num_stages):
        col_idx = col_off + tl.arange(0, BLOCK_N)
        mask = col_idx < N
        # Removed eviction_policy='evict_last' — rows are 1.5 MB, far exceeding L1 capacity
        x = tl.load(row_start + col_idx, mask=mask, other=float('-inf')).to(tl.float32)
        
        # tl.fma(x, LOG2E, neg_row_max_log2e) == (x - row_max) * LOG2E but FMA-fused
        out = tl.exp2(tl.fma(x, LOG2E, neg_row_max_log2e)) * inv_sum
        tl.store(out_row_start + col_idx, out.to(tl.bfloat16), mask=mask)

def kernel_function(*inputs):
    x, = inputs
    M, N = x.shape
    
    output = torch.empty_like(x)
    
    # One program per row
    grid = (M,)
    
    softmax_kernel[grid](
        output,
        x,
        M,
        N,
    )
    
    return output


def get_inputs():
    x = torch.rand(4096, 393216, device='cuda', dtype=torch.float32)
    return [x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    # Verify correctness on a small example first
    if not args.profile_only:
        x_small = torch.rand(4, 1024, device='cuda', dtype=torch.float32)
        ref = torch.softmax(x_small, dim=1)
        out = kernel_function(x_small)
        max_diff = (ref - out).abs().max().item()
        print(f"Max diff (small test): {max_diff:.6f}")
        assert max_diff < 1e-3, f"Correctness check failed: max_diff={max_diff}"
        print("Correctness check passed!")
    
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        print("Kernel executed successfully.")
        print(f"Output shape: {output.shape}")
        print(f"Output sum (first row): {output[0].sum().item():.6f} (should be ~1.0)")