import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32768, 'XBLOCK': 1}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_N': 16384, 'XBLOCK': 1}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_N': 8192, 'XBLOCK': 1}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_N': 4096, 'XBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_N': 2048, 'XBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1024, 'XBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'XBLOCK': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'XBLOCK': 1}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_N': 128, 'XBLOCK': 1}, num_warps=1, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def reverse_cumsum_kernel(
    in_ptr,
    out_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    # Each program handles one row
    row_id = tl.program_id(0)
    
    if row_id >= M:
        return
    
    row_base_in = in_ptr + row_id * N
    row_base_out = out_ptr + row_id * N
    
    # Number of blocks
    num_blocks = (N + BLOCK_N - 1) // BLOCK_N
    
    # We'll accumulate the running sum from the right
    running_sum = tl.zeros([1], dtype=tl.float32)
    
    # Process blocks from right to left
    for block_idx in range(num_blocks - 1, -1, -1):
        block_start = block_idx * BLOCK_N
        
        # Reverse the values within the block (single load, eliminates duplicate)
        rev_offsets = (BLOCK_N - 1) - tl.arange(0, BLOCK_N)
        actual_rev_offsets = block_start + rev_offsets
        rev_mask = actual_rev_offsets < N
        rev_vals = tl.load(row_base_in + actual_rev_offsets, mask=rev_mask, other=0.0).to(tl.float32)
        
        # Compute prefix cumsum on reversed values
        rev_cumsum = tl.cumsum(rev_vals, axis=0)
        
        # Add running_sum to all elements
        rev_cumsum = rev_cumsum + running_sum
        
        # Update running_sum: total sum of this block (sum is order-invariant)
        block_total = tl.sum(rev_vals, axis=0)
        running_sum = running_sum + block_total
        
        # Store back: position block_start + (BLOCK_N-1-k) gets rev_cumsum[k]
        store_offsets = block_start + rev_offsets
        store_mask = store_offsets < N
        tl.store(row_base_out + store_offsets, rev_cumsum.to(tl.float32), mask=store_mask)

def kernel_function(x):
    assert x.is_cuda and x.dtype == torch.float32
    M, N = x.shape
    out = torch.empty_like(x)
    
    grid = lambda meta: (M,)
    
    reverse_cumsum_kernel[grid](
        x, out,
        M, N,
    )
    
    return out


def get_inputs():
    return [torch.rand(32768, 32768, device='cuda', dtype=torch.float32)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    # Verify correctness on a small example first
    x_small = torch.rand(4, 8, device='cuda', dtype=torch.float32)
    ref = torch.cumsum(x_small.flip(1), dim=1).flip(1)
    out = kernel_function(x_small)
    
    if not args.profile_only:
        print(f"Max diff (small): {(ref - out).abs().max().item()}")
        assert torch.allclose(ref, out, atol=1e-3), f"Mismatch! Max diff: {(ref - out).abs().max().item()}"
        print("Small test passed!")
    
    # Run on full inputs
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        # Verify against PyTorch
        ref_full = torch.cumsum(inputs[0].flip(1), dim=1).flip(1)
        max_diff = (ref_full - output).abs().max().item()
        print(f"Max diff (full): {max_diff}")
        assert max_diff < 1e-3, f"Mismatch on full input! Max diff: {max_diff}"
        print("Full test passed!")
        print("Kernel executed successfully.")