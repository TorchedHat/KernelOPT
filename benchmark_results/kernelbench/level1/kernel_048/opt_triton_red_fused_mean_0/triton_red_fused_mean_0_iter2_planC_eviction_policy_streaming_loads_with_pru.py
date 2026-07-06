import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_COL': 32,  'BLOCK_RED': 32},  num_warps=4),
        triton.Config({'BLOCK_COL': 32,  'BLOCK_RED': 64},  num_warps=4),
        triton.Config({'BLOCK_COL': 32,  'BLOCK_RED': 128}, num_warps=4),
        triton.Config({'BLOCK_COL': 64,  'BLOCK_RED': 16},  num_warps=4),
        triton.Config({'BLOCK_COL': 64,  'BLOCK_RED': 32},  num_warps=4),
        triton.Config({'BLOCK_COL': 64,  'BLOCK_RED': 64},  num_warps=4),
        triton.Config({'BLOCK_COL': 64,  'BLOCK_RED': 64},  num_warps=8),
        triton.Config({'BLOCK_COL': 128, 'BLOCK_RED': 16},  num_warps=4),
        triton.Config({'BLOCK_COL': 128, 'BLOCK_RED': 32},  num_warps=4),
        triton.Config({'BLOCK_COL': 128, 'BLOCK_RED': 32},  num_warps=8),
        triton.Config({'BLOCK_COL': 128, 'BLOCK_RED': 64},  num_warps=8),
        triton.Config({'BLOCK_COL': 256, 'BLOCK_RED': 16},  num_warps=8),
        triton.Config({'BLOCK_COL': 256, 'BLOCK_RED': 32},  num_warps=8),
    ],
    key=['N_BATCH', 'N_COLS', 'N_RED'],
)
@triton.jit
def mean_dim1_kernel(
    x_ptr, out_ptr,
    N_BATCH, N_COLS, N_RED,
    stride_b, stride_r, stride_c,
    BLOCK_COL: tl.constexpr, BLOCK_RED: tl.constexpr,
):
    # Each program handles BLOCK_COL output columns for one batch element
    pid = tl.program_id(0)
    n_col_blocks = tl.cdiv(N_COLS, BLOCK_COL)
    
    batch_idx = pid // n_col_blocks
    col_block_idx = pid % n_col_blocks
    
    col_start = col_block_idx * BLOCK_COL
    col_offsets = col_start + tl.arange(0, BLOCK_COL)
    col_offsets = tl.max_contiguous(tl.multiple_of(col_offsets, BLOCK_COL), BLOCK_COL)
    col_mask = col_offsets < N_COLS
    
    # Accumulator: shape [BLOCK_COL]
    acc = tl.zeros([BLOCK_COL], dtype=tl.float32)
    
    # Base pointer for this batch
    batch_base = batch_idx * stride_b
    
    for r_start in range(0, N_RED, BLOCK_RED):
        r_offsets = r_start + tl.arange(0, BLOCK_RED)
        r_offsets = tl.multiple_of(r_offsets, BLOCK_RED)
        r_mask = r_offsets < N_RED
        
        # Load [BLOCK_RED, BLOCK_COL] tile
        # x[batch_idx, r, c] = x_ptr + batch_idx*stride_b + r*stride_r + c*stride_c
        ptrs = (batch_base + 
                r_offsets[:, None] * stride_r + 
                col_offsets[None, :] * stride_c)
        mask = r_mask[:, None] & col_mask[None, :]
        tile = tl.load(x_ptr + ptrs, mask=mask, other=0.0, eviction_policy='evict_first')
        acc += tl.sum(tile, axis=0)
    
    acc = acc / N_RED
    
    out_offsets = batch_idx * N_COLS + col_offsets
    tl.store(out_ptr + out_offsets, acc, mask=col_mask, eviction_policy='evict_first')

def kernel_function(x):
    N_BATCH, N_RED, N_COLS = x.shape
    out = torch.empty((N_BATCH, N_COLS), dtype=torch.float32, device=x.device)
    
    stride_b, stride_r, stride_c = x.stride()
    
    BLOCK_COL_val = 1  # will be autotuned
    n_col_blocks = (N_COLS + BLOCK_COL_val - 1) // BLOCK_COL_val
    
    # Grid: one program per (batch, col_block)
    def grid(meta):
        return (N_BATCH * triton.cdiv(N_COLS, meta['BLOCK_COL']),)
    
    mean_dim1_kernel[grid](
        x, out,
        N_BATCH, N_COLS, N_RED,
        stride_b, stride_r, stride_c,
    )
    
    return out


def get_inputs():
    x = torch.rand(128, 4096, 4095, device='cuda', dtype=torch.float32)
    return [x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    # Verify correctness
    if not args.profile_only:
        ref = torch.mean(inputs[0], dim=1)
        out = kernel_function(*inputs)
        torch.cuda.synchronize()
        max_diff = (out - ref).abs().max().item()
        print(f"Max diff: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")