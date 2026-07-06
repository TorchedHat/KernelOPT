import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 64,   'BLOCK_X': 32}, num_warps=4),
        triton.Config({'BLOCK_R': 128,  'BLOCK_X': 32}, num_warps=4),
        triton.Config({'BLOCK_R': 256,  'BLOCK_X': 16}, num_warps=4),
        triton.Config({'BLOCK_R': 256,  'BLOCK_X': 32}, num_warps=8),
        triton.Config({'BLOCK_R': 512,  'BLOCK_X': 16}, num_warps=8),
        triton.Config({'BLOCK_R': 512,  'BLOCK_X': 32}, num_warps=8),
        triton.Config({'BLOCK_R': 1024, 'BLOCK_X': 16}, num_warps=8),
        triton.Config({'BLOCK_R': 1024, 'BLOCK_X': 32}, num_warps=16),
    ],
    key=['R', 'N'],
)
@triton.jit
def argmax_dim1_kernel(
    x_ptr,
    out_ptr,
    B,       # batch size = 128
    R,       # reduction dim = 4096
    N,       # last dim = 4095
    BLOCK_R: tl.constexpr,
    BLOCK_X: tl.constexpr,
):
    # Each program handles one (batch, col_tile) pair
    pid = tl.program_id(0)
    num_col_tiles = tl.cdiv(N, BLOCK_X)
    batch_idx = pid // num_col_tiles
    col_tile  = pid %  num_col_tiles
    col_start = col_tile * BLOCK_X

    # Column offsets for this tile — contiguous in memory (innermost dim)
    col_offsets = col_start + tl.arange(0, BLOCK_X)   # [BLOCK_X]
    col_mask    = col_offsets < N                       # [BLOCK_X]

    # Base pointer for this batch
    batch_base = x_ptr + batch_idx * R * N

    # Initialize max value and argmax index per column
    max_val = tl.full([BLOCK_X], float('-inf'), dtype=tl.float32)
    max_idx = tl.full([BLOCK_X], 0, dtype=tl.int64)

    # Iterate over reduction dimension in blocks
    for r_start in tl.range(0, R, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)   # [BLOCK_R]
        r_mask    = r_offsets < R                       # [BLOCK_R]

        # 2D pointer array: [BLOCK_R, BLOCK_X]
        # r_offsets[:, None] * N gives row strides; col_offsets[None, :] gives col offsets
        # Innermost (col) dimension is contiguous — achieves full coalescing
        ptrs = batch_base + r_offsets[:, None] * N + col_offsets[None, :]
        mask = r_mask[:, None] & col_mask[None, :]     # [BLOCK_R, BLOCK_X]

        vals = tl.load(ptrs, mask=mask, other=float('-inf')).to(tl.float32)  # [BLOCK_R, BLOCK_X]

        # Find local max across reduction dim for each column
        local_max = tl.max(vals, axis=0)               # [BLOCK_X]

        # Find first index of local max (for tie-breaking)
        # is_max: [BLOCK_R, BLOCK_X]
        is_max = (vals == local_max[None, :]) & mask
        # indices: use R as sentinel for non-max positions
        indices = tl.where(
            is_max,
            tl.broadcast_to(r_offsets[:, None], [BLOCK_R, BLOCK_X]).to(tl.int64),
            tl.full([BLOCK_R, BLOCK_X], R, dtype=tl.int64)
        )
        local_idx = tl.min(indices, axis=0)            # [BLOCK_X]

        # Update global max and argmax
        update = local_max > max_val                   # [BLOCK_X]
        tie    = local_max == max_val                  # [BLOCK_X]
        max_val = tl.where(update, local_max, max_val)
        max_idx = tl.where(update, local_idx,
                  tl.where(tie, tl.minimum(max_idx, local_idx), max_idx))

    # Store results for this column tile
    out_ptrs = out_ptr + batch_idx * N + col_offsets   # [BLOCK_X]
    tl.store(out_ptrs, max_idx, mask=col_mask)

def kernel_function(x):
    B, R, N = x.shape  # 128, 4096, 4095
    out = torch.empty((B, N), dtype=torch.int64, device=x.device)
    
    grid = (B * N,)
    argmax_dim1_kernel[grid](
        x, out,
        B, R, N,
    )
    return out


def get_inputs():
    x = torch.rand(128, 4096, 4095, device='cuda')
    return [x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    # Verify correctness
    if not args.profile_only:
        x = inputs[0]
        out_triton = kernel_function(x)
        out_torch = torch.argmax(x, dim=1)
        
        matches = (out_triton == out_torch).all()
        print(f"Results match: {matches}")
        if not matches:
            diff = (out_triton != out_torch).sum()
            print(f"Number of mismatches: {diff}")
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")