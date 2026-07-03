import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 32,  'BLOCK_X': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 32,  'BLOCK_X': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 64,  'BLOCK_X': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 64,  'BLOCK_X': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 128, 'BLOCK_X': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 128, 'BLOCK_X': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 16,  'BLOCK_X': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 32,  'BLOCK_X': 128}, num_warps=8, num_stages=2),
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
    # Each program handles one (batch, col_block) pair
    # BLOCK_X = number of consecutive columns per CTA (column tile size)
    pid = tl.program_id(0)
    num_col_blocks = tl.cdiv(N, BLOCK_X)
    batch_idx = pid // num_col_blocks
    col_block = pid % num_col_blocks
    col_base = col_block * BLOCK_X

    # Column offsets and mask for this block
    col_offs = col_base + tl.arange(0, BLOCK_X)   # [BLOCK_X]
    col_mask = col_offs < N                         # [BLOCK_X]

    # Base pointer for this batch
    batch_base = x_ptr + batch_idx * R * N

    # Initialize per-column max value and argmax index
    max_val = tl.full([BLOCK_X], float('-inf'), dtype=tl.float32)
    max_idx = tl.zeros([BLOCK_X], dtype=tl.int64)

    # Iterate over reduction dimension in blocks
    for r_start in tl.range(0, R, BLOCK_R):
        r_offs = r_start + tl.arange(0, BLOCK_R)   # [BLOCK_R]
        r_mask = r_offs < R                          # [BLOCK_R]

        # Build 2D pointer array: [BLOCK_R, BLOCK_X]
        # x[batch_idx, r_offs[i], col_offs[j]] = batch_base + r_offs[i]*N + col_offs[j]
        ptrs = batch_base + tl.expand_dims(r_offs, 1) * N + tl.expand_dims(col_offs, 0)
        mask2d = tl.expand_dims(r_mask, 1) & tl.expand_dims(col_mask, 0)  # [BLOCK_R, BLOCK_X]

        # Coalesced load: innermost dim is col (stride-1 in memory)
        tile = tl.load(ptrs, mask=mask2d, other=float('-inf'))  # [BLOCK_R, BLOCK_X]
        tile = tile.to(tl.float32)

        # Per-column local max over BLOCK_R rows
        local_max = tl.max(tile, axis=0)   # [BLOCK_X]

        # Find smallest index achieving local_max per column
        # is_max: [BLOCK_R, BLOCK_X]
        is_max = (tile == tl.expand_dims(local_max, 0)) & tl.expand_dims(r_mask, 1)

        # cand_idx: use R as sentinel for non-max positions
        r_offs_2d = tl.broadcast_to(tl.expand_dims(r_offs.to(tl.int64), 1), [BLOCK_R, BLOCK_X])
        sentinel = tl.full([BLOCK_R, BLOCK_X], R, dtype=tl.int64)
        cand_idx = tl.where(is_max, r_offs_2d, sentinel)
        local_idx = tl.min(cand_idx, axis=0)   # [BLOCK_X]

        # Merge into running max per column
        update_gt = local_max > max_val
        update_eq = (local_max == max_val) & (local_idx < max_idx)
        max_idx = tl.where(update_gt | update_eq, local_idx, max_idx)
        max_val = tl.where(update_gt, local_max, max_val)

    # Store results for all BLOCK_X columns (masked)
    tl.store(out_ptr + batch_idx * N + col_offs, max_idx, mask=col_mask)

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