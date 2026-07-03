import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 32,  'BLOCK_X': 32},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_R': 64,  'BLOCK_X': 32},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_R': 128, 'BLOCK_X': 32},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_R': 32,  'BLOCK_X': 64},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_R': 64,  'BLOCK_X': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_R': 128, 'BLOCK_X': 64},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_R': 32,  'BLOCK_X': 128}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_R': 64,  'BLOCK_X': 128}, num_warps=16, num_stages=2),
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
    pid = tl.program_id(0)
    # BLOCK_X columns per CTA -> num_col_tiles = cdiv(N, BLOCK_X)
    num_col_tiles = tl.cdiv(N, BLOCK_X)
    batch_idx = pid // num_col_tiles
    col_tile = pid % num_col_tiles
    col_base = col_tile * BLOCK_X

    # Column offsets for this tile (contiguous in memory = coalesced)
    c_offs = col_base + tl.arange(0, BLOCK_X)  # [BLOCK_X]
    c_mask = c_offs < N  # guard for N=4095 (not power of 2)

    # Initialize per-column accumulators
    max_val = tl.full([BLOCK_X], float('-inf'), dtype=tl.float32)
    max_idx = tl.zeros([BLOCK_X], dtype=tl.int64)

    # Base pointer for this batch
    batch_base = x_ptr + batch_idx * R * N

    # Iterate over reduction dimension in BLOCK_R chunks
    for r_start in tl.range(0, R, BLOCK_R):
        r_offs = r_start + tl.arange(0, BLOCK_R)  # [BLOCK_R]
        r_mask = r_offs < R  # guard for boundary

        # 2D mask: [BLOCK_R, BLOCK_X]
        mask2d = r_mask[:, None] & c_mask[None, :]

        # Coalesced 2D load: innermost dim is col (stride-1)
        # ptrs[i, j] = batch_base + r_offs[i] * N + c_offs[j]
        ptrs = batch_base + r_offs[:, None] * N + c_offs[None, :]
        tile = tl.load(ptrs, mask=mask2d, other=float('-inf'))
        tile = tile.to(tl.float32)  # [BLOCK_R, BLOCK_X]

        # Find local max per column
        local_max = tl.max(tile, axis=0)  # [BLOCK_X]

        # Find smallest row index achieving local_max per column
        is_max = (tile == local_max[None, :]) & r_mask[:, None]
        r_offs_2d = tl.broadcast_to(r_offs[:, None], [BLOCK_R, BLOCK_X])
        sentinel = tl.full([BLOCK_R, BLOCK_X], R, dtype=tl.int64)
        row_indices = tl.where(is_max, r_offs_2d.to(tl.int64), sentinel)
        local_idx = tl.min(row_indices, axis=0)  # [BLOCK_X]

        # Merge with running max:
        # (1) new max > old max -> replace both
        # (2) new max == old max -> keep min index
        # (3) new max < old max -> keep old
        update = local_max > max_val
        tie = (local_max == max_val) & (local_idx < max_idx)
        max_idx = tl.where(update, local_idx, tl.where(tie, local_idx, max_idx))
        max_val = tl.where(update, local_max, max_val)

    # Store all BLOCK_X results
    out_ptrs = out_ptr + batch_idx * N + c_offs
    tl.store(out_ptrs, max_idx, mask=c_mask)

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