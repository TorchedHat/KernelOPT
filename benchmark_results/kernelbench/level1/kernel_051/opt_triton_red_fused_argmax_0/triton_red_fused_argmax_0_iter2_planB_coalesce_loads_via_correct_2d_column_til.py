import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 128,  'BLOCK_X': 16}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_R': 128,  'BLOCK_X': 32}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_R': 256,  'BLOCK_X': 16}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_R': 256,  'BLOCK_X': 32}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_R': 512,  'BLOCK_X': 16}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_R': 512,  'BLOCK_X': 32}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_R': 1024, 'BLOCK_X': 16}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_R': 1024, 'BLOCK_X': 32}, num_warps=16, num_stages=2),
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
    pid           = tl.program_id(0)
    num_col_tiles = tl.cdiv(N, BLOCK_X)   # must match grid formula
    batch_idx     = pid // num_col_tiles
    col_tile      = pid % num_col_tiles
    col_base      = col_tile * BLOCK_X
    col_offsets   = col_base + tl.arange(0, BLOCK_X)  # [BLOCK_X]
    col_mask      = col_offsets < N                    # tail guard for N=4095

    max_val = tl.full([BLOCK_X], float('-inf'), dtype=tl.float32)  # [BLOCK_X]
    max_idx = tl.zeros([BLOCK_X], dtype=tl.int64)                  # [BLOCK_X]

    # Use int64 arithmetic to avoid overflow: batch_idx * R * N can exceed 2^31
    batch_base = x_ptr + batch_idx.to(tl.int64) * R.to(tl.int64) * N.to(tl.int64)

    for r_start in tl.range(0, R, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)  # [BLOCK_R]
        r_mask    = r_offsets < R                      # [BLOCK_R]
        # 2D pointer: [BLOCK_R, BLOCK_X]; innermost dim is col (stride-1) — COALESCED
        ptrs = batch_base + r_offsets[:, None].to(tl.int64) * N.to(tl.int64) + col_offsets[None, :].to(tl.int64)
        mask = r_mask[:, None] & col_mask[None, :]
        vals = tl.load(ptrs, mask=mask, other=float('-inf'))  # [BLOCK_R, BLOCK_X]
        vals = vals.to(tl.float32)

        local_max = tl.max(vals, axis=0)   # [BLOCK_X] — max over rows
        is_max    = (vals == local_max[None, :]) & mask  # [BLOCK_R, BLOCK_X]
        r_idx_2d  = tl.where(is_max, r_offsets[:, None].to(tl.int64),
                              tl.full([BLOCK_R, BLOCK_X], R, dtype=tl.int64))
        local_idx = tl.min(r_idx_2d, axis=0)  # [BLOCK_X] — first occurrence

        improved  = local_max > max_val
        tied      = (local_max == max_val) & ~improved
        max_val   = tl.where(improved, local_max, max_val)
        max_idx   = tl.where(improved, local_idx,
                    tl.where(tied, tl.minimum(max_idx, local_idx), max_idx))

    out_ptrs = out_ptr + batch_idx.to(tl.int64) * N.to(tl.int64) + col_offsets.to(tl.int64)
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