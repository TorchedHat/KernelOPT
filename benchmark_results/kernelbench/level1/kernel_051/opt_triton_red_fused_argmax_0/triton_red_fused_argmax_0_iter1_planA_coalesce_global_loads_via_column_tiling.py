import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 64,  'BLOCK_X': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 128, 'BLOCK_X': 32},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 256, 'BLOCK_X': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 64,  'BLOCK_X': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 128, 'BLOCK_X': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 32,  'BLOCK_X': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 64,  'BLOCK_X': 128}, num_warps=8, num_stages=2),
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
    pid_batch = tl.program_id(0)
    pid_col   = tl.program_id(1)

    col_base    = pid_col * BLOCK_X
    col_offsets = col_base + tl.arange(0, BLOCK_X)   # [BLOCK_X]
    col_mask    = col_offsets < N

    base = x_ptr + pid_batch * R * N

    # Initialize max value and argmax index — shape [BLOCK_X]
    max_val = tl.full([BLOCK_X], float('-inf'), dtype=tl.float32)
    max_idx = tl.full([BLOCK_X], 0, dtype=tl.int64)

    for r_start in tl.range(0, R, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)   # [BLOCK_R]
        r_mask = r_offsets < R

        # 2D tile [BLOCK_R, BLOCK_X] — coalesced along BLOCK_X dimension
        ptrs   = base + r_offsets[:, None] * N + col_offsets[None, :]
        mask2d = r_mask[:, None] & col_mask[None, :]
        vals   = tl.load(ptrs, mask=mask2d, other=float('-inf'),
                         eviction_policy='evict_first')
        vals   = vals.to(tl.float32)

        local_max = tl.max(vals, axis=0)               # [BLOCK_X]
        is_max    = (vals == local_max[None, :]) & mask2d
        indices   = tl.where(is_max,
                             tl.broadcast_to(r_offsets[:, None].to(tl.int64), [BLOCK_R, BLOCK_X]),
                             tl.full([BLOCK_R, BLOCK_X], R, dtype=tl.int64))
        local_idx = tl.min(indices, axis=0)            # [BLOCK_X]

        update  = local_max > max_val
        max_idx = tl.where(update, local_idx,
                           tl.where(local_max == max_val,
                                    tl.minimum(max_idx, local_idx), max_idx))
        max_val = tl.maximum(max_val, local_max)

    out_ptrs = out_ptr + pid_batch * N + col_offsets
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