import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 1, 'BLOCK_X': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 1, 'BLOCK_X': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 1, 'BLOCK_X': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R': 1, 'BLOCK_X': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_R': 1, 'BLOCK_X': 64}, num_warps=4, num_stages=2),
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
    # 2D grid: pid_x covers column tiles (BLOCK_X columns per CTA), pid_y covers batch indices
    pid_x = tl.program_id(0)  # column tile
    pid_y = tl.program_id(1)  # batch index

    col_base = pid_x * BLOCK_X
    col_offsets = col_base + tl.arange(0, BLOCK_X)  # contiguous columns
    col_mask = col_offsets < N

    batch_base = x_ptr + pid_y * R * N

    # Initialize max value and argmax index
    max_val = tl.full([BLOCK_X], float('-inf'), dtype=tl.float32)
    max_idx = tl.full([BLOCK_X], 0, dtype=tl.int64)

    # Iterate over reduction dimension — contiguous loads per row (coalesced!)
    for r in tl.range(0, R):
        ptrs = batch_base + r * N + col_offsets  # CONTIGUOUS — coalesced!
        vals = tl.load(ptrs, mask=col_mask, other=float('-inf')).to(tl.float32)

        is_new_max = vals > max_val
        is_tie = vals == max_val
        max_val = tl.where(is_new_max, vals, max_val)
        r_vec = tl.full([BLOCK_X], r, dtype=tl.int64)
        max_idx = tl.where(is_new_max, r_vec, tl.where(is_tie, tl.minimum(max_idx, r_vec), max_idx))

    tl.store(out_ptr + pid_y * N + col_offsets, max_idx, mask=col_mask)

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