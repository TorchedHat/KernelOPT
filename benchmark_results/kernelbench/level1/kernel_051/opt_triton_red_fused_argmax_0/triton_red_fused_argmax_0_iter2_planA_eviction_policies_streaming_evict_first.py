import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 128,  'BLOCK_X': 1}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_R': 256,  'BLOCK_X': 1}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_R': 512,  'BLOCK_X': 1}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_R': 1024, 'BLOCK_X': 1}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_R': 2048, 'BLOCK_X': 1}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 4096, 'BLOCK_X': 1}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 4096, 'BLOCK_X': 1}, num_warps=32, num_stages=1),
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
    # Each program handles one (batch, col) pair
    pid = tl.program_id(0)
    # pid indexes over B * N output elements
    batch_idx = pid // N
    col_idx = pid % N

    # Base pointer for this (batch, col) slice
    # x[batch_idx, :, col_idx] has stride R=4096 in dim1, stride 1 in dim2
    # x_ptr + batch_idx * R * N + col_idx
    base = x_ptr + batch_idx * R * N + col_idx

    # Initialize max value and argmax index
    max_val = tl.full((), float('-inf'), dtype=tl.float32)
    max_idx = tl.full((), 0, dtype=tl.int64)

    # Iterate over reduction dimension in blocks
    for r_start in tl.range(0, R, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)
        # Vectorization hints: r_offsets is contiguous and BLOCK_R-aligned
        r_offsets = tl.max_contiguous(tl.multiple_of(r_offsets, BLOCK_R), BLOCK_R)
        mask = r_offsets < R
        # Load x[batch_idx, r_offsets, col_idx] — streaming data, used once
        ptrs = base + r_offsets * N
        vals = tl.load(ptrs, mask=mask, other=float('-inf'), eviction_policy='evict_first')
        vals = vals.to(tl.float32)

        # Find local max and its index
        local_max = tl.max(vals, axis=0)
        # Find the index of the local max
        # Use argmax: find first occurrence of max
        is_max = (vals == local_max) & mask
        # Get the index: use a trick with where
        indices = tl.where(is_max, r_offsets.to(tl.int64), tl.full([BLOCK_R], R, dtype=tl.int64))
        local_idx = tl.min(indices, axis=0)

        # Update global max
        if local_max > max_val:
            max_val = local_max
            max_idx = local_idx
        elif local_max == max_val:
            max_idx = tl.minimum(max_idx, local_idx)

    tl.store(out_ptr + batch_idx * N + col_idx, max_idx, eviction_policy='evict_first')

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