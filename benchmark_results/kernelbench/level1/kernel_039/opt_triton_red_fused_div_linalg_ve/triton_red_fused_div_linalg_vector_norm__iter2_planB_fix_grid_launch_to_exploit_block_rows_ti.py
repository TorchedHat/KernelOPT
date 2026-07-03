import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 1}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_N': 4096, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_N': 8192, 'BLOCK_ROWS': 1}, num_warps=32, num_stages=3),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 2}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 2}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_N': 4096, 'BLOCK_ROWS': 2}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 4}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 4}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 8}, num_warps=8,  num_stages=5),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 8}, num_warps=16, num_stages=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    row_base = tl.program_id(0) * BLOCK_ROWS
    for r in tl.static_range(BLOCK_ROWS):  # compile-time unroll
        row_idx = row_base + r
        if row_idx < M:
            row_start = row_idx * N

            # Pass 1: compute sum of squares
            acc = tl.zeros((), dtype=tl.float32)
            for col_off in tl.range(0, N, BLOCK_N):
                cols = col_off + tl.arange(0, BLOCK_N)
                mask = cols < N
                x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
                acc += tl.sum(x * x, axis=0)

            # Compute L2 norm
            norm = tl.sqrt(acc)

            # Pass 2: divide by norm and store
            for col_off in tl.range(0, N, BLOCK_N):
                cols = col_off + tl.arange(0, BLOCK_N)
                mask = cols < N
                x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
                tl.store(out_ptr + row_start + cols, (x / norm).to(tl.float32), mask=mask)

def kernel_function(x):
    M, N = x.shape
    out = torch.empty_like(x)
    
    grid = (M,)
    
    l2_norm_kernel[grid](
        x,
        out,
        M,
        N,
    )
    
    return out


def get_inputs():
    x = torch.rand(32768, 65535, device='cuda', dtype=torch.float32)
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
        out_ref = x / torch.norm(x, p=2, dim=1, keepdim=True)
        torch.cuda.synchronize()
        max_diff = (out_triton - out_ref).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")