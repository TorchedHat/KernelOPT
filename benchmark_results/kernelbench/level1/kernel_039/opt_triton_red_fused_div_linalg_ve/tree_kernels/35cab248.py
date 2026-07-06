import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 1},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 2},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 4},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 2},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 4},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 512,  'BLOCK_ROWS': 4},  num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_N': 512,  'BLOCK_ROWS': 8},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 256,  'BLOCK_ROWS': 8},  num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_N': 256,  'BLOCK_ROWS': 16}, num_warps=8,  num_stages=4),
    ],
    key=['N'],
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
    # Compute row indices for this CTA
    rows = tl.program_id(0) * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)  # [BLOCK_ROWS]
    row_mask = rows < M  # [BLOCK_ROWS]

    # Pass 1: compute sum of squares for each row using 2D FMA accumulator
    acc2d = tl.zeros([BLOCK_ROWS, BLOCK_N], dtype=tl.float32)  # [BLOCK_ROWS, BLOCK_N]
    for col_off in tl.range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        ptrs = x_ptr + rows[:, None] * N + cols[None, :]  # [BLOCK_ROWS, BLOCK_N]
        mask2d = row_mask[:, None] & (cols[None, :] < N)  # [BLOCK_ROWS, BLOCK_N]
        x = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
        acc2d = tl.math.fma(x, x, acc2d)  # single FFMA per element

    acc = tl.sum(acc2d, axis=1)  # one reduction after loop -> [BLOCK_ROWS]
    norm = tl.sqrt(acc)  # [BLOCK_ROWS]

    # Pass 2: normalize and store
    for col_off in tl.range(0, N, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)  # [BLOCK_N]
        ptrs = x_ptr + rows[:, None] * N + cols[None, :]  # [BLOCK_ROWS, BLOCK_N]
        out_ptrs = out_ptr + rows[:, None] * N + cols[None, :]  # [BLOCK_ROWS, BLOCK_N]
        mask2d = row_mask[:, None] & (cols[None, :] < N)
        x = tl.load(ptrs, mask=mask2d, other=0.0).to(tl.float32)
        out = x / norm[:, None]  # broadcast [BLOCK_ROWS,1] over [BLOCK_ROWS, BLOCK_N]
        tl.store(out_ptrs, out.to(tl.float32), mask=mask2d)

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