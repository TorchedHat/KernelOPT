import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 512, 'BLOCK_ROWS': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 4096, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 256, 'BLOCK_ROWS': 1}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_N': 8192, 'BLOCK_ROWS': 1}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 1}, num_warps=8, num_stages=5),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=5),
        triton.Config({'BLOCK_N': 4096, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_N': 1024, 'BLOCK_ROWS': 1}, num_warps=16, num_stages=6),
        triton.Config({'BLOCK_N': 512, 'BLOCK_ROWS': 1}, num_warps=8, num_stages=6),
        triton.Config({'BLOCK_N': 2048, 'BLOCK_ROWS': 1}, num_warps=8, num_stages=4),
    ],
    key=['N'],
)
@triton.jit
def l2_norm_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    N_padded,
    BLOCK_N: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return
    
    row_start = row_idx * N
    
    # Pass 1: compute sum of squares
    acc = tl.zeros((), dtype=tl.float32)
    
    for col_off in tl.range(0, N_padded, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    
    # Compute inverse L2 norm (rsqrt is single MUFU.RSQ instruction, avoids FDIV)
    inv_norm = tl.where(acc > 0.0, tl.math.rsqrt(acc), 0.0)
    
    # Pass 2: multiply by inv_norm and store (FMUL instead of FDIV)
    for col_off in tl.range(0, N_padded, BLOCK_N):
        cols = col_off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0).to(tl.float32)
        out = x * inv_norm
        tl.store(out_ptr + row_start + cols, out.to(tl.float32), mask=mask, eviction_policy='evict_first')

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