import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_N': 4096}, num_warps=32, num_stages=3),
        triton.Config({'BLOCK_N': 8192}, num_warps=32, num_stages=3),
        triton.Config({'BLOCK_N': 8192}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_N': 16384}, num_warps=32, num_stages=3),
        triton.Config({'BLOCK_N': 16384}, num_warps=32, num_stages=4),
    ],
    key=['N_PER_ROW'],
)
@triton.jit
def partial_sum_sq_kernel(
    x_ptr,
    partial_ptr,
    N_PER_ROW: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_start = row_id * N_PER_ROW
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, N_PER_ROW, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        cols = tl.max_contiguous(tl.multiple_of(cols, BLOCK_N), BLOCK_N)
        mask = cols < N_PER_ROW
        x = tl.load(x_ptr + row_start + cols, mask=mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    tl.store(partial_ptr + row_id, acc)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 4096}, num_warps=16, num_stages=4),
    ],
    key=['N_ROWS'],
)
@triton.jit

def final_reduce_kernel(
    partial_ptr,
    norm_sq_ptr,
    N_ROWS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, N_ROWS, BLOCK_R):
        cols = off + tl.arange(0, BLOCK_R)
        mask = cols < N_ROWS
        v = tl.load(partial_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(v, axis=0)
    norm = tl.sqrt(acc)
    tl.store(norm_sq_ptr, norm)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def normalize_kernel(
    x_ptr,
    norm_ptr,
    out_ptr,
    N_TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N_TOTAL
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    norm = tl.load(norm_ptr).to(tl.float32)
    out = x / norm
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def kernel_function(x):
    # x: [112, 64, 512, 512] float32
    # Frobenius norm = sqrt(sum(x^2)), then divide x by norm
    
    total_elements = x.numel()  # 112 * 64 * 512 * 512 = 1,879,048,192
    
    # We'll split the reduction into rows of 512*512 = 262144 elements
    # Number of rows = 112 * 64 = 7168
    N_PER_ROW = 512 * 512  # 262144
    N_ROWS = 112 * 64       # 7168
    
    x_flat = x.view(-1)
    
    # Step 1: partial sum of squares per row
    partial = torch.empty(N_ROWS, dtype=torch.float32, device=x.device)
    
    grid1 = (N_ROWS,)
    partial_sum_sq_kernel[grid1](
        x_flat,
        partial,
        N_PER_ROW=N_PER_ROW,
    )
    
    # Step 2: final reduce to get norm
    norm_buf = torch.empty(1, dtype=torch.float32, device=x.device)
    
    grid2 = (1,)
    final_reduce_kernel[grid2](
        partial,
        norm_buf,
        N_ROWS=N_ROWS,
    )
    
    # Step 3: normalize
    out = torch.empty_like(x)
    out_flat = out.view(-1)
    
    grid3 = lambda meta: (triton.cdiv(total_elements, meta['BLOCK_SIZE']),)
    normalize_kernel[grid3](
        x_flat,
        norm_buf,
        out_flat,
        N_TOTAL=total_elements,
    )
    
    return out


def get_inputs():
    x = torch.rand(112, 64, 512, 512, device='cuda', dtype=torch.float32)
    return [x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        # Verify correctness
        x = inputs[0]
        norm_ref = torch.norm(x, p='fro')
        out_ref = x / norm_ref
        max_diff = (output - out_ref).abs().max().item()
        print(f"Max difference from reference: {max_diff:.6e}")
        assert max_diff < 1e-3, f"Correctness check failed: max_diff={max_diff}"
        print("Kernel executed successfully.")