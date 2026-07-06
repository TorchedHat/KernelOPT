import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 512},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_N': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_N': 256},  num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_N': 512},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_N': 1024}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_N': 2048}, num_warps=32, num_stages=3),
        triton.Config({'BLOCK_N': 512},  num_warps=4,  num_stages=5),
        triton.Config({'BLOCK_N': 1024}, num_warps=8,  num_stages=5),
        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=5),
        triton.Config({'BLOCK_N': 512},  num_warps=4,  num_stages=6),
        triton.Config({'BLOCK_N': 1024}, num_warps=8,  num_stages=6),
    ],
    key=['N'],
)
@triton.jit
def l1_norm_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    # Compute the sum of absolute values for this row
    row_start = row_idx * N
    
    # Accumulate sum of abs values in float32
    acc = tl.zeros((), dtype=tl.float32)
    
    for col_off in tl.range(0, N, BLOCK_N):
        col_off_hint = tl.multiple_of(col_off, BLOCK_N)  # col_off is always a multiple of BLOCK_N by construction
        col_idx = col_off_hint + tl.arange(0, BLOCK_N)
        mask = col_idx < N
        x = tl.load(x_ptr + row_start + col_idx, mask=mask, other=0.0).to(tl.float32)
        # abs
        x_abs = tl.abs(x)
        acc += tl.sum(x_abs, axis=0)
    
    # Compute mean
    mean_val = acc / N
    
    # Precompute reciprocal once outside the loop to enable FMA emission
    inv_mean = 1.0 / mean_val
    
    # Second pass: multiply each element by inv_mean (enables FFMA instead of non-fused div)
    for col_off in tl.range(0, N, BLOCK_N):
        col_off_hint = tl.multiple_of(col_off, BLOCK_N)  # col_off is always a multiple of BLOCK_N by construction
        col_idx = col_off_hint + tl.arange(0, BLOCK_N)
        mask = col_idx < N
        x = tl.load(x_ptr + row_start + col_idx, mask=mask, other=0.0).to(tl.float32)
        out = x * inv_mean  # multiply instead of divide -- compiler emits FFMA
        tl.store(out_ptr + row_start + col_idx, out, mask=mask)

def kernel_function(x):
    M, N = x.shape
    out = torch.empty_like(x)
    
    # Launch one program per row
    grid = (M,)
    
    l1_norm_kernel[grid](
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
    
    if not args.profile_only:
        # Verify correctness
        x = inputs[0]
        
        # Reference implementation
        ref_out = x / torch.mean(torch.abs(x), dim=1, keepdim=True)
        
        # Triton implementation
        triton_out = kernel_function(x)
        
        torch.cuda.synchronize()
        
        max_diff = (ref_out - triton_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match!")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")