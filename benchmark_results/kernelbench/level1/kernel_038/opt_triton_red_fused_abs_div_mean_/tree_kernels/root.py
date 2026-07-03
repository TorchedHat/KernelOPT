import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_N': 1024}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_N': 2048}, num_warps=32, num_stages=2),
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
        col_idx = col_off + tl.arange(0, BLOCK_N)
        mask = col_idx < N
        x = tl.load(x_ptr + row_start + col_idx, mask=mask, other=0.0).to(tl.float32)
        # abs
        x_abs = tl.where(x >= 0.0, x, -x)
        acc += tl.sum(x_abs, axis=0)
    
    # Compute mean
    mean_val = acc / N
    
    # Second pass: divide each element by mean
    for col_off in tl.range(0, N, BLOCK_N):
        col_idx = col_off + tl.arange(0, BLOCK_N)
        mask = col_idx < N
        x = tl.load(x_ptr + row_start + col_idx, mask=mask, other=0.0).to(tl.float32)
        out = x / mean_val
        tl.store(out_ptr + row_start + col_idx, out.to(tl.float32), mask=mask)


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