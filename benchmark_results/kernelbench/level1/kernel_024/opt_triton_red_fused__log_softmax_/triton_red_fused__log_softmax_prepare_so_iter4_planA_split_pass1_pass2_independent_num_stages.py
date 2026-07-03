import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 16384, 'PASS1_STAGES': 2, 'PASS2_STAGES': 4}, num_warps=32),
        triton.Config({'BLOCK_N': 16384, 'PASS1_STAGES': 2, 'PASS2_STAGES': 5}, num_warps=32),
        triton.Config({'BLOCK_N': 16384, 'PASS1_STAGES': 2, 'PASS2_STAGES': 3}, num_warps=32),
        triton.Config({'BLOCK_N': 16384, 'PASS1_STAGES': 3, 'PASS2_STAGES': 5}, num_warps=32),
        triton.Config({'BLOCK_N': 16384, 'PASS1_STAGES': 3, 'PASS2_STAGES': 4}, num_warps=32),
        triton.Config({'BLOCK_N': 16384, 'PASS1_STAGES': 4, 'PASS2_STAGES': 4}, num_warps=32),
        triton.Config({'BLOCK_N': 8192,  'PASS1_STAGES': 2, 'PASS2_STAGES': 5}, num_warps=32),
        triton.Config({'BLOCK_N': 8192,  'PASS1_STAGES': 2, 'PASS2_STAGES': 4}, num_warps=32),
        triton.Config({'BLOCK_N': 8192,  'PASS1_STAGES': 3, 'PASS2_STAGES': 5}, num_warps=32),
        triton.Config({'BLOCK_N': 8192,  'PASS1_STAGES': 3, 'PASS2_STAGES': 4}, num_warps=16),
        triton.Config({'BLOCK_N': 8192,  'PASS1_STAGES': 2, 'PASS2_STAGES': 3}, num_warps=32),
        triton.Config({'BLOCK_N': 4096,  'PASS1_STAGES': 2, 'PASS2_STAGES': 5}, num_warps=16),
        triton.Config({'BLOCK_N': 4096,  'PASS1_STAGES': 2, 'PASS2_STAGES': 4}, num_warps=16),
        triton.Config({'BLOCK_N': 4096,  'PASS1_STAGES': 3, 'PASS2_STAGES': 5}, num_warps=16),
        triton.Config({'BLOCK_N': 4096,  'PASS1_STAGES': 2, 'PASS2_STAGES': 3}, num_warps=32),
        triton.Config({'BLOCK_N': 2048,  'PASS1_STAGES': 2, 'PASS2_STAGES': 5}, num_warps=16),
        triton.Config({'BLOCK_N': 2048,  'PASS1_STAGES': 2, 'PASS2_STAGES': 4}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def log_softmax_kernel(
    input_ptr,
    output_ptr,
    M,
    N,
    BLOCK_N: tl.constexpr,
    PASS1_STAGES: tl.constexpr,
    PASS2_STAGES: tl.constexpr,
):
    # Each program handles one row
    row_idx = tl.program_id(0)
    
    if row_idx >= M:
        return
    
    row_start = input_ptr + row_idx * N
    
    # Pass 1: find max (for numerical stability) and compute sum of exp
    # Use online softmax algorithm
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((), dtype=tl.float32)
    
    for col_off in tl.range(0, N, BLOCK_N, num_stages=PASS1_STAGES):
        cols = col_off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_start + cols, mask=mask, other=float('-inf')).to(tl.float32)
        
        # Online softmax update
        block_max = tl.max(x, axis=0)
        new_max = tl.maximum(row_max, block_max)
        
        # Update sum: scale old sum by exp(old_max - new_max), add new terms
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(x - new_max), axis=0)
        row_max = new_max
    
    log_sum = tl.log(row_sum)
    # Hoist loop-invariant scalar: offset = -(row_max + log_sum)
    # Equivalent to subtracting row_max and log_sum separately, but enables FMA
    offset = -(row_max + log_sum)
    
    # Pass 2: compute log_softmax = x - max - log(sum(exp(x - max)))
    out_start = output_ptr + row_idx * N
    
    # Use independent PASS2_STAGES to allow deeper pipelining for pure streaming pass
    for col_off in tl.range(0, N, BLOCK_N, num_stages=PASS2_STAGES):
        cols = col_off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_start + cols, mask=mask, other=0.0).to(tl.float32)
        
        # Single addition with hoisted offset (FMA-friendly)
        out = x + offset
        
        tl.store(out_start + cols, out.to(tl.float32), mask=mask)

def kernel_function(*inputs):
    x, = inputs
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Input must be float32"
    
    M, N = x.shape
    output = torch.empty_like(x)
    
    # Launch one program per row
    grid = (M,)
    
    log_softmax_kernel[grid](
        x,
        output,
        M,
        N,
    )
    
    return output


def get_inputs():
    x = torch.rand(4096, 393216, device='cuda', dtype=torch.float32)
    return [x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    
    # Verify correctness on a small example first
    if not args.profile_only:
        x_small = torch.rand(4, 1024, device='cuda', dtype=torch.float32)
        ref = torch.log_softmax(x_small, dim=1)
        out = kernel_function(x_small)
        max_diff = (ref - out).abs().max().item()
        print(f"Max difference (small test): {max_diff:.6f}")
        assert max_diff < 1e-3, f"Correctness check failed: max_diff={max_diff}"
        print("Correctness check passed!")
    
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")
        print("Kernel executed successfully.")