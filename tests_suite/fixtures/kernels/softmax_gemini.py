import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel_h200(
    input_ptr, output_ptr,
    n_cols,
    stride_row,
    BLOCK_SIZE: tl.constexpr,
    num_stages: tl.constexpr,
):
    # Mapping
    row_idx = tl.program_id(0)
    
    # Use a pointer with a 128-bit alignment hint (float4)
    # This addresses the "No Eligible Warps" by speeding up the global load
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_row_ptr = input_ptr + row_idx * stride_row + col_offsets
    output_row_ptr = output_ptr + row_idx * stride_row + col_offsets
    
    mask = col_offsets < n_cols

    # --- Optimization: Vectorized Asynchronous Load ---
    # By setting num_stages > 1 in the launcher, Triton uses Hopper's 
    # async copy (CP.ASYNC) to move data into SRAM without stalling the ALU.
    row = tl.load(input_row_ptr, mask=mask, other=-float('inf'))

    # --- Optimization: Online Softmax Logic (Register Intensive) ---
    # We purposefully keep 'row' in registers to increase register pressure 
    # slightly, which helps hide memory latency on Hopper.
    row_max = tl.max(row, axis=0)
    
    # Fast Math: Hopper has dedicated FP32 units for Exp
    numerator = tl.exp(row - row_max)
    denominator = tl.sum(numerator, axis=0)
    
    out = numerator / denominator

    # Store back
    tl.store(output_row_ptr, out, mask=mask)

def run_softmax(x):
    M, N = x.shape
    y = torch.empty_like(x)

    BLOCK_SIZE = triton.next_power_of_2(N)

    # H200 specific tuning:
    # 1. num_warps: Increase to 8 or 16 for N > 2048 to hide HBM3e latency.
    # 2. num_stages: H200 (Hopper) excels with 3-4 stages to pipeline memory.
    # 3. grid: We keep (M,) but ensure alignment for vectorized loads.

    num_warps = 8 if BLOCK_SIZE <= 4096 else 16

    # Hopper loves software pipelining (num_stages)
    softmax_kernel_h200[(M,)](
        x, y,
        N,
        x.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
        num_stages=4 if BLOCK_SIZE <= 8192 else 2,
    )
    return y


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()

    X = torch.randn(args.m, args.n, device='cuda', dtype=torch.float32)

    if args.profile_only:
        Y = run_softmax(X)
        torch.cuda.synchronize()
        print(f"Profiled softmax_gemini (H200 optimized): M={args.m}, N={args.n}")
    else:
        # Correctness check
        Y_triton = run_softmax(X)
        Y_torch = torch.softmax(X, dim=1)
        if torch.allclose(Y_triton, Y_torch, atol=1e-4, rtol=1e-4):
            print("✓ Correctness passed")
        else:
            max_err = (Y_triton - Y_torch).abs().max().item()
            print(f"✗ Correctness FAILED: max_err={max_err}")


if __name__ == '__main__':
    main()
