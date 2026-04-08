"""
Matrix Multiplication - Adapted for kernelopt profiling
Based on official Triton tutorial: 03-matrix-multiplication.py
"""

import argparse
import torch
import triton
import triton.language as tl


@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # Map program ids to the block of C it should compute
    # This is done in a grouped ordering to promote L2 data reuse
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Create pointers for the first blocks of A and B
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Iterate to compute a block of the C matrix
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # Accumulate along the K dimension
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)

    # Write back the block of the output matrix C
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


def run_matmul(a, b):
    """Wrapper function for matrix multiplication."""
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape

    # Allocate output
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # Fixed configuration (suboptimal - room for optimization!)
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32
    GROUP_SIZE_M = 8

    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_warps=4,
        num_stages=2,
    )
    return c


def check_correctness(m, k, n):
    """Verify against PyTorch's matmul."""
    a = torch.randn((m, k), device='cuda', dtype=torch.float16)
    b = torch.randn((k, n), device='cuda', dtype=torch.float16)

    triton_output = run_matmul(a, b)
    torch_output = torch.matmul(a, b)

    if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=1e-2):
        print("✓ Correctness check passed")
    else:
        max_err = (triton_output - torch_output).abs().max().item()
        print(f"✗ Correctness FAILED — max error: {max_err:.6f}")


def benchmark(m, k, n):
    """Benchmark Triton vs PyTorch matmul."""
    a = torch.randn((m, k), device='cuda', dtype=torch.float16)
    b = torch.randn((k, n), device='cuda', dtype=torch.float16)

    # Warmup
    for _ in range(5):
        run_matmul(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    REPS = 50

    # Triton
    start.record()
    for _ in range(REPS):
        run_matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / REPS

    # PyTorch
    start.record()
    for _ in range(REPS):
        torch.matmul(a, b)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / REPS

    # Calculate TFLOPS
    total_flops = 2 * m * n * k
    triton_tflops = total_flops / (triton_ms * 1e-3) / 1e12
    torch_tflops = total_flops / (torch_ms * 1e-3) / 1e12

    print(f"\n{'':=<60}")
    print(f"  Triton  : {triton_ms:.3f} ms ({triton_tflops:.2f} TFLOPS)")
    print(f"  PyTorch : {torch_ms:.3f} ms ({torch_tflops:.2f} TFLOPS)")
    print(f"  Speedup : {torch_ms / triton_ms:.2f}x")
    print(f"{'':=<60}")


def main():
    parser = argparse.ArgumentParser(description="Triton matrix multiplication kernel")
    parser.add_argument('--m', type=int, default=512, help='M dimension')
    parser.add_argument('--k', type=int, default=512, help='K dimension')
    parser.add_argument('--n', type=int, default=512, help='N dimension')
    parser.add_argument('--profile-only', action='store_true',
                        help='Run once for profiling (e.g., NCU)')
    args = parser.parse_args()

    print(f"Shape: M={args.m}, K={args.k}, N={args.n}")

    if args.profile_only:
        a = torch.randn((args.m, args.k), device='cuda', dtype=torch.float16)
        b = torch.randn((args.k, args.n), device='cuda', dtype=torch.float16)
        run_matmul(a, b)
        torch.cuda.synchronize()
        print("Profile-only run complete.")
    else:
        check_correctness(args.m, args.k, args.n)
        benchmark(args.m, args.k, args.n)


if __name__ == '__main__':
    main()
