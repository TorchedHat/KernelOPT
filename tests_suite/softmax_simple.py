import argparse
import torch
import triton
import triton.language as tl


# ==========================================
# Triton Kernel
# ==========================================
@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    # One program per row
    pid = tl.program_id(0)

    row_start_x = x_ptr + pid * stride_xm
    row_start_y = y_ptr + pid * stride_ym

    offs = tl.arange(0, BLOCK_SIZE)

    # ----------------------------
    # Load row
    # ----------------------------
    x = tl.load(
        row_start_x + offs * stride_xn,
        mask=offs < N,
        other=-float('inf')
    )

    # ----------------------------
    # Row max (numerical stability)
    # ----------------------------
    row_max = tl.max(x, axis=0)
    x = x - row_max

    # ----------------------------
    # exp + sum
    # ----------------------------
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)

    # ----------------------------
    # Normalize
    # ----------------------------
    softmax = exp_x / denom

    # ----------------------------
    # Store
    # ----------------------------
    tl.store(
        row_start_y + offs * stride_yn,
        softmax,
        mask=offs < N
    )


def run_triton_softmax(X: torch.Tensor) -> torch.Tensor:
    M, N = X.shape
    Y = torch.empty_like(X)

    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_SIZE <= 256 else 8

    softmax_kernel[(M,)](
        X, Y,
        M, N,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y


# ==========================================
# Helpers
# ==========================================
def check_correctness(m, n):
    X = torch.randn(m, n, device='cuda', dtype=torch.float32)

    ref = torch.softmax(X, dim=1)
    out = run_triton_softmax(X)

    if torch.allclose(ref, out, atol=1e-4, rtol=1e-4):
        print("✓ Correctness check passed")
    else:
        max_err = (ref - out).abs().max().item()
        print(f"✗ Correctness FAILED — max error: {max_err:.6f}")


def benchmark(m, n):
    X = torch.randn(m, n, device='cuda', dtype=torch.float32)

    # Warmup
    for _ in range(5):
        run_triton_softmax(X)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end   = torch.cuda.Event(enable_timing=True)
    REPS  = 50

    # Triton
    start.record()
    for _ in range(REPS):
        run_triton_softmax(X)
    end.record()
    torch.cuda.synchronize()
    triton_ms = start.elapsed_time(end) / REPS

    # PyTorch
    start.record()
    for _ in range(REPS):
        torch.softmax(X, dim=1)
    end.record()
    torch.cuda.synchronize()
    torch_ms = start.elapsed_time(end) / REPS

    print(f"\n{'':=<50}")
    print(f"  Triton  : {triton_ms:.3f} ms")
    print(f"  PyTorch : {torch_ms:.3f} ms")
    print(f"  Speedup : {torch_ms / triton_ms:.2f}x")
    print(f"{'':=<50}")


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Triton fused softmax kernel (2D)")
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--profile-only', action='store_true',
                        help='Run once for profiling (e.g., NCU)')
    args = parser.parse_args()

    print(f"Shape: M={args.m}, N={args.n}")

    if args.profile_only:
        X = torch.randn(args.m, args.n, device='cuda', dtype=torch.float32)
        run_triton_softmax(X)
        torch.cuda.synchronize()
        print("Profile-only run complete.")
    else:
        check_correctness(args.m, args.n)
        benchmark(args.m, args.n)


if __name__ == '__main__':
    main()