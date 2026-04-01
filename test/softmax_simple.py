import argparse
import torch
import triton
import triton.language as tl


@triton.jit
def softmax_kernel(
    x_ptr, y_ptr,
    M, N,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    row_start_x = x_ptr + pid * stride_xm
    row_start_y = y_ptr + pid * stride_ym
    offs = tl.arange(0, BLOCK_SIZE)

    # Load row
    x = tl.load(row_start_x + offs * stride_xn, mask=offs < N, other=-float('inf'))

    # Softmax: max, exp, sum, normalize
    row_max = tl.max(x, axis=0)
    x = x - row_max
    exp_x = tl.exp(x)
    denom = tl.sum(exp_x, axis=0)
    softmax = exp_x / denom

    # Store
    tl.store(row_start_y + offs * stride_yn, softmax, mask=offs < N)


def run_triton_softmax(X: torch.Tensor) -> torch.Tensor:
    M, N = X.shape
    Y = torch.empty_like(X)
    BLOCK_SIZE = triton.next_power_of_2(N)
    num_warps = 4 if BLOCK_SIZE <= 256 else 8

    softmax_kernel[(M,)](
        X, Y, M, N,
        X.stride(0), X.stride(1),
        Y.stride(0), Y.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return Y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m', type=int, default=4096)
    parser.add_argument('--n', type=int, default=1024)
    parser.add_argument('--profile-only', action='store_true')
    args = parser.parse_args()

    X = torch.randn(args.m, args.n, device='cuda', dtype=torch.float32)

    if args.profile_only:
        Y = run_triton_softmax(X)
        torch.cuda.synchronize()
        print(f"Profiled softmax: M={args.m}, N={args.n}")
    else:
        # Correctness check
        Y_triton = run_triton_softmax(X)
        Y_torch = torch.softmax(X, dim=1)
        if torch.allclose(Y_triton, Y_torch, atol=1e-4, rtol=1e-4):
            print("✓ Correctness passed")
        else:
            max_err = (Y_triton - Y_torch).abs().max().item()
            print(f"✗ Correctness FAILED: max_err={max_err}")


if __name__ == '__main__':
    main()
