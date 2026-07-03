import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M_TOTAL', 'N_OUT', 'K_TOTAL'],
)
@triton.jit
def conv1d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, C_IN, L_IN, C_OUT, L_OUT, KERNEL_SIZE,
    stride_xn, stride_xc, stride_xl,
    stride_wco, stride_wci, stride_wk,
    stride_yn, stride_yco, stride_yl,
    STRIDE_CONV, PAD, DILATION,
    M_TOTAL, N_OUT, K_TOTAL,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, l_out)
    batch = offs_m // L_OUT
    l_out = offs_m % L_OUT

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float16)

    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kk)
        in_c = offs_k // KERNEL_SIZE
        kk = offs_k % KERNEL_SIZE

        # Input position
        in_pos = l_out[:, None] * STRIDE_CONV + kk[None, :] * DILATION - PAD

        mask_m = (offs_m[:, None] < M_TOTAL) & (batch[:, None] < N)
        mask_k = offs_k[None, :] < K_TOTAL
        mask_ic = in_c[None, :] < C_IN
        mask_pos = (in_pos >= 0) & (in_pos < L_IN)

        mask_x = mask_m & mask_k & mask_ic & mask_pos

        x_ptrs = X_ptr + batch[:, None] * stride_xn + in_c[None, :] * stride_xc + in_pos * stride_xl
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: w[co, in_c, kk] -> shape (BLOCK_K, BLOCK_N)
        # offs_k indexes (in_c, kk), offs_n indexes co
        mask_w_k = (offs_k[:, None] < K_TOTAL) & (in_c[:, None] < C_IN) & (kk[:, None] < KERNEL_SIZE)
        mask_w_n = offs_n[None, :] < N_OUT
        mask_w = mask_w_k & mask_w_n

        w_ptrs = W_ptr + offs_n[None, :] * stride_wco + in_c[:, None] * stride_wci + kk[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, out_dtype=tl.float16)

    # Store output
    mask_out_m = (offs_m[:, None] < M_TOTAL) & (batch[:, None] < N) & (l_out[:, None] < L_OUT)
    mask_out_n = offs_n[None, :] < N_OUT
    mask_out = mask_out_m & mask_out_n

    y_ptrs = Y_ptr + batch[:, None] * stride_yn + offs_n[None, :] * stride_yco + l_out[:, None] * stride_yl
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)

def kernel_function(weight, x):
    # weight: (C_out, C_in, K), x: (N, C_in, L_in)
    N, C_IN, L_IN = x.shape
    C_OUT, _, KERNEL_SIZE = weight.shape
    STRIDE_CONV = 1
    PAD = 0
    DILATION = 1

    L_OUT = (L_IN + 2 * PAD - DILATION * (KERNEL_SIZE - 1) - 1) // STRIDE_CONV + 1

    y = torch.empty((N, C_OUT, L_OUT), device=x.device, dtype=x.dtype)

    M_TOTAL = N * L_OUT
    N_OUT = C_OUT
    K_TOTAL = C_IN * KERNEL_SIZE

    stride_xn, stride_xc, stride_xl = x.stride()
    stride_wco, stride_wci, stride_wk = weight.stride()
    stride_yn, stride_yco, stride_yl = y.stride()

    def cdiv(a, b): return (a + b - 1) // b

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(N_OUT, meta['BLOCK_N']))

    conv1d_kernel[grid](
        x, weight, y,
        N, C_IN, L_IN, C_OUT, L_OUT, KERNEL_SIZE,
        stride_xn, stride_xc, stride_xl,
        stride_wco, stride_wci, stride_wk,
        stride_yn, stride_yco, stride_yl,
        STRIDE_CONV, PAD, DILATION,
        M_TOTAL, N_OUT, K_TOTAL,
    )

    return y


def get_inputs():
    weight = torch.randn(128, 64, 3, dtype=torch.float32, device='cuda')
    x = torch.randn(32, 64, 131072, dtype=torch.float32, device='cuda')
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, x = inputs

    # Verify correctness
    if not args.profile_only:
        import torch.nn.functional as F
        ref = F.conv1d(x, weight, bias=None, stride=1, padding=0, dilation=1, groups=1)
        out = kernel_function(weight, x)
        torch.cuda.synchronize()
        max_diff = (out - ref).abs().max().item()
        print(f"Max diff: {max_diff:.6f}")
        assert max_diff < 1e-2, f"Too large diff: {max_diff}"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(weight, x)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")