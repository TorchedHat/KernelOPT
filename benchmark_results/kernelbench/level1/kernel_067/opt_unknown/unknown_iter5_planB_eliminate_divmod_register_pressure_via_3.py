import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # NEW: Large-tile configs restored from pre-regression baseline
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        # EXISTING: Keep all 8 current configs unchanged below
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
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
    # 3D logical decomposition using 2D physical grid:
    # pid_m encodes (batch, l_out_tile) — we decode batch as scalar per-program
    # by using pid_m // cdiv(L_OUT, BLOCK_M) and pid_m % cdiv(L_OUT, BLOCK_M)
    # This replaces the BLOCK_M-wide batch/l_out vectors with scalars.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Number of L_OUT tiles per batch element (compile-time expression using runtime val)
    # We compute: num_lout_tiles = cdiv(L_OUT, BLOCK_M)
    # batch_scalar = pid_m // num_lout_tiles  (scalar)
    # pid_lout     = pid_m %  num_lout_tiles  (scalar)
    num_lout_tiles = (L_OUT + BLOCK_M - 1) // BLOCK_M
    batch_scalar = pid_m // num_lout_tiles
    pid_lout     = pid_m - batch_scalar * num_lout_tiles  # equivalent to pid_m % num_lout_tiles

    # l_out offsets: 1D vector, no batch dimension needed
    l_out_offs = pid_lout * BLOCK_M + tl.arange(0, BLOCK_M)  # (BLOCK_M,)
    offs_n     = pid_n   * BLOCK_N + tl.arange(0, BLOCK_N)   # (BLOCK_N,)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for kk in range(KERNEL_SIZE):
        in_pos_base = l_out_offs * STRIDE_CONV + kk * DILATION - PAD  # (BLOCK_M,)
        mask_pos = (in_pos_base >= 0) & (in_pos_base < L_IN)

        for c_start in range(0, C_IN, BLOCK_K):
            offs_c = c_start + tl.arange(0, BLOCK_K)  # (BLOCK_K,)
            mask_c = offs_c < C_IN

            # Input tile: (BLOCK_M, BLOCK_K)
            # batch_scalar is a scalar — no batch vector needed
            mask_x = (l_out_offs[:, None] < L_OUT) & mask_pos[:, None] & mask_c[None, :]
            x_ptrs = (X_ptr
                      + batch_scalar * stride_xn
                      + offs_c[None, :] * stride_xc
                      + in_pos_base[:, None] * stride_xl)
            x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

            # Weight tile: (BLOCK_K, BLOCK_N)
            mask_w = mask_c[:, None] & (offs_n[None, :] < N_OUT)
            w_ptrs = W_ptr + offs_n[None, :] * stride_wco + offs_c[:, None] * stride_wci + kk * stride_wk
            w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

            acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=True)

    # Store output
    mask_out = (l_out_offs[:, None] < L_OUT) & (offs_n[None, :] < N_OUT)
    y_ptrs = (Y_ptr
              + batch_scalar * stride_yn
              + offs_n[None, :] * stride_yco
              + l_out_offs[:, None] * stride_yl)
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