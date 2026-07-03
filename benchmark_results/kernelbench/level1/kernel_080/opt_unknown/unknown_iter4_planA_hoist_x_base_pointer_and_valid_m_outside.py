import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        # Original 8 configs (large tiles)
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        # New 8 small-tile configs targeting lower register pressure
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2, num_stages=5),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=5),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=2, num_stages=4),
        # New larger-BLOCK_N configs for coalesced transposed weight loads
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    DIL_H, DIL_W,
    s_xn, s_xc, s_xh, s_xw,
    s_wn, s_wc, s_wh, s_ww,
    s_yn, s_yc, s_yh, s_yw,
    M_TOTAL, TOTAL_K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # HOISTED: M-dimension decodes — loop-invariant, compute once before K-loop
    out_w_1d = offs_m % OUT_W                    # [BLOCK_M]
    out_h_1d = (offs_m // OUT_W) % OUT_H        # [BLOCK_M]
    batch_1d = offs_m // (OUT_W * OUT_H)        # [BLOCK_M]

    # HOISTED: Precompute scaled output coords and base X pointer — loop-invariant
    out_h_scaled = out_h_1d * STRIDE_H          # [BLOCK_M]
    out_w_scaled = out_w_1d * STRIDE_W          # [BLOCK_M]
    base_x_m = X_ptr + batch_1d * s_xn + out_h_scaled * s_xh + out_w_scaled * s_xw  # [BLOCK_M]

    # HOISTED: valid_m is loop-invariant — compute once before K-loop
    valid_m = offs_m[:, None] < M_TOTAL         # [BLOCK_M, 1]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kh, kw)
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        in_c = offs_k // (KH * KW)

        # K-varying deltas only (M-side already in base_x_m)
        dh = kh[None, :] * DIL_H - PAD_H        # [1, BLOCK_K]
        dw = kw[None, :] * DIL_W - PAD_W        # [1, BLOCK_K]

        # Full absolute coordinates for spatial bounds check
        ih = out_h_scaled[:, None] + dh          # [BLOCK_M, BLOCK_K]
        iw = out_w_scaled[:, None] + dw          # [BLOCK_M, BLOCK_K]

        valid_k = offs_k[None, :] < TOTAL_K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        valid_x = valid_m & valid_k & valid_spatial

        # x_ptrs: use base_x_m (already includes M-side offsets) + K-varying terms only
        x_ptrs = base_x_m[:, None] + in_c[None, :] * s_xc + dh * s_xh + dw * s_xw
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Transposed weight load: shape [BLOCK_N, BLOCK_K] for coalesced K-dimension access
        # offs_n[:, None] -> [BLOCK_N, 1], offs_k[None, :] -> [1, BLOCK_K]
        valid_wn_col = offs_n[:, None] < OUT_C   # [BLOCK_N, 1]
        valid_wk_row = offs_k[None, :] < TOTAL_K  # [1, BLOCK_K]
        w_ptrs_T = W_ptr + offs_n[:, None] * s_wn + in_c[None, :] * s_wc + kh[None, :] * s_wh + kw[None, :] * s_ww
        w_tile_T = tl.load(w_ptrs_T, mask=valid_wn_col & valid_wk_row, other=0.0)  # [BLOCK_N, BLOCK_K]

        acc = tl.dot(x_tile, tl.trans(w_tile_T), acc)

    # Store output — use hoisted names
    valid_out_m = offs_m[:, None] < M_TOTAL
    valid_out_n = offs_n[None, :] < OUT_C

    y_ptrs = Y_ptr + batch_1d[:, None] * s_yn + offs_n[None, :] * s_yc + out_h_1d[:, None] * s_yh + out_w_1d[:, None] * s_yw
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out_m & valid_out_n)

def kernel_function(weight, x):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 2, 4
    DIL_H, DIL_W = 2, 3

    OUT_H = (IN_H + 2 * PAD_H - DIL_H * (KH - 1) - 1) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - DIL_W * (KW - 1) - 1) // STRIDE_W + 1

    M_TOTAL = N * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    s_xn, s_xc, s_xh, s_xw = x.stride()
    s_wn, s_wc, s_wh, s_ww = weight.stride()
    s_yn, s_yc, s_yh, s_yw = y.stride()

    def cdiv(a, b): return (a + b - 1) // b

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))

    conv2d_implicit_gemm_kernel[grid](
        x, weight, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        DIL_H, DIL_W,
        s_xn, s_xc, s_xh, s_xw,
        s_wn, s_wc, s_wh, s_ww,
        s_yn, s_yc, s_yh, s_yw,
        M_TOTAL, TOTAL_K,
    )

    return y


def get_inputs():
    weight = torch.randn(64, 32, 5, 9, dtype=torch.float32, device='cuda')
    x = torch.randn(8, 32, 512, 512, dtype=torch.float32, device='cuda')
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
        ref = F.conv2d(x, weight, bias=None, stride=1, padding=(2, 4), dilation=(2, 3))
        out = kernel_function(weight, x)
        torch.cuda.synchronize()
        max_diff = (out - ref).abs().max().item()
        print(f"Max diff: {max_diff}")
        assert max_diff < 1e-3, f"Max diff too large: {max_diff}"
        print("Correctness check passed!")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")