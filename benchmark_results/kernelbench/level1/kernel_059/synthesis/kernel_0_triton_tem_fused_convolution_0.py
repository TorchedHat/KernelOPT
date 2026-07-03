import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_D', 'IN_H', 'IN_W', 'OUT_C', 'OUT_D', 'OUT_H', 'OUT_W',
         'KD', 'KH', 'KW'],
)
@triton.jit
def conv3d_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_D: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_D: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KD: tl.constexpr, KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # Input strides
    s_xn: tl.constexpr, s_xc: tl.constexpr, s_xd: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr,
    # Weight strides
    s_wco: tl.constexpr, s_wci: tl.constexpr, s_wkd: tl.constexpr, s_wkh: tl.constexpr, s_wkw: tl.constexpr,
    # Output strides
    s_yn: tl.constexpr, s_yc: tl.constexpr, s_yd: tl.constexpr, s_yh: tl.constexpr, s_yw: tl.constexpr,
    M_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index -> (batch, out_d, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    tmp2 = tmp // OUT_H
    out_d_idx = tmp2 % OUT_D
    batch_idx = tmp2 // OUT_D

    # Total K = IN_C * KD * KH * KW
    TOTAL_K: tl.constexpr = IN_C * KD * KH * KW
    BLOCK_K_COUNT: tl.constexpr = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_block in range(BLOCK_K_COUNT):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

        # Decode K index -> (kd, kh, kw, in_c)
        # K layout: in_c varies fastest, then kw, kh, kd
        # Weight layout: [Co, Ci, KD, KH, KW] with strides s_wco, s_wci, s_wkd, s_wkh, s_wkw
        # Inductor uses: k = in_c + Ci*(kw + KW*(kh + KH*kd))
        # Actually from Inductor code: stride_wc_in=9, stride_wd=3, stride_wh=1, stride_ww=1
        # Weight shape [64,3,3,3,1] strides [27,9,3,1,1]
        # So weight[co, ci, kd, kh, kw] = W + co*27 + ci*9 + kd*3 + kh*1 + kw*1
        # K index: let's use ci varies slowest for better memory access
        # K = kd * KH * KW * IN_C + kh * KW * IN_C + kw * IN_C + ci
        # But Inductor uses: k = ci * (KD*KH*KW) + kd * KH * KW + kh * KW + kw
        # Let's match Inductor: k_block iterates over IN_C in inner loop
        # From Inductor: dijk = d*KH*KW*BLOCK_K_COUNT + h*KW*BLOCK_K_COUNT + w*BLOCK_K_COUNT + k_block
        # k = k_block * BLOCK_K -> ci offset
        # dij = dijk // BLOCK_K_COUNT -> d*KH*KW + h*KW + w
        # So K index = ci + IN_C * (kw + KW * (kh + KH * kd))
        # offs_k = k_block * BLOCK_K + arange(BLOCK_K)
        # ci = offs_k % IN_C ... no wait
        # Actually from Inductor loop structure:
        # for dijk in range(KD * KH * KW * BLOCK_K_COUNT):
        #   k = (dijk % BLOCK_K_COUNT) * BLOCK_K  -> ci offset
        #   dij = dijk // BLOCK_K_COUNT
        #   j = dij % KW; di = dij // KW; i = di % KH; d = di // KH
        # So the K dimension is: outer=(kd,kh,kw), inner=ci
        # K_flat = kd * KH * KW * IN_C + kh * KW * IN_C + kw * IN_C + ci
        # But Inductor iterates k (ci block) in inner loop
        # Let me just use: offs_k indexes into [kd, kh, kw, ci] with ci fastest
        ci = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KW
        tmp_k2 = tmp_k // KW
        kh_idx = tmp_k2 % KH
        kd_idx = tmp_k2 // KH

        # Input spatial coordinates
        in_d = out_d_idx[:, None] * STRIDE_D + kd_idx[None, :] - PAD_D
        in_h = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        in_w = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < TOTAL_K

        valid_spatial = (
            (batch_idx[:, None] < BATCH) &
            (in_d >= 0) & (in_d < IN_D) &
            (in_h >= 0) & (in_h < IN_H) &
            (in_w >= 0) & (in_w < IN_W) &
            valid_k[None, :]
        )

        # Input pointers [BLOCK_M, BLOCK_K]
        x_ptrs = (X_ptr +
                  batch_idx[:, None] * s_xn +
                  ci[None, :] * s_xc +

        x_tile = tl.load(x_ptrs, mask=valid_spatial, other=0.0)

        # Weight pointers [BLOCK_K, BLOCK_N]
        # weight[co, ci, kd, kh, kw]
        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_wco +
                  ci[:, None] * s_wci +
                  kd_idx[:, None] * s_wkd +
                  kh_idx[:, None] * s_wkh +
                  kw_idx[:, None] * s_wkw)

        valid_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output [BLOCK_M, BLOCK_N]
    # Output shape: [N, Co, OD, OH, OW]
    valid_out = (
        (offs_m[:, None] < M_TOTAL) &
        (offs_n[None, :] < OUT_C)
    )

    y_ptrs = (Y_ptr +
              batch_idx[:, None] * s_yn +
              offs_n[None, :] * s_yc +
              out_d_idx[:, None] * s_yd +
              out_h_idx[:, None] * s_yh +
              out_w_idx[:, None] * s_yw)

    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def kernel_function(weight, x):
    # weight: [64, 3, 3, 3, 1] (Co, Ci, KD, KH, KW)
    # x: [16, 3, 256, 256, 10] (N, Ci, D, H, W)
    # conv3d with kernel (3,3,1), stride=1, padding=0
    # Output: [16, 64, 254, 254, 10]

    N, Ci, D, H, W = x.shape
    Co, _, KD, KH, KW = weight.shape
    STRIDE_D, STRIDE_H, STRIDE_W = 1, 1, 1
    PAD_D, PAD_H, PAD_W = 0, 0, 0

    OD = (D + 2 * PAD_D - KD) // STRIDE_D + 1
    OH = (H + 2 * PAD_H - KH) // STRIDE_H + 1
    OW = (W + 2 * PAD_W - KW) // STRIDE_W + 1

    M_TOTAL = N * OD * OH * OW

    y = torch.empty((N, Co, OD, OH, OW), device=x.device, dtype=x.dtype)

    # Strides
    s_xn, s_xc, s_xd, s_xh, s_xw = x.stride()
    s_wco, s_wci, s_wkd, s_wkh, s_wkw = weight.stride()
    s_yn, s_yc, s_yd, s_yh, s_yw = y.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    # We need constexpr for BLOCK_M, BLOCK_N - use fixed values for autotune
    BLOCK_M = 256
    BLOCK_N = 64

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(Co, meta['BLOCK_N']))

    conv3d_kernel[grid](
        x, weight, y,
        N, Ci, D, H, W,
        Co, OD, OH, OW,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        s_xn, s_xc, s_xd, s_xh, s_xw,
        s_wco, s_wci, s_wkd, s_wkh, s_wkw,
        s_yn, s_yc, s_yd, s_yh, s_yw,
        M_TOTAL,
    )

    return y


def get_inputs():
    weight = torch.randn(64, 3, 3, 3, 1, device='cuda', dtype=torch.float32)
    x = torch.randn(16, 3, 256, 256, 10, device='cuda', dtype=torch.float32)
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify correctness
        import torch.nn.functional as F
        weight_ref = inputs[0]
        x_ref = inputs[1]
        ref = F.conv3d(x_ref, weight_ref, bias=None, stride=1, padding=0)
        print(f"Output shape: {output.shape}")
        print(f"Reference shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff}")
        assert max_diff < 1e-3, f"Max diff {max_diff} exceeds tolerance"
        print("Kernel executed successfully.")