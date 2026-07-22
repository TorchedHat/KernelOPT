import torch
import triton
import triton.language as tl
import argparse

# ─── Fused Conv2d (implicit GEMM, NHWC) + optional bias + optional ReLU ───────


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128,'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128,'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
    ],
    key=['M_TOTAL', 'OUT_C', 'IN_C', 'KH', 'KW'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    M_TOTAL, HAS_BIAS: tl.constexpr, HAS_RELU: tl.constexpr,
    # NHWC strides for X: (BATCH, IN_H, IN_W, IN_C)
    s_xn, s_xh, s_xw,  # s_xc = 1
    # NHWC strides for Y: (BATCH, OUT_H, OUT_W, OUT_C)
    s_yn, s_yh, s_yw,  # s_yc = 1
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp   = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    mask_m = offs_m < M_TOTAL
    mask_n = offs_n < OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)  -- layout: in_c fastest
        in_c = offs_k % IN_C
        kw_   = (offs_k // IN_C) % KW
        kh_   = offs_k // (IN_C * KW)

        ih = out_h[:, None] * STRIDE_H + kh_[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw_[None, :] - PAD_W

        valid_k = offs_k < TOTAL_K
        valid_x = (
            mask_m[:, None]
            & (ih >= 0) & (ih < IN_H)
            & (iw >= 0) & (iw < IN_W)
        )

        x_ptrs = X_ptr + batch[:, None] * s_xn + ih * s_xh + iw * s_xw + in_c[None, :]
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight layout: (OUT_C, KH, KW, IN_C) -> BLOCK_K x BLOCK_N
        # w index: offs_n * KH*KW*IN_C + kh_*KW*IN_C + kw_*IN_C + in_c
        w_ptrs = W_ptr + offs_n[None, :] * (KH * KW * IN_C) + kh_[:, None] * (KW * IN_C) + kw_[:, None] * IN_C + in_c[:, None]
        valid_w = valid_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    if HAS_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store in NHWC
    y_ptrs = Y_ptr + batch[:, None] * s_yn + out_h[:, None] * s_yh + out_w[:, None] * s_yw + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit

def conv2d_bias_relu(x_nhwc, w_oihw, bias, stride=1, padding=1, relu=True):
    """
    returns y_nhwc: (N, OUT_H, OUT_W, OUT_C)
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w_oihw.shape
    SH = SW = stride
    PH = PW = padding

    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1
    M_TOTAL = N * OUT_H * OUT_W

    y_nhwc = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    # Flatten weight to (OUT_C, KH*KW*IN_C) -- but kernel uses (OUT_C, KH, KW, IN_C) indexing
    # We pass w_oihw directly and compute index inside kernel

    s_xn = IN_H * IN_W * IN_C
    s_xh = IN_W * IN_C
    s_xw = IN_C
    s_yn = OUT_H * OUT_W * OUT_C
    s_yh = OUT_W * OUT_C
    s_yw = OUT_C

    HAS_BIAS = bias is not None
    HAS_RELU = relu

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        x_nhwc, w_oihw, bias if HAS_BIAS else x_nhwc, y_nhwc,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        M_TOTAL, HAS_BIAS, HAS_RELU,
        s_xn, s_xh, s_xw,
        s_yn, s_yh, s_yw,
    )
    return y_nhwc


# ─── MaxPool2d kernel ──────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # offs indexes (n, out_h, out_w, c) in NHWC
    c     = offs % C
    tmp   = offs // C
    out_w = tmp % OUT_W
    tmp2  = tmp // OUT_W
    out_h = tmp2 % OUT_H
    n     = tmp2 // OUT_H

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            ih = out_h * SH + kh
            iw = out_w * SW + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_idx = n * (IN_H * IN_W * C) + ih * (IN_W * C) + iw * C + c
            val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    y_idx = n * (OUT_H * OUT_W * C) + out_h * (OUT_W * C) + out_w * C + c
    tl.store(Y_ptr + y_idx, max_val, mask=mask)


def maxpool2d_nhwc(x_nhwc, kernel_size=2, stride=2):
    N, IN_H, IN_W, C = x_nhwc.shape
    KH = KW = kernel_size
    SH = SW = stride
    OUT_H = (IN_H - KH) // SH + 1
    OUT_W = (IN_W - KW) // SW + 1
    y_nhwc = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)
    N_TOTAL = N * OUT_H * OUT_W * C
    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x_nhwc, y_nhwc,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW,
        N_TOTAL,
    )
    return y_nhwc


# ─── Fused Linear + Bias + optional ReLU ──────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    HAS_BIAS: tl.constexpr, HAS_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # W is (N, K), so W[n, k]
        w_ptrs = W_ptr + offs_n[:, None] * K + offs_k[None, :]
        w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    if HAS_RELU:
        acc = tl.maximum(acc, 0.0)

    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def linear_bias_relu(x, weight, bias, relu=True):
    """
    returns: (M, N)
    """
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    HAS_BIAS = bias is not None
    HAS_RELU = relu
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_bias_relu_kernel[grid](
        x, weight, bias if HAS_BIAS else x, y,
        M, N, K,
        HAS_BIAS, HAS_RELU,
    )
    return y


# ─── Main kernel_function ──────────────────────────────────────────────────────

def kernel_function(
    p_features_0_weight, p_features_0_bias,
    p_features_2_weight, p_features_2_bias,
    p_features_5_weight, p_features_5_bias,
    p_features_7_weight, p_features_7_bias,
    p_features_10_weight, p_features_10_bias,
    p_features_12_weight, p_features_12_bias,
    p_features_14_weight, p_features_14_bias,
    p_features_17_weight, p_features_17_bias,
    p_features_19_weight, p_features_19_bias,
    p_features_21_weight, p_features_21_bias,
    p_features_24_weight, p_features_24_bias,
    p_features_26_weight, p_features_26_bias,
    p_features_28_weight, p_features_28_bias,
    p_classifier_0_weight, p_classifier_0_bias,
    p_classifier_3_weight, p_classifier_3_bias,
    p_classifier_6_weight, p_classifier_6_bias,
    x,
):
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    def make_w_nhwc(w):
        # w: (OUT_C, IN_C, KH, KW) -> keep as is, kernel handles indexing
        return w.contiguous()

    # Block 1
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_0_weight), p_features_0_bias, stride=1, padding=1, relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_2_weight), p_features_2_bias, stride=1, padding=1, relu=True)
    x_nhwc = maxpool2d_nhwc(x_nhwc, kernel_size=2, stride=2)

    # Block 2
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_5_weight), p_features_5_bias, stride=1, padding=1, relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_7_weight), p_features_7_bias, stride=1, padding=1, relu=True)
    x_nhwc = maxpool2d_nhwc(x_nhwc, kernel_size=2, stride=2)

    # Block 3
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_10_weight), p_features_10_bias, stride=1, padding=1, relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_12_weight), p_features_12_bias, stride=1, padding=1, relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_14_weight), p_features_14_bias, stride=1, padding=1, relu=True)
    x_nhwc = maxpool2d_nhwc(x_nhwc, kernel_size=2, stride=2)

    # Block 4
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_17_weight), p_features_17_bias, stride=1, padding=1, relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_19_weight), p_features_19_bias, stride=1, padding=1, relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_21_weight), p_features_21_bias, stride=1, padding=1, relu=True)
    x_nhwc = maxpool2d_nhwc(x_nhwc, kernel_size=2, stride=2)

    # Block 5
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_24_weight), p_features_24_bias, stride=1, padding=1, relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_26_weight), p_features_26_bias, stride=1, padding=1, relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, make_w_nhwc(p_features_28_weight), p_features_28_bias, stride=1, padding=1, relu=True)
    x_nhwc = maxpool2d_nhwc(x_nhwc, kernel_size=2, stride=2)

    # Flatten: (N, 7, 7, 512) -> (N, 25088)
    N = x_nhwc.shape[0]
    # Convert back to NCHW order for flatten: (N, H, W, C) -> (N, C, H, W) -> flatten
    # But we need (N, 512*7*7) = (N, 25088) matching PyTorch's flatten(x, 1) after NCHW
    # PyTorch stores as NCHW: flatten gives N, C*H*W = N, 512*7*7
    # Our NHWC: N, H, W, C -> need to reorder to match NCHW flatten
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_flat = x_nchw.reshape(N, -1)

    # Classifier
    x_flat = linear_bias_relu(x_flat, p_classifier_0_weight, p_classifier_0_bias, relu=True)
    # dropout p=0.0 -> identity
    x_flat = linear_bias_relu(x_flat, p_classifier_3_weight, p_classifier_3_bias, relu=True)
    # dropout p=0.0 -> identity
    x_flat = linear_bias_relu(x_flat, p_classifier_6_weight, p_classifier_6_bias, relu=False)

    return x_flat


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'
    return [
        torch.randn(64, 3, 3, 3, device=device),      # p_features_0_weight
        torch.randn(64, device=device),                 # p_features_0_bias
        torch.randn(64, 64, 3, 3, device=device),      # p_features_2_weight
        torch.randn(64, device=device),                 # p_features_2_bias
        torch.randn(128, 64, 3, 3, device=device),     # p_features_5_weight
        torch.randn(128, device=device),                # p_features_5_bias
        torch.randn(128, 128, 3, 3, device=device),    # p_features_7_weight
        torch.randn(128, device=device),                # p_features_7_bias
        torch.randn(256, 128, 3, 3, device=device),    # p_features_10_weight
        torch.randn(256, device=device),                # p_features_10_bias
        torch.randn(256, 256, 3, 3, device=device),    # p_features_12_weight
        torch.randn(256, device=device),                # p_features_12_bias
        torch.randn(256, 256, 3, 3, device=device),    # p_features_14_weight
        torch.randn(256, device=device),                # p_features_14_bias
        torch.randn(512, 256, 3, 3, device=device),    # p_features_17_weight
        torch.randn(512, device=device),                # p_features_17_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_19_weight
        torch.randn(512, device=device),                # p_features_19_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_21_weight
        torch.randn(512, device=device),                # p_features_21_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_24_weight
        torch.randn(512, device=device),                # p_features_24_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_26_weight
        torch.randn(512, device=device),                # p_features_26_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_28_weight
        torch.randn(512, device=device),                # p_features_28_bias
        torch.randn(4096, 25088, device=device),        # p_classifier_0_weight
        torch.randn(4096, device=device),               # p_classifier_0_bias
        torch.randn(4096, 4096, device=device),         # p_classifier_3_weight
        torch.randn(4096, device=device),               # p_classifier_3_bias
        torch.randn(1000, 4096, device=device),         # p_classifier_6_weight
        torch.randn(1000, device=device),               # p_classifier_6_bias
        torch.randn(10, 3, 224, 224, device=device),   # x
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn
        import torch.nn.functional as F

        # Build reference model
        (p_features_0_weight, p_features_0_bias,
         p_features_2_weight, p_features_2_bias,
         p_features_5_weight, p_features_5_bias,
         p_features_7_weight, p_features_7_bias,
         p_features_10_weight, p_features_10_bias,
         p_features_12_weight, p_features_12_bias,
         p_features_14_weight, p_features_14_bias,
         p_features_17_weight, p_features_17_bias,
         p_features_19_weight, p_features_19_bias,
         p_features_21_weight, p_features_21_bias,
         p_features_24_weight, p_features_24_bias,
         p_features_26_weight, p_features_26_bias,
         p_features_28_weight, p_features_28_bias,
         p_classifier_0_weight, p_classifier_0_bias,
         p_classifier_3_weight, p_classifier_3_bias,
         p_classifier_6_weight, p_classifier_6_bias,
         x) = inputs

        def ref_conv_relu(x, w, b, padding=1):
            return F.relu(F.conv2d(x, w, b, padding=padding))

        def ref_linear_relu(x, w, b):
            return F.relu(F.linear(x, w, b))

        with torch.no_grad():
            r = x
            r = ref_conv_relu(r, p_features_0_weight, p_features_0_bias)
            r = ref_conv_relu(r, p_features_2_weight, p_features_2_bias)
            r = F.max_pool2d(r, 2, 2)
            r = ref_conv_relu(r, p_features_5_weight, p_features_5_bias)
            r = ref_conv_relu(r, p_features_7_weight, p_features_7_bias)
            r = F.max_pool2d(r, 2, 2)
            r = ref_conv_relu(r, p_features_10_weight, p_features_10_bias)
            r = ref_conv_relu(r, p_features_12_weight, p_features_12_bias)
            r = ref_conv_relu(r, p_features_14_weight, p_features_14_bias)
            r = F.max_pool2d(r, 2, 2)
            r = ref_conv_relu(r, p_features_17_weight, p_features_17_bias)
            r = ref_conv_relu(r, p_features_19_weight, p_features_19_bias)
            r = ref_conv_relu(r, p_features_21_weight, p_features_21_bias)
            r = F.max_pool2d(r, 2, 2)
            r = ref_conv_relu(r, p_features_24_weight, p_features_24_bias)
            r = ref_conv_relu(r, p_features_26_weight, p_features_26_bias)
            r = ref_conv_