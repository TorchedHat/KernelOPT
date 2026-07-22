import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused Conv2d (implicit GEMM) + Bias + ReLU kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    # Tensor dimensions
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    # Strides for X (NHWC layout)
    stride_xn, stride_xh, stride_xw, stride_xc,
    # Strides for W (NHWC layout: out_c, kh, kw, in_c)
    stride_wn, stride_wh, stride_ww, stride_wc,
    # Strides for Y (NHWC layout)
    stride_yn, stride_yh, stride_yw, stride_yc,
    # Conv params
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # Problem sizes for autotune key
    M, N, K,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    # Tile sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # Whether to apply ReLU
    APPLY_RELU: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c_idx = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KERNEL_W
        kh_idx = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            (batch_idx[:, None] < BATCH) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_ptrs = (
            batch_idx[:, None] * stride_xn +
            in_c_idx[None, :] * stride_xc
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: shape (OUT_C, KERNEL_H, KERNEL_W, IN_C) in NHWC-like layout
        # W stored as (out_c, kh, kw, in_c)
        valid_w = (
            (offs_k[:, None] < TOTAL_K) &
            (offs_n[None, :] < OUT_C)
        )
        w_ptrs = (
            offs_n[None, :] * stride_wn +
            kh_idx[:, None] * stride_wh +
            kw_idx[:, None] * stride_ww +
            in_c_idx[:, None] * stride_wc
        )
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output
    valid_out = (
        (batch_idx[:, None] < BATCH) &
        (out_h_idx[:, None] < OUT_H) &
        (out_w_idx[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )

    y_ptrs = (
        batch_idx[:, None] * stride_yn +
        out_h_idx[:, None] * stride_yh +
        out_w_idx[:, None] * stride_yw +
        offs_n[None, :] * stride_yc
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def conv2d_relu_triton(x_nhwc, w, bias, stride_h, stride_w, pad_h, pad_w, apply_relu=True):
    """
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w.shape

    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1

    # Permute weight to (OUT_C, KH, KW, IN_C)
    w_perm = w.permute(0, 2, 3, 1).contiguous()

    y = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    M = N * OUT_H * OUT_W
    Nk = OUT_C
    K = IN_C * KH * KW

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Nk, meta['BLOCK_N']),
    )

    conv2d_relu_kernel[grid](
        x_nhwc, w_perm, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w_perm.stride(0), w_perm.stride(1), w_perm.stride(2), w_perm.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        stride_h, stride_w,
        pad_h, pad_w,
        M, Nk, K,
        KH, KW,
        APPLY_RELU=apply_relu,
    )
    return y


# ============================================================
# Max Pool 2D kernel (ceil_mode=True, 3x3, stride=2)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: offs -> (n, c, oh, ow)
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            # X is NHWC
            x_idx = n * (IN_H * IN_W * C) + ih * (IN_W * C) + iw * C + c
            val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    # Y is NHWC
    y_idx = n * (OUT_H * OUT_W * C) + oh * (OUT_W * C) + ow * C + c
    tl.store(Y_ptr + y_idx, max_val, mask=mask)


def maxpool2d_triton(x_nhwc, kh, kw, stride_h, stride_w, pad_h, pad_w, ceil_mode=True):
    N, IN_H, IN_W, C = x_nhwc.shape

    if ceil_mode:
        OUT_H = math.ceil((IN_H + 2 * pad_h - kh) / stride_h) + 1
        OUT_W = math.ceil((IN_W + 2 * pad_w - kw) / stride_w) + 1
        # Clamp to valid range
        OUT_H = min(OUT_H, math.ceil(IN_H / stride_h))
        OUT_W = min(OUT_W, math.ceil(IN_W / stride_w))
    else:
        OUT_H = (IN_H + 2 * pad_h - kh) // stride_h + 1
        OUT_W = (IN_W + 2 * pad_w - kw) // stride_w + 1

    y = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)
    N_TOTAL = N * C * OUT_H * OUT_W

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x_nhwc, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kh, kw, stride_h, stride_w, pad_h, pad_w,
        N_TOTAL,
    )
    return y


# ============================================================
# Adaptive Average Pool 2D kernel (output 1x1)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_TOTAL', 'IN_H', 'IN_W'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    """Adaptive avg pool to 1x1 output. X is NHWC."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # offs -> (n, c)
    c = offs % C
    n = offs // C

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    total = IN_H * IN_W

    for h in range(IN_H):
        for w in range(IN_W):
            x_idx = n * (IN_H * IN_W * C) + h * (IN_W * C) + w * C + c
            val = tl.load(X_ptr + x_idx, mask=mask, other=0.0)
            acc = acc + val

    acc = acc / total
    tl.store(Y_ptr + offs, acc, mask=mask)


def adaptive_avg_pool2d_triton(x_nhwc):
    N, IN_H, IN_W, C = x_nhwc.shape
    y = torch.empty((N, 1, 1, C), device=x_nhwc.device, dtype=torch.float32)
    N_TOTAL = N * C

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x_nhwc, y,
        N, C, IN_H, IN_W,
        N_TOTAL,
    )
    return y


# ============================================================
# Fire Module helper
# ============================================================

def fire_module_triton(x_nhwc, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """
    """
    # Squeeze: 1x1 conv + relu
    sq = conv2d_relu_triton(x_nhwc, sq_w, sq_b, 1, 1, 0, 0, apply_relu=True)

    # Expand 1x1 + relu
    ex1 = conv2d_relu_triton(sq, ex1_w, ex1_b, 1, 1, 0, 0, apply_relu=True)

    # Expand 3x3 + relu
    ex3 = conv2d_relu_triton(sq, ex3_w, ex3_b, 1, 1, 1, 1, apply_relu=True)

    # Cat along channel dim
    out = torch.cat([ex1, ex3], dim=3)
    return out


# ============================================================
# Main kernel_function
# ============================================================

def kernel_function(
):
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    x_nhwc = conv2d_relu_triton(x_nhwc, p_features_0_weight, p_features_0_bias,
                                 2, 2, 0, 0, apply_relu=True)
    # -> (64, 253, 253, 96)

    # features[1]: MaxPool2d(3, stride=2, ceil_mode=True)
    x_nhwc = maxpool2d_triton(x_nhwc, 3, 3, 2, 2, 0, 0, ceil_mode=True)
    # -> (64, 126, 126, 96)  [ceil((253-3)/2)+1 = ceil(125)+1 = 126]

    # features[3]: FireModule(96, 16, 64, 64)
    x_nhwc = fire_module_triton(x_nhwc,
                                 p_features_3_squeeze_weight, p_features_3_squeeze_bias,
                                 p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,
    # -> (64, 126, 126, 128)

    # features[4]: FireModule(128, 16, 64, 64)
    x_nhwc = fire_module_triton(x_nhwc,
                                 p_features_4_squeeze_weight, p_features_4_squeeze_bias,
                                 p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,
    # -> (64, 126, 126, 128)

    # features[5]: FireModule(128, 32, 128, 128)
    x_nhwc = fire_module_triton(x_nhwc,
                                 p_features_5_squeeze_weight, p_features_5_squeeze_bias,
                                 p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,
    # -> (64, 126, 126, 256)

    # features[6]: MaxPool2d(3, stride=2, ceil_mode=True)
    x_nhwc = maxpool2d_triton(x_nhwc, 3, 3, 2, 2, 0, 0, ceil_mode=True)
    # -> (64, 63, 63, 256)

    # features[7]: FireModule(256, 32, 128, 128)
    x_nhwc = fire_module_triton(x_nhwc,
                                 p_features_7_squeeze_weight, p_features_7_squeeze_bias,
                                 p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,
    # -> (64, 63, 63, 256)

    # features[8]: FireModule(256, 48, 192, 192)
    x_nhwc = fire_module_triton(x_nhwc,
                                 p_features_8_squeeze_weight, p_features_8_squeeze_bias,
                                 p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,
    # -> (64, 63, 63, 384)

    # features[9]: FireModule(384, 48, 192, 192)
    x_nhwc = fire_module_triton(x_nhwc,
                                 p_features_9_squeeze_weight, p_features_9_squeeze_bias,
                                 p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,
    # -> (64, 63, 63, 384)

    # features[10]: FireModule(384, 64, 256, 256)
    x_nhwc = fire_module_triton(x_nhwc,
                                 p_features_10_squeeze_weight, p_features_10_squeeze_bias,
                                 p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,
    # -> (64, 63, 63, 512)

    # features[11]: MaxPool2d(3, stride=2, ceil_mode=True)
    x_nhwc = maxpool2d_triton(x_nhwc, 3, 3, 2, 2, 0, 0, ceil_mode=True)
    # -> (64, 31, 31, 512)

    # features[12]: FireModule(512, 64, 256, 256)
    x_nhwc = fire_module_triton(x_nhwc,
                                 p_features_12_squeeze_weight, p_features_12_squeeze_bias,
                                 p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,
    # -> (64, 31, 31, 512)

    # classifier[0]: Dropout(p=0.0) - identity at inference
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    x_nhwc = conv2d_relu_triton(x_nhwc, p_classifier_1_weight, p_classifier_1_bias,
                                 1, 1, 0, 0, apply_relu=True)
    # -> (64, 31, 31, 1000)

    # classifier[3]: AdaptiveAvgPool2d(1, 1)
    x_nhwc = adaptive_avg_pool2d_triton(x_nhwc)
    # -> (64, 1, 1, 1000)

    # Convert back to NCHW and flatten
    # x_nhwc is (64, 1, 1, 1000) -> flatten to (64, 1000)
    out = x_nhwc.reshape(x_nhwc.shape[0], -1)
    return out


def get_inputs():
    batch_size = 64
    return [
        torch.randn(96, 3, 7, 7, dtype=torch.float32, device='cuda'),      # p_features_0_weight
        torch.randn(96, dtype=torch.float32, device='cuda'),                 # p_features_0_bias
        torch.randn(16, 96, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_3_squeeze_weight
        torch.randn(16, dtype=torch.float32, device='cuda'),                 # p_features_3_squeeze_bias
        torch.randn(64, 16, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_3_expand1x1_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_3_expand1x1_bias
        torch.randn(64, 16, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_3_expand3x3_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_3_expand3x3_bias
        torch.randn(16, 128, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_4_squeeze_weight
        torch.randn(16, dtype=torch.float32, device='cuda'),                 # p_features_4_squeeze_bias
        torch.randn(64, 16, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_4_expand1x1_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_4_expand1x1_bias
        torch.randn(64, 16, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_4_expand3x3_weight
        torch.randn