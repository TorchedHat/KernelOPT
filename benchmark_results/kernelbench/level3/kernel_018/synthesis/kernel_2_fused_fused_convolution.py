import torch
import triton
import triton.language as tl
import math

# ============================================================
# Implicit GEMM Conv2d + ReLU Triton Kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    # Tensor dimensions
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # Strides for X (NHWC layout)
    sx_n, sx_h, sx_w, sx_c,
    # Strides for W (NCHW: out_c, in_c, kh, kw -> stored as out_c, kh, kw, in_c for NHWC)
    sw_oc, sw_kh, sw_kw, sw_ic,
    # Strides for Y (NHWC layout)
    sy_n, sy_h, sy_w, sy_c,
    # Problem size for autotuning
    M, N, K,
    # Apply ReLU?
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

    # K = IN_C * KH * KW
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KW
        kh_idx = tmp_k // KW

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = (batch_idx < BATCH) & (out_h_idx < OUT_H) & (out_w_idx < OUT_W)
        valid_k = offs_k < K
        valid_x = valid_m[:, None] & valid_k[None, :] & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)

        x_ptrs = X_ptr + (
            batch_idx[:, None] * sx_n +
            in_c_idx[None, :] * sx_c
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: shape (OUT_C, KH, KW, IN_C) in memory
        valid_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + (
            offs_n[None, :] * sw_oc +
            kh_idx[:, None] * sw_kh +
            kw_idx[:, None] * sw_kw +
            in_c_idx[:, None] * sw_ic
        )
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output (NHWC)
    valid_out = (
        (batch_idx[:, None] < BATCH) &
        (out_h_idx[:, None] < OUT_H) &
        (out_w_idx[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )
    y_ptrs = Y_ptr + (
        batch_idx[:, None] * sy_n +
        out_h_idx[:, None] * sy_h +
        out_w_idx[:, None] * sy_w +
        offs_n[None, :] * sy_c
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def conv2d_relu(x_nhwc, w_nhwc, bias, stride_h, stride_w, pad_h, pad_w, apply_relu=True):
    """
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, KH, KW, _ = w_nhwc.shape
    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1

    M = N * OUT_H * OUT_W
    K_dim = IN_C * KH * KW
    N_dim = OUT_C

    y = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_dim, meta['BLOCK_N']),
    )

    conv2d_relu_kernel[grid](
        x_nhwc, w_nhwc, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        # X strides (NHWC)
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        # W strides (OC, KH, KW, IC)
        w_nhwc.stride(0), w_nhwc.stride(1), w_nhwc.stride(2), w_nhwc.stride(3),
        # Y strides (NHWC)
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        M, N_dim, K_dim,
        APPLY_RELU=apply_relu,
    )
    return y


# ============================================================
# Max Pool 2D Triton Kernel (ceil_mode=True, kernel=3, stride=2)
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
    # X strides (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    # Y strides (NHWC)
    sy_n, sy_h, sy_w, sy_c,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: N_TOTAL = BATCH * OUT_H * OUT_W * C
    c_idx = offs % C
    tmp = offs // C
    ow_idx = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh_idx = tmp2 % OUT_H
    n_idx = tmp2 // OUT_H

    result = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh_idx * STRIDE_H + kh - PAD_H
            iw = ow_idx * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (n_idx < BATCH)
            x_ptrs = X_ptr + n_idx * sx_n + ih * sx_h + iw * sx_w + c_idx * sx_c
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            result = tl.maximum(result, val)

    y_ptrs = Y_ptr + n_idx * sy_n + oh_idx * sy_h + ow_idx * sy_w + c_idx * sy_c
    tl.store(y_ptrs, result, mask=mask)


def maxpool2d_ceil(x_nhwc, kernel_size=3, stride=2, padding=0):
    """
    Returns: (N, OH, OW, C) float32 with ceil_mode=True
    """
    N, IN_H, IN_W, C = x_nhwc.shape
    KH = KW = kernel_size
    SH = SW = stride
    PAD = padding

    # ceil_mode
    OUT_H = math.ceil((IN_H + 2 * PAD - KH) / SH) + 1
    OUT_W = math.ceil((IN_W + 2 * PAD - KW) / SW) + 1

    y = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)
    N_TOTAL = N * OUT_H * OUT_W * C

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x_nhwc, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW, PAD, PAD,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        N_TOTAL,
    )
    return y


# ============================================================
# Adaptive Avg Pool 2D (output 1x1) Triton Kernel
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
    sx_n, sx_h, sx_w, sx_c,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    """Each thread handles one (n, c) pair, averaging over H*W."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    c_idx = offs % C
    n_idx = offs // C

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for h in range(IN_H):
        for w in range(IN_W):
            x_ptrs = X_ptr + n_idx * sx_n + h * sx_h + w * sx_w + c_idx * sx_c
            val = tl.load(x_ptrs, mask=mask, other=0.0)
            acc = acc + val

    total = IN_H * IN_W
    result = acc / total
    # Output is (N, 1, 1, C) -> store at n*C + c
    tl.store(Y_ptr + offs, result, mask=mask)


def adaptive_avg_pool2d_1x1(x_nhwc):
    """
    """
    N, IN_H, IN_W, C = x_nhwc.shape
    y = torch.empty((N, C), device=x_nhwc.device, dtype=torch.float32)
    N_TOTAL = N * C

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x_nhwc, y,
        N, C, IN_H, IN_W,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        N_TOTAL,
    )
    return y.view(N, 1, 1, C)


# ============================================================
# Helper: convert weight from NCHW to NHWC-like (OC, KH, KW, IC)
# ============================================================
def prepare_weight(w_nchw):
    """w_nchw: (OC, IC, KH, KW) -> (OC, KH, KW, IC) contiguous"""
    return w_nchw.permute(0, 2, 3, 1).contiguous()


# ============================================================
# Fire Module helper
# ============================================================
def fire_module(x_nhwc,
                sq_w, sq_b,
                ex1_w, ex1_b,
    """
    """
    # Squeeze
    sq = conv2d_relu(x_nhwc, sq_w, sq_b, 1, 1, 0, 0, apply_relu=True)
    # Expand 1x1
    ex1 = conv2d_relu(sq, ex1_w, ex1_b, 1, 1, 0, 0, apply_relu=True)
    # Expand 3x3
    ex3 = conv2d_relu(sq, ex3_w, ex3_b, 1, 1, 1, 1, apply_relu=True)
    # Cat along channel dim (last dim in NHWC)
    return torch.cat([ex1, ex3], dim=3)


# ============================================================
# Main kernel_function
# ============================================================
def kernel_function(
    p_features_0_weight,
    p_features_0_bias,
    p_features_3_squeeze_weight,
    p_features_3_squeeze_bias,
    p_features_3_expand1x1_weight,
    p_features_3_expand1x1_bias,
    p_features_3_expand3x3_weight,
    p_features_3_expand3x3_bias,
    p_features_4_squeeze_weight,
    p_features_4_squeeze_bias,
    p_features_4_expand1x1_weight,
    p_features_4_expand1x1_bias,
    p_features_4_expand3x3_weight,
    p_features_4_expand3x3_bias,
    p_features_5_squeeze_weight,
    p_features_5_squeeze_bias,
    p_features_5_expand1x1_weight,
    p_features_5_expand1x1_bias,
    p_features_5_expand3x3_weight,
    p_features_5_expand3x3_bias,
    p_features_7_squeeze_weight,
    p_features_7_squeeze_bias,
    p_features_7_expand1x1_weight,
    p_features_7_expand1x1_bias,
    p_features_7_expand3x3_weight,
    p_features_7_expand3x3_bias,
    p_features_8_squeeze_weight,
    p_features_8_squeeze_bias,
    p_features_8_expand1x1_weight,
    p_features_8_expand1x1_bias,
    p_features_8_expand3x3_weight,
    p_features_8_expand3x3_bias,
    p_features_9_squeeze_weight,
    p_features_9_squeeze_bias,
    p_features_9_expand1x1_weight,
    p_features_9_expand1x1_bias,
    p_features_9_expand3x3_weight,
    p_features_9_expand3x3_bias,
    p_features_10_squeeze_weight,
    p_features_10_squeeze_bias,
    p_features_10_expand1x1_weight,
    p_features_10_expand1x1_bias,
    p_features_10_expand3x3_weight,
    p_features_10_expand3x3_bias,
    p_features_12_squeeze_weight,
    p_features_12_squeeze_bias,
    p_features_12_expand1x1_weight,
    p_features_12_expand1x1_bias,
    p_features_12_expand3x3_weight,
    p_features_12_expand3x3_bias,
    p_classifier_1_weight,
    p_classifier_1_bias,
    x,
):
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (64, 512, 512, 3)

    # Prepare all weights to (OC, KH, KW, IC)
    w0 = prepare_weight(p_features_0_weight)  # (96, 7, 7, 3)

    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    out = conv2d_relu(x_nhwc, w0, p_features_0_bias, 2, 2, 0, 0, apply_relu=True)
    # out: (64, 253, 253, 96)

    # features[1]: MaxPool2d(3, stride=2, ceil_mode=True)
    out = maxpool2d_ceil(out, kernel_size=3, stride=2, padding=0)
    # out: (64, 126, 126, 96)

    # features[3]: FireModule(96, 16, 64, 64)
    out = fire_module(
        out,
        prepare_weight(p_features_3_squeeze_weight), p_features_3_squeeze_bias,
        prepare_weight(p_features_3_expand1x1_weight), p_features_3_expand1x1_bias,
        prepare_weight(p_features_3_expand3x3_weight), p_features_3_expand3x3_bias,
    )
    # out: (64, 126, 126, 128)

    # features[4]: FireModule(128, 16, 64, 64)
    out = fire_module(
        out,
        prepare_weight(p_features_4_squeeze_weight), p_features_4_squeeze_bias,
        prepare_weight(p_features_4_expand1x1_weight), p_features_4_expand1x1_bias,
        prepare_weight(p_features_4_expand3x3_weight), p_features_4_expand3x3_bias,
    )
    # out: (64, 126, 126, 128)

    # features[5]: FireModule(128, 32, 128, 128)
    out = fire_module(
        out,
        prepare_weight(p_features_5_squeeze_weight), p_features_5_squeeze_bias,
        prepare_weight(p_features_5_expand1x1_weight), p_features_5_expand1x1_bias,
        prepare_weight(p_features_5_expand3x3_weight), p_features_5_expand3x3_bias,
    )
    # out: (64, 126, 126, 256)

    # features[6]: MaxPool2d(3, stride=2, ceil_mode=True)
    out = maxpool2d_ceil(out, kernel_size=3, stride=2, padding=0)
    # out: (64, 63, 63, 256)

    # features[7]: FireModule(256, 32, 128, 128)
    out = fire_module(
        out,
        prepare_weight(p_features_7_squeeze_weight), p_features_7_squeeze_bias,
        prepare_weight(p_features_7_expand1x1_weight), p_features_7_expand1x1_bias,
        prepare_weight(p_features_7_expand3x3_weight), p_features_7_expand3x3_bias,
    )
    # out: (64, 63, 63, 256)

    # features[8]: FireModule(256, 48, 192, 192)
    out = fire_module(
        out,
        prepare_weight(p_features_8_squeeze_weight), p_features_8_squeeze_bias,
        prepare_weight(p_features_8_expand1x1_weight), p_features_8_expand1x1_bias,
        prepare_weight(p_features_8_expand3x3_weight), p_features_8_expand3x3_bias,
    )
    # out: (64, 63, 63, 384)

    # features[9]: FireModule(384, 48, 192, 192)
    out = fire_module(
        out,
        prepare_weight(p_features_9_squeeze_weight), p_features_9_squeeze_bias,
        prepare_weight(p_features_9_expand1x1_weight), p_features_9_expand1x1_bias,
        prepare_weight(p_features_9_expand3x3_weight), p_features_9_expand3x3_bias,
    )
    # out: (64, 63, 63, 384)

    # features[10]: FireModule(384, 64, 256, 256)
    out = fire_module(
        out,
        prepare_weight(p_features_10_squeeze_weight), p_features_10_squeeze_bias,
        prepare_weight(p_features_10_expand1x1_weight), p_features_10_expand1x1_bias,
        prepare_weight(p_features_10_expand3x3_weight), p_features_10_expand3x3_bias,
    )
    # out: (64, 63, 63, 512)

    # features[11]: MaxPool2d(3, stride=2, ceil_mode=True)
    out = maxpool2d_ceil(out, kernel_size=3, stride=2, padding=0)
    # out: (64, 31, 31, 512)

    # features[12]: FireModule(512, 64, 256, 256)
    out = fire_module(
        out,
        prepare_weight(p_features_12_squeeze_weight), p_features_12_squeeze_bias,
        prepare_weight(p_features_12_expand1x1_weight), p_features_12_expand1x1_bias,
        prepare_weight(p_features_12_expand3x3_weight), p_features_12_expand3x3_bias,
    )
    # out: (64, 31, 31, 512)

    # classifier[0]: Dropout(p=0.0) -> identity
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    w_cls = prepare_weight(p_classifier_1_weight)  # (1000, 1, 1, 512)
    out = conv2d_relu(out, w_cls, p_classifier_1_bias, 1, 1, 0, 0, apply_relu=True)
    # out: (64, 31, 31, 1000)

    # classifier[3]: AdaptiveAvgPool2d(1, 1)
    out = adaptive_avg_pool2d_1x1(out)
    # out: (64, 1, 1, 1000)

    # flatten -> (64, 1000)
    out = out.view(64, 1000)
    return out


def get_inputs():
    return [
        torch.randn(96, 3, 7, 7, device='cuda'),      # p_features_0_weight
        torch.randn(96, device='cuda'),                 # p_features_0_bias
        torch.randn(16, 96, 1, 1, device='cuda'),      # p_features_3_squeeze_weight
        torch.randn(16, device='cuda'),                 # p_features_3_squeeze_bias
        torch.randn(64, 16, 1, 1, device='cuda'),      # p_features_3_expand1x1_weight
        torch.randn(64, device='cuda'),                 # p_features_3_expand1x1_bias
        torch.randn(64, 16, 3, 3, device='cuda'),      # p_features_3_expand3x3_weight
        torch.randn(64, device='cuda'),                 # p_features_3_expand3x3_bias
        torch.randn(16, 128, 1, 1, device='cuda'),     # p_features_4_squeeze_weight
        torch.randn(16, device='cuda'),                 # p_features_4_squeeze_bias
        torch.randn(64, 16, 1, 1, device='cuda'),      # p_features_4_expand1x1_weight
        torch.randn(64, device='cuda'),                 # p_features_4_expand1x1_bias
        torch.randn(64, 16, 3, 3, device='cuda'),      # p_features_4_expand3x3_weight
        torch.randn(64, device='cuda'),                 # p_features_4_expand3x3_bias
        torch.randn(32, 128, 1, 1, device='cuda'),     # p_features_5_squeeze_weight
        torch.randn(32, device='cuda'),                 # p_features_5_squeeze_bias
        torch.randn(128, 32, 1, 1, device='cuda'),     # p_features_5_expand1x1_weight
        torch.randn(128, device='cuda'),                # p_features_5_expand1x1_bias
        torch.randn(128, 32, 3, 3, device='cuda'),     # p_features_5_expand3x3_weight
        torch.randn(128, device='cuda'),                # p_features_5_expand3x3_bias
        torch.randn(32, 256, 1, 1, device='cuda'),     # p_features_7_squeeze_weight
        torch.randn(32, device='cuda'),                 # p_features_7_squeeze_bias
        torch.randn(128, 32, 1, 1, device='cuda'),     # p_features_7_expand1x1_weight
        torch.randn(128, device='cuda'),                # p_features_7_expand1x1_bias
        torch.randn(128, 32, 3, 3, device='cuda'),     # p_features_7_expand3x