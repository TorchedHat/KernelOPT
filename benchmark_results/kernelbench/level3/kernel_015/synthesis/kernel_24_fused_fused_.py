import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math
import argparse

# ============================================================
# Fused BatchNorm (inference) + ReLU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channel index: layout is (N, C, H, W) contiguous => channel = (offset // HW) % C
    c_idx = (offsets // HW) % C
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps,
    )
    return out


# ============================================================
# Fused BatchNorm (inference, no ReLU) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // HW) % C
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps,
    )
    return out


# ============================================================
# Implicit GEMM Conv2d kernel (general)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, K,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (OUT_C, IN_C, KH, KW) -> stored as (OUT_C, KH, KW, IN_C) for NHWC
    s_wo, s_wkh, s_wkw, s_wc,
    # strides for Y (NHWC layout)
    s_yn, s_yh, s_yw, s_yc,
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

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KW
        kh = tmp_k // KW

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m < M
        valid_k = offs_k < K
        valid = (
            valid_m[:, None] &
            valid_k[None, :] &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W)
        )

        x_ptrs = (
            batch_idx[:, None] * s_xn +
            in_c[None, :] * s_xc
        )
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Weight: (OUT_C, KH, KW, IN_C) layout
        w_ptrs = (
            offs_n[None, :] * s_wo +
            kh[:, None] * s_wkh +
            kw[:, None] * s_wkw +
            in_c[:, None] * s_wc
        )
        valid_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output in NHWC layout
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    y_ptrs = (
        batch_idx[:, None] * s_yn +
        out_h_idx[:, None] * s_yh +
        out_w_idx[:, None] * s_yw +
        offs_n[None, :] * s_yc
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def triton_conv2d(x_nhwc, w, OUT_H, OUT_W, stride_h, stride_w, pad_h, pad_w):
    """
    returns: (N, OUT_H, OUT_W, OUT_C) contiguous
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w.shape

    # Rearrange weight to (OUT_C, KH, KW, IN_C)
    w_nhwc = w.permute(0, 2, 3, 1).contiguous()

    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW

    y = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=x_nhwc.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_kernel[grid](
        x_nhwc, w_nhwc, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        M, K,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w_nhwc.stride(0), w_nhwc.stride(1), w_nhwc.stride(2), w_nhwc.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )
    return y


def conv2d_nchw(x, w, stride=1, padding=0):
    """Wrapper: x in NCHW, w in (OC, IC, KH, KW), returns NCHW"""
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    N, C, H, W = x.shape
    OC, IC, KH, KW = w.shape
    OUT_H = (H + 2 * padding[0] - KH) // stride[0] + 1
    OUT_W = (W + 2 * padding[1] - KW) // stride[1] + 1
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    y_nhwc = triton_conv2d(x_nhwc, w, OUT_H, OUT_W, stride[0], stride[1], padding[0], padding[1])
    return y_nhwc.permute(0, 3, 1, 2).contiguous()


# ============================================================
# Max Pool 2d kernel (3x3, stride=2, padding=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def maxpool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Decode: (N, C, OUT_H, OUT_W)
    ow = offsets % OUT_W
    tmp = offsets // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_maxpool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    OUT_H = (H + 2 * padding - kernel_size) // stride + 1
    OUT_W = (W + 2 * padding - kernel_size) // stride + 1
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = out.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, out,
        N, C, H, W, OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride, padding, padding,
        numel,
    )
    return out


# ============================================================
# Avg Pool 2d kernel (2x2, stride=2)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def avgpool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    ow = offsets % OUT_W
    tmp = offsets // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = KH * KW

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh
            iw = ow * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            acc = acc + val

    tl.store(out_ptr + offsets, acc / count, mask=mask)


def triton_avgpool2d(x, kernel_size=2, stride=2):
    N, C, H, W = x.shape
    OUT_H = (H - kernel_size) // stride + 1
    OUT_W = (W - kernel_size) // stride + 1
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = out.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    avgpool2d_kernel[grid](
        x, out,
        N, C, H, W, OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride,
        numel,
    )
    return out


# ============================================================
# Adaptive Avg Pool 2d -> (1,1) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
    ],
    key=['NC'],
)
@triton.jit
def adaptive_avgpool_1x1_kernel(
    x_ptr, out_ptr,
    NC, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    nc_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = nc_offsets < NC

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(HW):
        val = tl.load(x_ptr + nc_offsets * HW + i, mask=mask, other=0.0)
        acc = acc + val
    acc = acc / HW
    tl.store(out_ptr + nc_offsets, acc, mask=mask)


def triton_adaptive_avgpool_1x1(x):
    N, C, H, W = x.shape
    NC = N * C
    HW = H * W
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_SIZE']),)
    adaptive_avgpool_1x1_kernel[grid](
        x.contiguous(), out,
        NC, HW,
    )
    return out


# ============================================================
# Linear kernel (using tl.dot)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def triton_linear(x, w, b):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, w, b, out, M, N, K)
    return out


# ============================================================
# Dense layer: BN + ReLU + Conv2d (3x3, pad=1) + Dropout(0)
# ============================================================
def dense_layer(x, bn_w, bn_b, bn_mean, bn_var, conv_w):
    """One dense layer: BN+ReLU -> Conv2d(3x3,pad=1) -> (dropout=0, identity)"""
    x = fused_bn_relu(x, bn_w, bn_b, bn_mean, bn_var)
    x = conv2d_nchw(x, conv_w, stride=1, padding=1)
    # dropout with p=0 is identity
    return x


# ============================================================
# Main kernel_function implementing DenseNet forward pass
# ============================================================
def kernel_function(
    p_features_0_weight,
    p_features_1_weight,
    p_features_1_bias,
    p_dense_blocks_0_layers_0_0_weight,
    p_dense_blocks_0_layers_0_0_bias,
    p_dense_blocks_0_layers_0_2_weight,
    p_dense_blocks_0_layers_1_0_weight,
    p_dense_blocks_0_layers_1_0_bias,
    p_dense_blocks_0_layers_1_2_weight,
    p_dense_blocks_0_layers_2_0_weight,
    p_dense_blocks_0_layers_2_0_bias,
    p_dense_blocks_0_layers_2_2_weight,
    p_dense_blocks_0_layers_3_0_weight,
    p_dense_blocks_0_layers_3_0_bias,
    p_dense_blocks_0_layers_3_2_weight,
    p_dense_blocks_0_layers_4_0_weight,
    p_dense_blocks_0_layers_4_0_bias,
    p_dense_blocks_0_layers_4_2_weight,
    p_dense_blocks_0_layers_5_0_weight,
    p_dense_blocks_0_layers_5_0_bias,
    p_dense_blocks_0_layers_5_2_weight,
    p_transition_layers_0_transition_0_weight,
    p_transition_layers_0_transition_0_bias,
    p_transition_layers_0_transition_2_weight,
    p_dense_blocks_1_layers_0_0_weight,
    p_dense_blocks_1_layers_0_0_bias,
    p_dense_blocks_1_layers_0_2_weight,
    p_dense_blocks_1_layers_1_0_weight,
    p_dense_blocks_1_layers_1_0_bias,
    p_dense_blocks_1_layers_1_2_weight,
    p_dense_blocks_1_layers_2_0_weight,
    p_dense_blocks_1_layers_2_0_bias,
    p_dense_blocks_1_layers_2_2_weight,
    p_dense_blocks_1_layers_3_0_weight,
    p_dense_blocks_1_layers_3_0_bias,
    p_dense_blocks_1_layers_3_2_weight,
    p_dense_blocks_1_layers_4_0_weight,
    p_dense_blocks_1_layers_4_0_bias,
    p_dense_blocks_1_layers_4_2_weight,
    p_dense_blocks_1_layers_5_0_weight,
    p_dense_blocks_1_layers_5_0_bias,
    p_dense_blocks_1_layers_5_2_weight,
    p_dense_blocks_1_layers_6_0_weight,
    p_dense_blocks_1_layers_6_0_bias,
    p_dense_blocks_1_layers_6_2_weight,
    p_dense_blocks_1_layers_7_0_weight,
    p_dense_blocks_1_layers_7_0_bias,
    p_dense_blocks_1_layers_7_2_weight,
    p_dense_blocks_1_layers_8_0_weight,
    p_dense_blocks_1_layers_8_0_bias,
    p_dense_blocks_1_layers_8_2_weight,
    p_dense_blocks_1_layers_9_0_weight,
    p_dense_blocks_1_layers_9_0_bias,