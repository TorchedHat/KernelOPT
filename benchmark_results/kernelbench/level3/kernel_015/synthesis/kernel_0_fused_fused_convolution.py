import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math

# ─────────────────────────────────────────────────────────────────────────────
# Fused BatchNorm (inference) + ReLU kernel
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
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
    var  = tl.load(var_ptr  + c_idx, mask=mask, other=1.0)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0)
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](x, out, mean, var, weight, bias, numel, C, HW, eps)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Fused BatchNorm (inference, no ReLU) kernel
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
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
    var  = tl.load(var_ptr  + c_idx, mask=mask, other=1.0)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0)
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    tl.store(out_ptr + offsets, out, mask=mask)


def bn(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](x, out, mean, var, weight, bias, numel, C, HW, eps)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Implicit GEMM Conv2d kernel (for 3x3 and 1x1 convolutions)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N_out, K,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (OUT_C, IN_C, KH, KW) -> stored as (OUT_C, KH, KW, IN_C) for NHWC
    s_wo, s_wkh, s_wkw, s_wc,
    # strides for Y (NHWC layout)
    s_yn, s_yh, s_yw, s_yc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KW
        kh = tmp_k // KW

        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = (offs_m < M)
        valid_k = (offs_k < K)
        valid = (valid_m[:, None] & valid_k[None, :] &
                 (ih >= 0) & (ih < IN_H) &
                 (iw >= 0) & (iw < IN_W))

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * s_xn +
                  in_c[None, :] * s_xc)
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # W layout: (OUT_C, KH, KW, IN_C) for NHWC
        valid_w = (valid_k[:, None] & (offs_n[None, :] < OUT_C))
        w_ptrs = (W_ptr +
                  kh[:, None] * s_wkh +
                  kw[:, None] * s_wkw +
                  in_c[:, None] * s_wc +
                  offs_n[None, :] * s_wo)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output in NHWC layout
    valid_out = ((offs_m < M)[:, None] & (offs_n < OUT_C)[None, :])
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * s_yn +
              out_h_idx[:, None] * s_yh +
              out_w_idx[:, None] * s_yw +
              offs_n[None, :] * s_yc)
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def conv2d_triton(x, weight, stride=1, padding=0):
    """
    returns: (N, OUT_C, OUT_H, OUT_W)
    """
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    if isinstance(stride, int):
        STRIDE_H = STRIDE_W = stride
    else:
        STRIDE_H, STRIDE_W = stride
    if isinstance(padding, int):
        PAD_H = PAD_W = padding
    else:
        PAD_H, PAD_W = padding

    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1

    # Convert to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # Convert weight to (OUT_C, KH, KW, IN_C) for NHWC access
    w_nhwc = weight.permute(0, 2, 3, 1).contiguous()

    y_nhwc = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x.device, dtype=x.dtype)

    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW

    s_xn = x_nhwc.stride(0)
    s_xh = x_nhwc.stride(1)
    s_xw = x_nhwc.stride(2)
    s_xc = x_nhwc.stride(3)

    # w_nhwc: (OUT_C, KH, KW, IN_C)
    s_wo  = w_nhwc.stride(0)
    s_wkh = w_nhwc.stride(1)
    s_wkw = w_nhwc.stride(2)
    s_wc  = w_nhwc.stride(3)

    s_yn = y_nhwc.stride(0)
    s_yh = y_nhwc.stride(1)
    s_yw = y_nhwc.stride(2)
    s_yc = y_nhwc.stride(3)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_nhwc, y_nhwc,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M, OUT_C, K,
        s_xn, s_xh, s_xw, s_xc,
        s_wo, s_wkh, s_wkw, s_wc,
        s_yn, s_yh, s_yw, s_yc,
    )

    return y_nhwc.permute(0, 3, 1, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Max Pool 2d (3x3, stride=2, padding=1) using Triton
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
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

    # Decode: output is (N, C, OUT_H, OUT_W)
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
            idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(val > max_val, val, max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


def maxpool2d_triton(x, kernel_size=3, stride=2, padding=1):
    N, C, IN_H, IN_W = x.shape
    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size
    if isinstance(stride, int):
        STRIDE_H = STRIDE_W = stride
    else:
        STRIDE_H, STRIDE_W = stride
    if isinstance(padding, int):
        PAD_H = PAD_W = padding
    else:
        PAD_H, PAD_W = padding

    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = N * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
        numel,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Avg Pool 2d (2x2, stride=2) using Triton
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
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
            idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + idx, mask=valid, other=0.0)
            acc = acc + val

    tl.store(out_ptr + offsets, acc / count, mask=mask)


def avgpool2d_triton(x, kernel_size=2, stride=2):
    N, C, IN_H, IN_W = x.shape
    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size
    if isinstance(stride, int):
        STRIDE_H = STRIDE_W = stride
    else:
        STRIDE_H, STRIDE_W = stride

    OUT_H = (IN_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W - KW) // STRIDE_W + 1
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = N * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    avgpool2d_kernel[grid](
        x, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, STRIDE_H, STRIDE_W,
        numel,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Avg Pool 2d (output 1x1) using Triton
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 32},  num_warps=2),
    ],
    key=['NC'],
)
@triton.jit
def adaptive_avgpool2d_1x1_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    NC,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < NC

    n = offsets // C
    c = offsets % C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    base = n * C * HW + c * HW
    for i in range(HW):
        val = tl.load(x_ptr + base + i, mask=mask, other=0.0)
        acc = acc + val

    tl.store(out_ptr + offsets, acc / HW, mask=mask)


def adaptive_avgpool2d_triton(x):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_SIZE']),)
    adaptive_avgpool2d_1x1_kernel[grid](
        x, out,
        N, C, HW, NC,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Linear kernel using Triton (GEMM: (M, K) x (K, N) -> (M, N))
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    b = tl.load(b_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def linear_triton(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, weight, bias, out, M, N, K)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# DenseNet forward pass
# ─────────────────────────────────────────────────────────────────────────────
def dense_layer_forward(x, bn_w, bn_b, bn_mean, bn_var, conv_w, dropout_p=0.0):
    """BN -> ReLU -> Conv2d -> Dropout (dropout_p=0 in inference)"""
    x = bn_relu(x, bn_mean, bn_var, bn_w, bn_b)
    x = conv2d_triton(x, conv_w, stride=1, padding=1)
    # dropout with p=0 is identity
    return x


def transition_forward(x, bn_w, bn_b, bn_mean, bn_var, conv_w):
    """BN -> ReLU -> Conv2d(1x1) -> AvgPool2d(2x2)"""
    x = bn_relu(x, bn_mean, bn_var, bn_w, bn_b)
    x = conv2d_triton(x, conv_w, stride=1, padding=0)
    x = avgpool2d_triton(x, kernel_