import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    HAS_BIAS: tl.constexpr,
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

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)

        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m < M
        valid_k = offs_k < K
        valid = (valid_m[:, None] & valid_k[None, :] &
                 (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W))

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * stride_xn +
                  in_c[None, :] * stride_xc +
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        valid_n = offs_n < OUT_C
        w_ptrs = (W_ptr +
                  offs_n[None, :] * stride_wco +
                  in_c[:, None] * stride_wci +
                  kh[:, None] * stride_wkh +
                  kw[:, None] * stride_wkw)
        w_mask = valid_k[:, None] & valid_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    valid_m = offs_m < M
    valid_n = offs_n < OUT_C
    out_mask = valid_m[:, None] & valid_n[None, :]

    y_ptrs = (Y_ptr +
              batch_idx[:, None] * stride_yn +
              offs_n[None, :] * stride_yc +
              out_h_idx[:, None] * stride_yh +
              out_w_idx[:, None] * stride_yw)
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def conv2d_triton(x, weight, bias, stride=(1,1), padding=(0,0), apply_relu=False):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1

    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x, weight, bias if bias is not None else x,
        y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        M, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        HAS_BIAS=(bias is not None),
        APPLY_RELU=apply_relu,
    )
    return y


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def max_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # Decode: offs -> (n, c, oh, ow)
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    result = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            result = tl.maximum(result, val)

    tl.store(Y_ptr + offs, result, mask=mask)


def max_pool2d_triton(x, kernel_size=3, stride=2, padding=1):
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1
    numel = N * C * OUT_H * OUT_W
    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        numel,
    )
    return y


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    # For adaptive avg pool to (1,1), each output covers the full input
    # h_start = oh * IN_H // OUT_H, h_end = (oh+1)*IN_H//OUT_H
    h_start = oh * IN_H // OUT_H
    h_end = (oh + 1) * IN_H // OUT_H
    w_start = ow * IN_W // OUT_W
    w_end = (ow + 1) * IN_W // OUT_W

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for ih in range(IN_H):
        for iw in range(IN_W):
            valid = mask & (ih >= h_start) & (ih < h_end) & (iw >= w_start) & (iw < w_end)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(X_ptr + x_idx, mask=valid, other=0.0)
            acc = acc + tl.where(valid, val, tl.zeros((BLOCK_SIZE,), dtype=tl.float32))
            count = count + tl.where(valid, tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32), tl.zeros((BLOCK_SIZE,), dtype=tl.float32))

    result = acc / tl.maximum(count, tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32))
    tl.store(Y_ptr + offs, result, mask=mask)


def adaptive_avg_pool2d_triton(x, output_size=(1, 1)):
    N, C, IN_H, IN_W = x.shape
    OUT_H, OUT_W = output_size
    numel = N * C * OUT_H * OUT_W
    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        numel,
    )
    return y


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
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
        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = W_ptr + offs_n[None, :] * K + offs_k[:, None]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc, mask=out_mask)


def linear_triton(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, weight, bias, y, M, N, K)
    return y


def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    # Branch 1x1
    b1 = conv2d_triton(x, w_1x1, b_1x1, (1, 1), (0, 0), apply_relu=True)
    # Branch 3x3
    b2 = conv2d_triton(x, w_3x3_r, b_3x3_r, (1, 1), (0, 0), apply_relu=True)
    b2 = conv2d_triton(b2, w_3x3, b_3x3, (1, 1), (1, 1), apply_relu=True)
    # Branch 5x5
    b3 = conv2d_triton(x, w_5x5_r, b_5x5_r, (1, 1), (0, 0), apply_relu=True)
    b3 = conv2d_triton(b3, w_5x5, b_5x5, (1, 1), (2, 2), apply_relu=True)
    # Branch pool
    bp = max_pool2d_triton(x, kernel_size=3, stride=1, padding=1)
    bp = conv2d_triton(bp, w_pool, b_pool, (1, 1), (0, 0), apply_relu=True)
    return torch.cat([b1, b2, b3, bp], dim=1)


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_conv3_weight, p_conv3_bias,
    p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
    p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
    p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
    p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
    p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
    p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
    p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
    p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
    p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
    p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
    p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
    p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
    p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
    p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
    p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
    p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
    p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
    p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
    p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
    p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
    p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
    p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
    p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
    p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
    p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
    p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias,
    p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
    p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
    p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
    p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
    p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
    p_inception4d_branch_pool_1_weight, p_inception4d_branch_pool_1_bias,
    p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
    p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
    p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
    p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
    p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
    p_inception4e_branch_pool_1_weight, p_inception4e_branch_pool_1_bias,
    p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
    p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
    p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
    p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
    p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
    p_inception5a_branch_pool_1_weight, p_inception5a_branch_pool_1_bias,
    p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
    p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
    p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
    p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
    p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
    p_inception5b_branch_pool_1_weight, p_inception5b_branch_pool_1_bias,
    p_fc_weight, p_fc_bias,
    x,
):
    # conv1 + relu + maxpool
    x = conv2d_triton(x, p_conv1_weight, p_conv1_bias, (2, 2), (3, 3), apply_relu=True)
    x = max_pool2d_triton(x, kernel_size=3, stride=2, padding=1)
    # conv2 + relu
    x = conv2d_triton(x, p_conv2_weight, p_conv2_bias, (1, 1), (0, 0), apply_relu=True)
    # conv3 + relu + maxpool
    x = conv2d_triton(x, p_conv3_weight, p_conv3_bias, (1, 1), (1, 1), apply_relu=True)
    x = max_pool2d_triton(x, kernel_size=3, stride=2, padding=1)

    # inception3a
    x = inception_module(x,
                         p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
                         p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
                         p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
                         p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
                         p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,

    # inception3b
    x = inception_module(x,
                         p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
                         p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
                         p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
                         p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
                         p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,

    x = max_pool2d_triton(x, kernel_size=3, stride=2, padding=1)

    # inception4a
    x = inception_module(x,
                         p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
                         p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
                         p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
                         p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
                         p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,

    # inception4b
    x = inception_module(x,
                         p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
                         p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
                         p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
                         p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
                         p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,

    # inception4c
    x = inception_module(x,
                         p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
                         p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
                         p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
                         p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
                         p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,