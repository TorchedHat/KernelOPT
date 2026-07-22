import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    C,
    spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // spatial) % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    out = x + bias
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    C,
    spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // spatial) % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    out = x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    HAS_BIAS: tl.constexpr,
    RELU: tl.constexpr,
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
        # Decode K -> (in_c, kh, kw)
        in_c_idx = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KW
        kh_idx = offs_k // (IN_C * KW)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_h & valid_w

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> index [offs_n, in_c, kh, kw]
        valid_n = offs_n[None, :] < N
        mask_w = valid_k[:, None] & valid_n
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias and apply ReLU
    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    if RELU:
        acc = tl.where(acc > 0.0, acc, 0.0)

    # Store output: layout [BATCH, OUT_C, OUT_H, OUT_W]
    out_stride_n = OUT_C * OUT_H * OUT_W
    out_stride_c = OUT_H * OUT_W
    out_stride_h = OUT_W
    out_stride_w = 1

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = (Out_ptr
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_conv2d(x, weight, bias=None, stride=(1,1), padding=(0,0), relu=False):
    """Triton implicit GEMM convolution."""
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1

    M = BATCH * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KH * KW

    out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    # Make contiguous
    x_c = x.contiguous()
    w_c = weight.contiguous()

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    has_bias = bias is not None
    b_ptr = bias if has_bias else x  # dummy pointer

    conv_implicit_gemm_kernel[grid](
        x_c, w_c, b_ptr if has_bias else x_c, out,
        M, N, K,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        x_c.stride(0), x_c.stride(1), x_c.stride(2), x_c.stride(3),
        w_c.stride(0), w_c.stride(1), w_c.stride(2), w_c.stride(3),
        HAS_BIAS=has_bias,
        RELU=relu,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    numel,
    BATCH, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Decode flat index -> (n, c, oh, ow)
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
            in_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    BATCH, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1
    out = torch.empty((BATCH, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = BATCH * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out, numel,
        BATCH, C, IN_H, IN_W,
        OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    numel,
    BATCH, C, IN_H, IN_W,
    OUT_H, OUT_W,
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

    # Compute input window
    h_start = oh * IN_H // OUT_H
    h_end = (oh + 1) * IN_H // OUT_H
    w_start = ow * IN_W // OUT_W
    w_end = (ow + 1) * IN_W // OUT_W

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = (h_end - h_start) * (w_end - w_start)

    for ih in range(IN_H):
        for iw in range(IN_W):
            in_window = (ih >= h_start) & (ih < h_end) & (iw >= w_start) & (iw < w_end)
            valid = mask & in_window
            in_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + in_idx, mask=valid, other=0.0)
            acc = tl.where(valid, acc + val, acc)

    out_val = acc / count.to(tl.float32)
    tl.store(out_ptr + offsets, out_val, mask=mask)


def triton_adaptive_avg_pool2d(x, output_size=(1, 1)):
    BATCH, C, IN_H, IN_W = x.shape
    OUT_H, OUT_W = output_size
    out = torch.empty((BATCH, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = BATCH * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, out, numel,
        BATCH, C, IN_H, IN_W,
        OUT_H, OUT_W,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    HAS_BIAS: tl.constexpr,
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
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_k[:, None] * N + offs_n[None, :], mask=mask_w, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc=acc)

    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=mask_out)


def triton_linear(x, weight, bias=None):
    """x: [M, K], weight: [N, K] -> out: [M, N]"""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # weight is [N, K], we need [K, N] for matmul
    w_t = weight.t().contiguous()
    has_bias = bias is not None
    b_ptr = bias if has_bias else x

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, w_t, b_ptr if has_bias else x, out,
        M, N, K,
        HAS_BIAS=has_bias,
    )
    return out


def inception_module(x,
                     w_1x1, b_1x1,
                     w_r3, b_r3, w_3x3, b_3x3,
                     w_r5, b_r5, w_5x5, b_5x5,
    # Branch 1x1
    b1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), relu=True)

    # Branch 3x3
    r3 = triton_conv2d(x, w_r3, b_r3, stride=(1,1), padding=(0,0), relu=True)
    b3 = triton_conv2d(r3, w_3x3, b_3x3, stride=(1,1), padding=(1,1), relu=True)

    # Branch 5x5
    r5 = triton_conv2d(x, w_r5, b_r5, stride=(1,1), padding=(0,0), relu=True)
    b5 = triton_conv2d(r5, w_5x5, b_5x5, stride=(1,1), padding=(2,2), relu=True)

    # Branch pool
    pooled = triton_max_pool2d(x, kernel_size=3, stride=1, padding=1)
    bp = triton_conv2d(pooled, w_pool, b_pool, stride=(1,1), padding=(0,0), relu=True)

    return torch.cat([b1, b3, b5, bp], dim=1)


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
    # Initial layers
    x = triton_conv2d(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), relu=True)
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)
    x = triton_conv2d(x, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), relu=True)
    x = triton_conv2d(x, p_conv3_weight, p_conv3_bias, stride=(1,1), padding=(1,1), relu=True)
    x = triton_max