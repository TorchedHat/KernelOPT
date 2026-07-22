import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernels
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def bias_relu_kernel(x_ptr, bias_ptr, out_ptr, n_elements, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c = offs % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    out = tl.where(out > 0, out, tl.zeros_like(out))
    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def bias_kernel(x_ptr, bias_ptr, out_ptr, n_elements, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c = offs % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.where(x > 0, x, tl.zeros_like(x))
    tl.store(out_ptr + offs, out, mask=mask)


# Max pool 2D kernel: input NCHW, output NCHW
# kernel_size=3, stride=2, padding=1 (most common case)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def maxpool2d_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    KH, KW, SH, SW, PH, PW,
):
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    tl.store(out_ptr + offs, max_val, mask=mask)


# Adaptive avg pool 2D: output size 1x1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'IH', 'IW'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW,
):
    pid = tl.program_id(0)
    total = N * C
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    c = offs % C
    n = offs // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = IH * IW

    for h in range(IH):
        for w in range(IW):
            idx = n * (C * IH * IW) + c * (IH * IW) + h * IW + w
            val = tl.load(inp_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    acc = acc / count
    tl.store(out_ptr + offs, acc, mask=mask)


# Linear kernel: y = x @ W^T + b
# x: (M, K), W: (N, K), b: (N,), y: (M, N)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=w_mask, other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# Conv2d implicit GEMM kernel
# Input: NCHW, Weight: (OC, IC, KH, KW), Output: NCHW
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M_total', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    SH, SW, PH, PW,
    M_total, TOTAL_K,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M: n, oh, ow
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    n = tmp // OUT_H

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K: ic, kh, kw
        ic = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KW
        kh = tmp_k // KW

        ih = oh[:, None] * SH + kh[None, :] - PH
        iw = ow[:, None] * SW + kw[None, :] - PW

        valid_m = offs_m[:, None] < M_total
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        x_mask = valid_m & valid_k & valid_ih & valid_iw

        x_idx = n[:, None] * (IN_C * IN_H * IN_W) + ic[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=x_mask, other=0.0)

        # Weight: (OUT_C, IN_C, KH, KW) -> index: oc * (IN_C*KH*KW) + ic * (KH*KW) + kh * KW + kw
        w_idx = offs_n[None, :] * TOTAL_K + offs_k[:, None]
        w_mask = (offs_n[None, :] < OUT_C) & (offs_k[:, None] < TOTAL_K)
        w_tile = tl.load(W_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    out_mask = (offs_m[:, None] < M_total) & (offs_n[None, :] < OUT_C)
    out_idx = n[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + oh[:, None] * OUT_W + ow[:, None]
    tl.store(Out_ptr + out_idx, acc, mask=out_mask)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0)):
    """Perform conv2d using Triton implicit GEMM kernel, then add bias+relu if needed."""
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1

    M_total = N * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW

    # Reshape weight to (OUT_C, TOTAL_K)
    w_flat = weight.reshape(OUT_C, TOTAL_K).contiguous()

    out = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M_total, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x, w_flat, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW, PH, PW,
        M_total, TOTAL_K,
    )

    if bias is not None:
        n_elements = out.numel()
        grid2 = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        bias_kernel[grid2](out, bias, out, n_elements, OUT_C)

    return out


def triton_conv2d_relu(x, weight, bias, stride=(1,1), padding=(0,0)):
    """Perform conv2d + bias + relu using Triton."""
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1

    M_total = N * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW

    w_flat = weight.reshape(OUT_C, TOTAL_K).contiguous()
    out = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M_total, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x, w_flat, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW, PH, PW,
        M_total, TOTAL_K,
    )

    if bias is not None:
        n_elements = out.numel()
        grid2 = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        bias_relu_kernel[grid2](out, bias, out, n_elements, OUT_C)
    else:
        n_elements = out.numel()
        grid2 = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        relu_kernel[grid2](out, out, n_elements)

    return out


def triton_maxpool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, IH, IW = x.shape
    OH = (IH + 2 * padding - kernel_size) // stride + 1
    OW = (IW + 2 * padding - kernel_size) // stride + 1

    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, out,
        N, C, IH, IW, OH, OW,
        kernel_size, kernel_size, stride, stride, padding, padding,
    )
    return out


def triton_adaptive_avg_pool2d_1x1(x):
    N, C, IH, IW = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    total = N * C

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x, out,
        N, C, IH, IW,
    )
    return out


def triton_linear(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_kernel[grid](x, weight, bias, out, M, N, K)
    return out


def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    """Inception module: 4 branches concatenated."""
    # Branch 1: 1x1 conv + relu
    branch1 = triton_conv2d_relu(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0))

    # Branch 2: 1x1 reduce + 3x3 conv + relu
    b2 = triton_conv2d_relu(x, w_3x3_r, b_3x3_r, stride=(1,1), padding=(0,0))
    branch2 = triton_conv2d_relu(b2, w_3x3, b_3x3, stride=(1,1), padding=(1,1))

    # Branch 3: 1x1 reduce + 5x5 conv + relu
    b3 = triton_conv2d_relu(x, w_5x5_r, b_5x5_r, stride=(1,1), padding=(0,0))
    branch3 = triton_conv2d_relu(b3, w_5x5, b_5x5, stride=(1,1), padding=(2,2))

    # Branch 4: maxpool + 1x1 conv + relu
    b4 = triton_maxpool2d(x, kernel_size=3, stride=1, padding=1)
    branch4 = triton_conv2d_relu(b4, w_pool, b_pool, stride=(1,1), padding=(0,0))

    return torch.cat([branch1, branch2, branch3, branch4], dim=1)


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
):
    # Make inputs contiguous
    x = x.contiguous()

    # conv1: 3->64, 7x7, stride=2, pad=3
    x = triton_conv2d_relu(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3))
    # maxpool1: 3x3, stride=2, pad=1
    x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=1)
    # conv2: 64->64, 1x1
    x = triton_conv2d_relu(x, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0))
    # conv3: 64->192, 3x3, pad=1
    x = triton_conv2d_relu(x, p_conv3_weight, p_conv