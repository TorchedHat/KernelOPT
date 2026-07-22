import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernels
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk
        offs_k += BLOCK_K

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def relu_kernel(
    X_ptr, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    x = tl.where(x > 0, x, tl.zeros_like(x))
    tl.store(X_ptr + offs, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def bias_relu_kernel(
    X_ptr, bias_ptr, N, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    c_idx = offs % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x = x + b
    x = tl.where(x > 0, x, tl.zeros_like(x))
    tl.store(X_ptr + offs, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def bias_add_kernel(
    X_ptr, bias_ptr, N, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    c_idx = offs % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x = x + b
    tl.store(X_ptr + offs, x, mask=mask)


# Max pool 2D kernel: kernel_size=3, stride=2, padding=1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def maxpool2d_k3s2p1_kernel(
    X_ptr, Y_ptr,
    N, C, H, W,
    OH, OW,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh * 2 - 1 + kh
            iw = ow * 2 - 1 + kw
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            idx = n * C * H * W + c * H * W + ih * W + iw
            val = tl.load(X_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    tl.store(Y_ptr + offs, max_val, mask=mask)


# Max pool 2D kernel: kernel_size=3, stride=1, padding=1 (used in inception branch_pool)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def maxpool2d_k3s1p1_kernel(
    X_ptr, Y_ptr,
    N, C, H, W,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    ow = offs % W
    tmp = offs // W
    oh = tmp % H
    tmp2 = tmp // H
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh - 1 + kh
            iw = ow - 1 + kw
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            idx = n * C * H * W + c * H * W + ih * W + iw
            val = tl.load(X_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    tl.store(Y_ptr + offs, max_val, mask=mask)


# Adaptive avg pool 2D: output 1x1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    X_ptr, Y_ptr,
    N, C, H, W,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    c = offs % C
    n = offs // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    HW = H * W
    for i in range(H):
        for j in range(W):
            idx = n * C * H * W + c * H * W + i * W + j
            val = tl.load(X_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    acc = acc / (H * W).to(tl.float32)
    tl.store(Y_ptr + offs, acc, mask=mask)


# Implicit GEMM convolution kernel for 1x1 convolutions
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bias_relu_kernel(
    X_ptr, W_ptr, bias_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    do_relu: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    X_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    W_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining
        x = tl.load(X_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        w = tl.load(W_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(x, w, acc=acc)
        X_ptrs += BLOCK_K * stride_xk
        W_ptrs += BLOCK_K * stride_wk
        offs_k += BLOCK_K

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    if do_relu:
        acc = tl.where(acc > 0, acc, tl.zeros_like(acc))

    Y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptrs, acc.to(tl.float32), mask=mask)


# General convolution kernel (for 3x3, 5x5, 7x7)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_general_bias_relu_kernel(
    X_ptr, W_ptr, bias_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    M, N, K,
    do_relu: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c = tmp_k // KERNEL_H

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_batch = batch[:, None] < BATCH

        x_mask = valid_m & valid_k & valid_ih & valid_iw & valid_batch

        x_idx = batch[:, None] * IN_C * IN_H * IN_W + in_c[None, :] * IN_H * IN_W + ih * IN_W + iw
        x = tl.load(X_ptr + x_idx, mask=x_mask, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> index [n, in_c, kh, kw]
        w_idx = offs_n[None, :] * IN_C * KERNEL_H * KERNEL_W + in_c[:, None] * KERNEL_H * KERNEL_W + kh[:, None] * KERNEL_W + kw[:, None]
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w = tl.load(W_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc=acc)

    bias = tl.load(bias_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    if do_relu:
        acc = tl.where(acc > 0, acc, tl.zeros_like(acc))

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    # Output in NCHW: [batch, out_c, out_h, out_w]
    out_idx = batch[:, None] * OUT_C * OUT_H * OUT_W + offs_n[None, :] * OUT_H * OUT_W + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(Y_ptr + out_idx, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Helper functions
# ============================================================

def triton_conv1x1(x, w, bias, do_relu=True):
    """x: [N, C, H, W], w: [OC, IC, 1, 1], bias: [OC]"""
    N, C, H, W = x.shape
    OC = w.shape[0]
    M = N * H * W
    K = C

    # Reshape x to [M, K] (NHWC-like but we use NCHW layout)
    # x is NCHW: [N, C, H, W], we need [N*H*W, C]
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
    x_2d = x_nhwc.reshape(M, K)

    w_2d = w.reshape(OC, K)  # [OC, K]

    y = torch.empty((M, OC), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OC, meta['BLOCK_N']))
    conv1x1_bias_relu_kernel[grid](
        x_2d, w_2d, bias, y,
        M, OC, K,
        do_relu,
    )

    # Reshape back to NCHW
    y_nhwc = y.reshape(N, H, W, OC)
    return y_nhwc.permute(0, 3, 1, 2).contiguous()


def triton_conv_general(x, w, bias, stride=1, padding=0, do_relu=True):
    """General convolution using Triton implicit GEMM"""
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = w.shape
    if isinstance(stride, int):
        SH, SW = stride, stride
    else:
        SH, SW = stride
    if isinstance(padding, int):
        PH, PW = padding, padding
    else:
        PH, PW = padding

    OH = (IH + 2 * PH - KH) // SH + 1
    OW = (IW + 2 * PW - KW) // SW + 1
    M = N * OH * OW
    K = IC * KH * KW

    y = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OC, meta['BLOCK_N']))
    conv_general_bias_relu_kernel[grid](
        x, w, bias, y,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        M, OC, K,
        do_relu,
    )
    return y


def triton_maxpool_k3s2p1(x):
    """MaxPool2d(kernel_size=3, stride=2, padding=1)"""
    N, C, H, W = x.shape
    OH = (H + 2 * 1 - 3) // 2 + 1
    OW = (W + 2 * 1 - 3) // 2 + 1
    N_total = N * C * OH * OW
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N_total, meta['BLOCK_SIZE']),)
    maxpool2d_k3s2p1_kernel[grid](
        x, y,
        N, C, H, W,
        OH, OW,
        N_total,
    )
    return y


def triton_maxpool_k3s1p1(x):
    """MaxPool2d(kernel_size=3, stride=1, padding=1)"""
    N, C, H, W = x.shape
    N_total = N * C * H * W
    y = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N_total, meta['BLOCK_SIZE']),)
    maxpool2d_k3s1p1_kernel[grid](
        x, y,
        N, C, H, W,
        N_total,
    )
    return y


def triton_adaptive_avg_pool_1x1(x):
    """AdaptiveAvgPool2d(1)"""
    N, C, H, W = x.shape
    N_total = N * C
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N_total, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x, y,
        N, C, H, W,
        N_total,
    )
    return y


def triton_linear(x, w, bias):
    """Linear layer: x @ w.T + bias"""
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_bias_kernel[grid](
        x, w, bias, y,
        M, N, K,
        K, 1,
        K, 1,
        N, 1,
    )
    return y


def inception_module(x, w1x1, b1x1,
                     w3x3_r, b3x3_r, w3x3, b3x3,
                     w5x5_r, b5x5_r, w5x5, b5x5,
    """Inception module forward pass"""
    # Branch 1x1
    branch1 = triton_conv1x1(x, w1x1, b1x1, do_relu=True)

    # Branch 3x3
    b3_reduce = triton_conv1x1(x, w3x3_r, b3x3_r, do_relu=True)
    branch3 = triton_conv_general(b3_reduce, w3x3, b3x3, stride=1, padding=1, do_relu=True)

    # Branch 5x5
    b5_reduce = triton_conv1x1(x, w5x5_r, b5x5_r, do_relu=True)
    branch5 = triton_conv_general(b5_reduce, w5x5, b5x5, stride=1, padding=2, do_relu=True)

    # Branch pool
    pooled = triton_maxpool_k3s1p1(x)
    branch_pool = triton_conv1x1(pooled, wp_r, bp_r, do_relu=True)

    return torch.cat([branch1, branch3, branch5, branch_pool], dim=1)


# ============================================================
# Main kernel function
# ============================================================

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