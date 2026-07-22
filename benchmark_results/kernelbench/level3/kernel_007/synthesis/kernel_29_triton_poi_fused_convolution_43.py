import torch
import triton
import triton.language as tl

# [Full implementation follows]


import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused bias + ReLU kernel (pointwise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def fused_bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + bias
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def fused_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Max Pool 2D kernel (3x3, stride=2, padding=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'H_in', 'W_in', 'H_out', 'W_out'],
)
@triton.jit
def max_pool2d_kernel_s2(
    inp_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles one output element
    pid = tl.program_id(0)
    total = N * C * H_out * W_out
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    w_out = offsets % W_out
    tmp = offsets // W_out
    h_out = tmp % H_out
    tmp2 = tmp // H_out
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            h_in = h_out * 2 + kh - 1
            w_in = w_out * 2 + kw - 1
            valid = mask & (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
            idx = n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid & (val > max_val), val, max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# Max Pool 2D kernel (3x3, stride=1, padding=1) - for inception branch_pool
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'H_in', 'W_in', 'H_out', 'W_out'],
)
@triton.jit
def max_pool2d_kernel_s1(
    inp_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * H_out * W_out
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    w_out = offsets % W_out
    tmp = offsets // W_out
    h_out = tmp % H_out
    tmp2 = tmp // H_out
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            h_in = h_out + kh - 1
            w_in = w_out + kw - 1
            valid = mask & (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
            idx = n * C * H_in * W_in + c * H_in * W_in + h_in * W_in + w_in
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid & (val > max_val), val, max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# ============================================================
# Adaptive Average Pool 2D (global average pool to 1x1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    inp_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    c = offsets % C
    n = offsets // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    hw = H * W
    for h in range(7):  # H=7 for final layer
        for w in range(7):  # W=7 for final layer
            idx = n * C * H * W + c * H * W + h * W + w
            val = tl.load(inp_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    acc = acc / (hw.to(tl.float32))
    tl.store(out_ptr + offsets, acc, mask=mask)


# ============================================================
# Linear (matmul) kernel using tl.dot
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :], mask=w_mask, other=0.0)
        acc = tl.dot(x, w, acc)

    bias_mask = offs_n < N
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ============================================================
# Implicit GEMM Conv2d kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KH', 'KW'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = BATCH * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode output positions
    w_out = offs_m % OUT_W
    tmp = offs_m // OUT_W
    h_out = tmp % OUT_H
    n_idx = tmp // OUT_H

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        # Decode k into (in_c, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        ic_idx = tmp_k // KH

        # Input coordinates
        h_in = h_out[:, None] * stride_h + kh_idx[None, :] - pad_h
        w_in = w_out[:, None] * stride_w + kw_idx[None, :] - pad_w

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_h = (h_in >= 0) & (h_in < IN_H)
        valid_w = (w_in >= 0) & (w_in < IN_W)
        x_mask = valid_m & valid_k & valid_h & valid_w

        x_idx = n_idx[:, None] * IN_C * IN_H * IN_W + ic_idx[None, :] * IN_H * IN_W + h_in * IN_W + w_in
        x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> index [n, ic, kh, kw]
        w_idx = offs_n[None, :] * IN_C * KH * KW + ic_idx[:, None] * KH * KW + kh_idx[:, None] * KW + kw_idx[:, None]
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc)

    if has_bias:
        bias_mask = offs_n < N
        bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
        acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Store in NCHW format: [n, c, h, w]
    out_idx = n_idx[:, None] * OUT_C * OUT_H * OUT_W + offs_n[None, :] * OUT_H * OUT_W + h_out[:, None] * OUT_W + w_out[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


# ============================================================
# ReLU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_conv2d(x, w, bias, stride_h, stride_w, pad_h, pad_w):
    N, C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = w.shape
    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)
    M = N * H_out * W_out
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )
    has_bias = bias is not None
    bias_ptr = bias if has_bias else x  # dummy
    conv2d_kernel[grid](
        x, w, bias_ptr, out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        has_bias,
    )
    return out


def triton_relu(x):
    out = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, n)
    return out


def triton_max_pool2d_s2(x):
    N, C, H_in, W_in = x.shape
    H_out = (H_in + 2 * 1 - 3) // 2 + 1
    W_out = (W_in + 2 * 1 - 3) // 2 + 1
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    total = N * C * H_out * W_out
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    max_pool2d_kernel_s2[grid](x, out, N, C, H_in, W_in, H_out, W_out)
    return out


def triton_max_pool2d_s1(x):
    N, C, H_in, W_in = x.shape
    H_out = H_in
    W_out = W_in
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    total = N * C * H_out * W_out
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    max_pool2d_kernel_s1[grid](x, out, N, C, H_in, W_in, H_out, W_out)
    return out


def triton_adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    total = N * C
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](x, out, N, C, H, W)
    return out


def triton_linear(x, w, bias):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # w is [N, K], need [K, N] for matmul
    w_t = w.t().contiguous()
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_kernel[grid](x, w_t, bias, out, M, N, K)
    return out


def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    # Branch 1x1
    b1 = triton_conv2d(x, w_1x1, b_1x1, 1, 1, 0, 0)
    b1 = triton_relu(b1)

    # Branch 3x3
    b2 = triton_conv2d(x, w_3x3_r, b_3x3_r, 1, 1, 0, 0)
    b2 = triton_relu(b2)
    b2 = triton_conv2d(b2, w_3x3, b_3x3, 1, 1, 1, 1)
    b2 = triton_relu(b2)

    # Branch 5x5
    b3 = triton_conv2d(x, w_5x5_r, b_5x5_r, 1, 1, 0, 0)
    b3 = triton_relu(b3)
    b3 = triton_conv2d(b3, w_5x5, b_5x5, 1, 1, 2, 2)
    b3 = triton_relu(b3)

    # Branch pool
    bp = triton_max_pool2d_s1(x)
    bp = triton_conv2d(bp, w_pool, b_pool, 1, 1, 0, 0)
    bp = triton_relu(bp)

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
    p_inception5a_branch5x5_0_weight,