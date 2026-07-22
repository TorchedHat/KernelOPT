import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Fused Conv2d + Bias + ReLU kernel (implicit GEMM style)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    M, N, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    USE_RELU: tl.constexpr,
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

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_block in range(BLOCK_K_COUNT):
        k_start = k_block * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c_idx = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KERNEL_W
        kh_idx = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = (batch_idx < BATCH) & (out_h_idx < OUT_H) & (out_w_idx < OUT_W)
        valid_k = offs_k < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m[:, None] & valid_k[None, :] & valid_ih & valid_iw

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: shape (OUT_C, IN_C, KH, KW)
        # offs_n -> out_c, offs_k -> (in_c, kh, kw)
        mask_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    if USE_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store
    mask_out = ((batch_idx < BATCH) & (out_h_idx < OUT_H) & (out_w_idx < OUT_W))[:, None] & (offs_n < OUT_C)[None, :]
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_triton(x, weight, bias, stride=(1,1), padding=(0,0), use_relu=False):
    """Triton conv2d with optional bias and ReLU fusion."""
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    OUT_H = (IN_H + 2 * pad_h - KERNEL_H) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KERNEL_W) // stride_w + 1

    M = BATCH * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KERNEL_H * KERNEL_W

    y = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    if bias is None:
        bias = torch.zeros(OUT_C, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        x, weight, bias, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        stride_h, stride_w,
        pad_h, pad_w,
        KERNEL_H, KERNEL_W,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        use_relu,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Max Pool 2D kernel
# ─────────────────────────────────────────────────────────────────────────────

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
    BATCH, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: offs -> (batch, c, out_h, out_w)
    out_w_idx = offs % OUT_W
    tmp = offs // OUT_W
    out_h_idx = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c_idx = tmp2 % C
    batch_idx = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            ih = out_h_idx * STRIDE_H + kh - PAD_H
            iw = out_w_idx * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + batch_idx * C * IN_H * IN_W + c_idx * IN_H * IN_W + ih * IN_W + iw
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(Y_ptr + offs, max_val, mask=mask)


def maxpool2d_triton(x, kernel_size=3, stride=2, padding=1):
    BATCH, C, IN_H, IN_W = x.shape
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = padding

    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1

    y = torch.empty((BATCH, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    N_TOTAL = BATCH * C * OUT_H * OUT_W

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, y,
        BATCH, C, IN_H, IN_W,
        OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        N_TOTAL,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Average Pool 2D kernel (global average pool for 1x1 output)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 128}, num_warps=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
    ],
    key=['HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    X_ptr, Y_ptr,
    BATCH, C, H, W,
    HW,
    BLOCK_N: tl.constexpr,
):
    # One program per (batch, channel)
    pid = tl.program_id(0)
    batch_idx = pid // C
    c_idx = pid % C

    base = X_ptr + batch_idx * C * H * W + c_idx * H * W

    acc = tl.zeros((1,), dtype=tl.float32)
    for block_start in range(0, HW, BLOCK_N):
        offs = block_start + tl.arange(0, BLOCK_N)
        mask = offs < HW
        vals = tl.load(base + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    avg = acc / HW
    tl.store(Y_ptr + pid, avg)


def adaptive_avg_pool2d_triton(x, output_size=(1, 1)):
    BATCH, C, H, W = x.shape
    assert output_size == (1, 1) or output_size == 1
    y = torch.empty((BATCH, C, 1, 1), device=x.device, dtype=x.dtype)
    HW = H * W
    grid = (BATCH * C,)
    adaptive_avg_pool_kernel[grid](
        x, y,
        BATCH, C, H, W,
        HW,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Linear kernel (GEMM: M x K @ K x N)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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

        x_tile = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                         mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
                         mask=mask_w, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
             acc.to(tl.float32), mask=mask_out)


def linear_triton(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# ReLU kernel (standalone, for inception branches that need it)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def relu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offs, y, mask=mask)


def relu_triton(x):
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, y, N)
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Inception module helper
# ─────────────────────────────────────────────────────────────────────────────

def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    # Branch 1: 1x1 conv + relu
    branch1 = conv2d_triton(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), use_relu=True)

    # Branch 2: 1x1 reduce + relu, then 3x3 conv + relu
    branch2 = conv2d_triton(x, w_3x3_r, b_3x3_r, stride=(1,1), padding=(0,0), use_relu=True)
    branch2 = conv2d_triton(branch2, w_3x3, b_3x3, stride=(1,1), padding=(1,1), use_relu=True)

    # Branch 3: 1x1 reduce + relu, then 5x5 conv + relu
    branch3 = conv2d_triton(x, w_5x5_r, b_5x5_r, stride=(1,1), padding=(0,0), use_relu=True)
    branch3 = conv2d_triton(branch3, w_5x5, b_5x5, stride=(1,1), padding=(2,2), use_relu=True)

    # Branch 4: maxpool 3x3 s1 p1, then 1x1 conv + relu
    branch4 = maxpool2d_triton(x, kernel_size=3, stride=1, padding=1)
    branch4 = conv2d_triton(branch4, w_pool, b_pool, stride=(1,1), padding=(0,0), use_relu=True)

    return torch.cat([branch1, branch2, branch3, branch4], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel_function
# ─────────────────────────────────────────────────────────────────────────────

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
    # conv1: 7x7, stride=2, pad=3
    x = conv2d_triton(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), use_relu=True)
    # maxpool1: 3x3,