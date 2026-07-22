import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused Conv2d + Bias + ReLU kernel (implicit GEMM style)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    IN_H, IN_W, OUT_H, OUT_W,
    BATCH, IN_C, OUT_C,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    HAS_RELU: tl.constexpr,
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

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    if HAS_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_bias_relu(x, w, b, stride=(1,1), padding=(0,0), has_relu=True):
    """Fused conv2d + bias + optional relu"""
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = w.shape
    assert C_in == C_in_w

    SH, SW = stride
    PH, PW = padding

    OH = (H + 2*PH - KH) // SH + 1
    OW = (W + 2*PW - KW) // SW + 1

    y = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OH * OW
    Nout = C_out
    K = C_in * KH * KW

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(Nout, meta['BLOCK_N']))

    conv2d_bias_relu_kernel[grid](
        x, w, b, y,
        M, Nout, K,
        H, W, OH, OW,
        N, C_in, C_out,
        KH, KW,
        SH, SW,
        PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        HAS_RELU=has_relu,
    )
    return y


# ============================================================
# MaxPool2d kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N_total'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    N_total,
    IN_H, IN_W, OUT_H, OUT_W,
    CHANNELS,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    # Decode: offs -> (n, c, oh, ow)
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % CHANNELS
    n = tmp2 // CHANNELS

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for ki in range(KH):
        for kj in range(KW):
            ih = oh * STRIDE_H + ki - PAD_H
            iw = ow * STRIDE_W + kj - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + n * stride_xn + c * stride_xc + ih * stride_xh + iw * stride_xw
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    y_ptrs = Y_ptr + n * stride_yn + c * stride_yc + oh * stride_yh + ow * stride_yw
    tl.store(y_ptrs, max_val, mask=mask)


def maxpool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding

    OH = (H + 2*PH - KH) // SH + 1
    OW = (W + 2*PW - KW) // SW + 1

    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    N_total = N * C * OH * OW

    grid = lambda meta: (triton.cdiv(N_total, meta['BLOCK_SIZE']),)

    maxpool2d_kernel[grid](
        x, y,
        N_total,
        H, W, OH, OW,
        C,
        KH, KW,
        SH, SW,
        PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )
    return y


# ============================================================
# Adaptive Average Pool2d kernel (global average pool)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['NC'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    X_ptr, Y_ptr,
    NC, IN_H, IN_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NC

    n = offs // (stride_xn // (stride_xc * IN_H * IN_W) if stride_xn > 0 else 1)
    # simpler: decode n and c from offs
    # NC = N * C, offs indexes (n, c)
    # We'll just sum over all H*W
    # offs -> n*C + c, but we need to know C
    # Pass C separately
    # Actually let's just do it differently - pass N and C

    # This kernel is called with NC = N*C
    # We need to know C to decode n and c
    # Let's just use flat indexing: for each (n,c) pair, sum over H*W
    # We'll pass C as a separate arg
    # Rewrite below
    pass


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['NC'],
)
@triton.jit
def adaptive_avg_pool2d_kernel_v2(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    NC = N * C
    mask = offs < NC

    c = offs % C
    n = offs // C

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    total = IN_H * IN_W

    for h in range(IN_H):
        for w in range(IN_W):
            x_ptrs = X_ptr + n * stride_xn + c * stride_xc + h * stride_xh + w * stride_xw
            val = tl.load(x_ptrs, mask=mask, other=0.0)
            acc = acc + val

    acc = acc / total
    tl.store(Y_ptr + offs, acc, mask=mask)


def adaptive_avg_pool2d(x, output_size=(1, 1)):
    N, C, H, W = x.shape
    assert output_size == (1, 1), "Only global average pooling supported"
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    NC = N * C

    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel_v2[grid](
        x, y,
        N, C, H, W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        NC=NC,
    )
    return y


# ============================================================
# Linear kernel (GEMM: [M, K] x [K, N] -> [M, N])
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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
        w_tile = tl.load(W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                         mask=mask_w, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


def linear(x, w, b):
    """x: [M, K], w: [N, K], b: [N] -> [M, N]"""
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, w, b, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
    )
    return y


# ============================================================
# Inception module helper
# ============================================================

def inception_module(x,
                     w1x1, b1x1,
                     w3x3_r, b3x3_r, w3x3, b3x3,
                     w5x5_r, b5x5_r, w5x5, b5x5,
    """
    """
    # branch1x1
    b1 = conv2d_bias_relu(x, w1x1, b1x1, stride=(1,1), padding=(0,0), has_relu=True)

    # branch3x3
    b3 = conv2d_bias_relu(x, w3x3_r, b3x3_r, stride=(1,1), padding=(0,0), has_relu=True)
    b3 = conv2d_bias_relu(b3, w3x3, b3x3, stride=(1,1), padding=(1,1), has_relu=True)

    # branch5x5
    b5 = conv2d_bias_relu(x, w5x5_r, b5x5_r, stride=(1,1), padding=(0,0), has_relu=True)
    b5 = conv2d_bias_relu(b5, w5x5, b5x5, stride=(1,1), padding=(2,2), has_relu=True)

    # branch_pool
    bp = maxpool2d(x, kernel_size=3, stride=1, padding=1)
    bp = conv2d_bias_relu(bp, w_pool, b_pool, stride=(1,1), padding=(0,0), has_relu=True)

    return torch.cat([b1, b3, b5, bp], dim=1)


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
    # conv1: 7x7, stride=2, padding=3
    x = conv2d_bias_relu(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), has_relu=True)
    # maxpool1: 3x3, stride=2, padding=1
    x = maxpool2d(x, kernel_size=3, stride=2, padding=1)
    # conv2: 1x1
    x = conv2d_bias_relu(x, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), has_relu=True)
    # conv3: 3x3, padding=1
    x = conv2d_bias_relu(x, p_conv3_weight, p_conv3_bias, stride=(1,1), padding=(1,1), has_relu=True)
    # maxpool2: 3x3, stride=2, padding=1
    x = maxpool2d(x, kernel_size=3, stride=2, padding=1)

    # inception3a
    x = inception_module(x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
        p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    )

    # inception3b
    x = inception_module(x,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,