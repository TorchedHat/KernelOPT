import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: fused bias + ReLU (pointwise epilogue)
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
def bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    numel, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = offsets % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    out = x + bias
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


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
def bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    numel, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = offsets % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    out = x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Triton kernel: max pool 2d (3x3, stride=2, pad=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N', 'C', 'IN_H', 'IN_W', 'OUT_H', 'OUT_W'],
)
@triton.jit
def maxpool2d_kernel(
    in_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W, PAD_H, PAD_W, KH, KW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    numel = N * C * OUT_H * OUT_W
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

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
            val = tl.load(in_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# ============================================================
# Triton kernel: max pool 2d (3x3, stride=1, pad=1) for inception branch_pool
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N', 'C', 'IN_H', 'IN_W'],
)
@triton.jit
def maxpool2d_s1_kernel(
    in_ptr, out_ptr,
    N, C, IN_H, IN_W,
    BLOCK_SIZE: tl.constexpr,
):
    # stride=1, pad=1, kernel=3x3, so OUT_H=IN_H, OUT_W=IN_W
    pid = tl.program_id(0)
    numel = N * C * IN_H * IN_W
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    ow = offsets % IN_W
    tmp = offsets // IN_W
    oh = tmp % IN_H
    tmp2 = tmp // IN_H
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh + kh - 1
            iw = ow + kw - 1
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            in_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(in_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# ============================================================
# Triton kernel: adaptive avg pool 2d (output 1x1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def adaptive_avg_pool_1x1_kernel(
    in_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    numel = N * C
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c = offsets % C
    n = offsets // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    HW = H * W
    for h in range(H):
        for w in range(W):
            in_idx = n * (C * H * W) + c * (H * W) + h * W + w
            val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
            acc = acc + val

    acc = acc / HW.to(tl.float32)
    tl.store(out_ptr + offsets, acc, mask=mask)


# ============================================================
# Triton kernel: linear (GEMM) with bias
# M x K @ K x N + N -> M x N
# ============================================================
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
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x, w, acc)

    offs_m_out = offs_m
    offs_n_out = offs_n
    bias = tl.load(bias_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)
    tl.store(out_ptr + offs_m_out[:, None] * N + offs_n_out[None, :], acc, mask=out_mask)


# ============================================================
# Triton kernel: implicit GEMM convolution (general)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KH', 'KW'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    WITH_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    n = tmp // OUT_H

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        ic = k_offs % IN_C
        kw_idx = (k_offs // IN_C) % KW
        kh_idx = k_offs // (IN_C * KW)

        ih = oh[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            (offs_m[:, None] < M) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (k_offs[None, :] < TOTAL_K)
        )

        x_idx = n[:, None] * (IN_C * IN_H * IN_W) + ic[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=valid_x, other=0.0)

        # Weight layout: [OUT_C, IN_C, KH, KW]
        # w_idx for each (k, n): w[offs_n, ic, kh_idx, kw_idx]
        w_idx = (
            offs_n[None, :] * (IN_C * KH * KW) +
            ic[:, None] * (KH * KW) +
            kh_idx[:, None] * KW +
            kw_idx[:, None]
        )
        valid_w = (offs_n[None, :] < OUT_C) & (k_offs[:, None] < TOTAL_K)
        w_tile = tl.load(W_ptr + w_idx, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    if WITH_RELU:
        acc = tl.where(acc > 0.0, acc, 0.0)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    # Output layout: [BATCH, OUT_C, OUT_H, OUT_W] -> index: n*OUT_C*OUT_H*OUT_W + c*OUT_H*OUT_W + h*OUT_W + w
    out_idx = (
        n[:, None] * (OUT_C * OUT_H * OUT_W) +
        offs_n[None, :] * (OUT_H * OUT_W) +
        oh[:, None] * OUT_W +
        ow[:, None]
    )
    tl.store(Y_ptr + out_idx, acc, mask=out_mask)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), relu=False):
    N, C, H, W = x.shape
    OC, IC, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding
    OH = (H + 2*PH - KH) // SH + 1
    OW = (W + 2*PW - KW) // SW + 1

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous()

    y = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OH * OW
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    conv2d_kernel[grid](
        x_contig, w_contig, b_contig, y,
        N, C, H, W,
        OC, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        relu,
    )
    return y


def triton_maxpool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    OH = (H + 2*padding - kernel_size) // stride + 1
    OW = (W + 2*padding - kernel_size) // stride + 1
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    numel = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x.contiguous(), y,
        N, C, H, W, OH, OW,
        stride, stride, padding, padding, kernel_size, kernel_size,
    )
    return y


def triton_maxpool2d_s1(x):
    # stride=1, pad=1, kernel=3x3
    N, C, H, W = x.shape
    y = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)
    numel = N * C * H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    maxpool2d_s1_kernel[grid](
        x.contiguous(), y,
        N, C, H, W,
    )
    return y


def triton_adaptive_avg_pool_1x1(x):
    N, C, H, W = x.shape
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    numel = N * C
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool_1x1_kernel[grid](
        x.contiguous(), y,
        N, C, H, W,
    )
    return y


def triton_linear(x, weight, bias):
    # x: [M, K], weight: [N, K], bias: [N]
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_kernel[grid](
        x.contiguous(), weight.contiguous(), bias.contiguous(), y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return y


def inception_module(x,
                     w_1x1, b_1x1,
                     w_r3, b_r3, w_3x3, b_3x3,
                     w_r5, b_r5, w_5x5, b_5x5,
    # branch1x1
    b1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), relu=True)

    # branch3x3
    r3 = triton_conv2d(x, w_r3, b_r3, stride=(1,1), padding=(0,0), relu=True)
    b3 = triton_conv2d(r3, w_3x3, b_3x3, stride=(1,1), padding=(1,1), relu=True)

    # branch5x5
    r5 = triton_conv2d(x, w_r5, b_r5, stride=(1,1), padding=(0,0), relu=True)
    b5 = triton_conv2d(r5, w_5x5, b_5x5, stride=(1,1), padding=(2,2), relu=True)

    # branch_pool: maxpool(3x3,s=1,p=1) then 1x1 conv
    pool = triton_maxpool2d_s1(x)
    bp = triton_conv2d(pool, w_pool, b_pool, stride=(1,1), padding=(0,0), relu=True)

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
    # conv1: 7x7, stride=2, pad=3
    out = triton_conv2d(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), relu=True)
    # maxpool1: 3x3, stride=2, pad=1
    out = triton_maxpool2d(out, kernel_size=3, stride=2, padding=1)
    # conv2: 1x1
    out = triton_conv2d(out, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), relu=True)
    # conv3: 3x3, pad=1
    out = tri