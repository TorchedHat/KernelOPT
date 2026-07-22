import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Fused Conv2d + Bias + ReLU kernel (implicit GEMM)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    WITH_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Decode M index -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * (IN_C * IN_H * IN_W) +
                  in_c[None, :] * (IN_H * IN_W) +
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: [OUT_C, IN_C, KERNEL_H, KERNEL_W] -> index [n, k]
        # w[oc, ic, kh, kw] = W_ptr + oc*(IN_C*KH*KW) + ic*(KH*KW) + kh*KW + kw
        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n

        w_ptrs = (W_ptr +
                  offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) +
                  in_c[:, None] * (KERNEL_H * KERNEL_W) +
                  kh[:, None] * KERNEL_W +
                  kw[:, None])
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    if WITH_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store output: Y[batch, oc, oh, ow] = Y_ptr + batch*(OUT_C*OUT_H*OUT_W) + oc*(OUT_H*OUT_W) + oh*OUT_W + ow
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
              offs_n[None, :] * (OUT_H * OUT_W) +
              out_h_idx[:, None] * OUT_W +
              out_w_idx[:, None])
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_triton(x, weight, bias, stride=(1,1), padding=(0,0), with_relu=True):
    """Triton conv2d + bias + optional relu"""
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OH = (H + 2*PH - KH) // SH + 1
    OW = (W + 2*PW - KW) // SW + 1

    x_contig = x.contiguous()
    w_contig = weight.contiguous()
    b_contig = bias.contiguous()

    y = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OH * OW
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        x_contig, w_contig, b_contig, y,
        N, C_in, H, W,
        C_out, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        with_relu,
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
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    N: tl.constexpr, C: tl.constexpr,
    IH: tl.constexpr, IW: tl.constexpr,
    OH: tl.constexpr, OW: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * OH * OW

    mask = offs < total

    # Decode index
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            x_ptrs = X_ptr + n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(Y_ptr + offs, max_val, mask=mask)


def maxpool2d_triton(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    OH = (H + 2*PH - KH) // SH + 1
    OW = (W + 2*PW - KW) // SW + 1

    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x.contiguous(), y,
        N, C, H, W, OH, OW,
        KH, KW, SH, SW, PH, PW,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Average Pool 2D (global average pool to 1x1)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'IH', 'IW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    X_ptr, Y_ptr,
    N: tl.constexpr, C: tl.constexpr,
    IH: tl.constexpr, IW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C
    mask = offs < total

    c = offs % C
    n = offs // C

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for h in range(IH):
        for w in range(IW):
            x_ptrs = X_ptr + n * (C * IH * IW) + c * (IH * IW) + h * IW + w
            val = tl.load(x_ptrs, mask=mask, other=0.0)
            acc = acc + val

    avg = acc / (IH * IW)
    tl.store(Y_ptr + offs, avg, mask=mask)


def adaptive_avg_pool2d_triton(x):
    N, C, H, W = x.shape
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    total = N * C
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x.contiguous(), y.view(N * C),
        N, C, H, W,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Linear kernel (GEMM): Y = X @ W^T + b
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
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
        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile.T, acc=acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


def linear_triton(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x.contiguous(), weight.contiguous(), bias.contiguous(), y,
        M, N, K,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Inception module helper
# ─────────────────────────────────────────────────────────────────────────────

def inception_module(x,
                     w_1x1, b_1x1,
                     w_r3, b_r3, w_3x3, b_3x3,
                     w_r5, b_r5, w_5x5, b_5x5,
    # Branch 1x1
    branch1 = conv2d_triton(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), with_relu=True)

    # Branch 3x3: reduce then 3x3
    branch3_r = conv2d_triton(x, w_r3, b_r3, stride=(1,1), padding=(0,0), with_relu=True)
    branch3 = conv2d_triton(branch3_r, w_3x3, b_3x3, stride=(1,1), padding=(1,1), with_relu=True)

    # Branch 5x5: reduce then 5x5
    branch5_r = conv2d_triton(x, w_r5, b_r5, stride=(1,1), padding=(0,0), with_relu=True)
    branch5 = conv2d_triton(branch5_r, w_5x5, b_5x5, stride=(1,1), padding=(2,2), with_relu=True)

    # Branch pool: maxpool then 1x1
    pool = maxpool2d_triton(x, kernel_size=3, stride=1, padding=1)
    branch_pool = conv2d_triton(pool, w_pool, b_pool, stride=(1,1), padding=(0,0), with_relu=True)

    return torch.cat([branch1, branch3, branch5, branch_pool], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel function
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
    out = conv2d_triton(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), with_relu=True)
    # maxpool1: 3x3, stride=2, pad=1
    out = maxpool2d_triton(out, kernel_size=3, stride=2, padding=1)
    # conv2: 1x1
    out = conv2d_triton(out, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), with_relu=True)
    # conv3: 3x3, pad=1
    out = conv2d_triton(out, p_conv3_weight, p_conv3_bias, stride=(1,1), padding=(1,1), with_relu=True)
    # maxpool2: 3x3, stride=2, pad=1
    out = maxpool2d_triton(out, kernel_size=3, stride=2, padding=1)

    # inception3a
    out = inception_module(out,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
        p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    )

    # inception3b
    out = inception_module(out,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
        p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
        p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
        p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
        p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    )

    # maxpool3
    out = maxpool2d_triton(out, kernel_size=3, stride=2, padding=1)

    # inception4a
    out = inception_module(out,
        p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
        p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
        p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
        p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,