import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Fused Conv2d (implicit GEMM) + bias + ReLU kernel
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
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W',
         'KERNEL_H', 'KERNEL_W', 'STRIDE_H', 'STRIDE_W', 'PAD_H', 'PAD_W'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    HAS_BIAS: tl.constexpr, DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W
    K_TOTAL = IN_C * KERNEL_H * KERNEL_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_TOTAL, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kh, kw)
        kw_idx = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh_idx = tmp_k % KERNEL_H
        ic_idx = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < K_TOTAL
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * (IN_C * IN_H * IN_W) +
                  ic_idx[None, :] * (IN_H * IN_W) +
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: shape [OUT_C, IN_C, KERNEL_H, KERNEL_W]
        # w[oc, ic, kh, kw] = W_ptr + oc*(IN_C*KH*KW) + ic*(KH*KW) + kh*KW + kw
        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n
        w_ptrs = (W_ptr +
                  offs_n[None, :] * K_TOTAL +
                  ic_idx[:, None] * (KERNEL_H * KERNEL_W) +
                  kh_idx[:, None] * KERNEL_W +
                  kw_idx[:, None])
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + bias[None, :]

    # ReLU
    if DO_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output: Y shape [BATCH, OUT_C, OUT_H, OUT_W]
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
              offs_n[None, :] * (OUT_H * OUT_W) +
              out_h_idx[:, None] * OUT_W +
              out_w_idx[:, None])
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), do_relu=False):
    """Triton implicit GEMM conv2d with optional bias and ReLU."""
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OH = (H + 2*PH - KH) // SH + 1
    OW = (W + 2*PW - KW) // SW + 1

    y = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OH * OW
    has_bias = bias is not None

    if not has_bias:
        bias_ptr = x  # dummy, won't be used
    else:
        bias_ptr = bias

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        x, weight, bias_ptr, y,
        N, C_in, H, W,
        C_out, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        has_bias, do_relu,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Max Pool 2D kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def max_pool2d_kernel(
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
    total = N * C * OH * OW
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # Decode flat index -> (n, c, oh, ow)
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    # Compute max over kernel window
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


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size
    if isinstance(stride, int):
        SH = SW = stride
    else:
        SH, SW = stride
    if isinstance(padding, int):
        PH = PW = padding
    else:
        PH, PW = padding

    OH = (H + 2*PH - KH) // SH + 1
    OW = (W + 2*PW - KW) // SW + 1

    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, y,
        N, C, H, W, OH, OW,
        KH, KW, SH, SW, PH, PW,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Average Pool 2D kernel (global average pool for 1x1 output)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'IH', 'IW'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    X_ptr, Y_ptr,
    N: tl.constexpr, C: tl.constexpr,
    IH: tl.constexpr, IW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    total_nc = N * C
    offs_nc = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask_nc = offs_nc < total_nc

    c_idx = offs_nc % C
    n_idx = offs_nc // C

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    total_pixels = IH * IW

    for h in range(IH):
        for w in range(IW):
            x_ptrs = X_ptr + n_idx * (C * IH * IW) + c_idx * (IH * IW) + h * IW + w
            val = tl.load(x_ptrs, mask=mask_nc, other=0.0)
            acc = acc + val

    avg = acc / total_pixels
    tl.store(Y_ptr + offs_nc, avg, mask=mask_nc)


def triton_adaptive_avg_pool2d(x, output_size=(1, 1)):
    N, C, H, W = x.shape
    assert output_size == (1, 1) or output_size == 1
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    total_nc = N * C
    grid = lambda meta: (triton.cdiv(total_nc, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x, y,
        N, C, H, W,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Linear kernel (GEMM: Y = X @ W^T + b)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
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
        # W is [N, K], we want W^T so load W[offs_n, offs_k]
        w_tile = tl.load(W_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    has_bias = bias is not None
    if not has_bias:
        bias_ptr = x
    else:
        bias_ptr = bias
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias_ptr, y,
        M, N, K,
        has_bias,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# ReLU kernel (pointwise)
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
def relu_kernel(X_ptr, Y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offs, y, mask=mask)


def triton_relu(x):
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, y, N)
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel_function implementing GoogLeNet forward pass
# ─────────────────────────────────────────────────────────────────────────────

def inception_module(x,
                     w_1x1, b_1x1,
                     w_r3, b_r3, w_3x3, b_3x3,
                     w_r5, b_r5, w_5x5, b_5x5,
    """One inception module: 4 branches concatenated."""
    # Branch 1: 1x1 conv + relu
    b1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), do_relu=True)

    # Branch 2: 1x1 reduce + relu, then 3x3 conv + relu
    b2 = triton_conv2d(x, w_r3, b_r3, stride=(1,1), padding=(0,0), do_relu=True)
    b2 = triton_conv2d(b2, w_3x3, b_3x3, stride=(1,1), padding=(1,1), do_relu=True)

    # Branch 3: 1x1 reduce + relu, then 5x5 conv + relu
    b3 = triton_conv2d(x, w_r5, b_r5, stride=(1,1), padding=(0,0), do_relu=True)
    b3 = triton_conv2d(b3, w_5x5, b_5x5, stride=(1,1), padding=(2,2), do_relu=True)

    # Branch 4: max pool 3x3 stride=1 pad=1, then 1x1 conv + relu
    b4 = triton_max_pool2d(x, kernel_size=3, stride=1, padding=1)
    b4 = triton_conv2d(b4, w_pool, b_pool, stride=(1,1), padding=(0,0), do_relu=True)

    return torch.cat([b1, b2, b3, b4], dim=1)


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
    x = triton_conv2d(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), do_relu=True)
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)
    x = triton_conv2d(x, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), do_relu=True)
    x = triton_conv2d(x, p_conv3_weight, p_conv3_bias, stride=(1,1), padding=(1,1), do_relu=True)
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)

    # Inception 3a
    x = inception_module(x,