import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Triton implicit-GEMM convolution kernel (NHWC layout internally)
# Fuses: conv + bias + optional ReLU
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    # tensor dims
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides for X (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (NCHW -> stored as [OUT_C, IN_C, KH, KW])
    sw_o, sw_i, sw_h, sw_w,
    # total sizes for autotuning key
    M, N, K,
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

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c_idx = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KERNEL_W
        kh_idx = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m < M
        valid_k = offs_k < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m[:, None] & valid_k[None, :] & valid_ih & valid_iw

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> index [offs_n, in_c_idx, kh_idx, kw_idx]
        valid_n = offs_n < OUT_C
        mask_w = valid_k[:, None] & valid_n[None, :]
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=(offs_n < OUT_C), other=0.0)
    acc = acc + bias[None, :]

    # Optional ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output (NCHW layout)
    valid_m = offs_m < M
    valid_n = offs_n < OUT_C
    mask_out = valid_m[:, None] & valid_n[None, :]

    # Output in NCHW: [batch, out_c, out_h, out_w]
    # stride: [OUT_C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1]
    out_ptrs = (Y_ptr
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_conv2d(x_nchw, weight, bias, stride=1, padding=0, apply_relu=False):
    """
    """
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    N, IN_C, IN_H, IN_W = x_nchw.shape
    OUT_C, _, KH, KW = weight.shape
    OUT_H = (IN_H + 2 * padding[0] - KH) // stride[0] + 1
    OUT_W = (IN_W + 2 * padding[1] - KW) // stride[1] + 1

    # Convert input to NHWC for coalesced access
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x_nchw.device, dtype=torch.float32)

    M = N * OUT_H * OUT_W
    Nk = OUT_C
    K = IN_C * KH * KW

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Nk, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        x_nhwc, weight, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride[0], stride[1],
        padding[0], padding[1],
        # NHWC strides
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        # Weight strides (NCHW)
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        M, Nk, K,
        APPLY_RELU=apply_relu,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Max Pool 2D kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def max_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: [N, C, OUT_H, OUT_W]
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(Y_ptr + offs, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    N, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
    OUT_W = (IN_W + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    N_TOTAL = N * C * OUT_H * OUT_W

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W,
        OUT_H, OUT_W,
        kernel_size[0], kernel_size[1],
        stride[0], stride[1],
        padding[0], padding[1],
        N_TOTAL,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Average Pool 2D (global average pool to 1x1)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
    ],
    key=['NC'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    X_ptr, Y_ptr,
    NC, IN_H, IN_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NC

    HW = IN_H * IN_W
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for i in range(HW):
        val = tl.load(X_ptr + offs * HW + i, mask=mask, other=0.0)
        acc = acc + val

    avg = acc / HW
    tl.store(Y_ptr + offs, avg, mask=mask)


def triton_adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    NC = N * C
    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x.contiguous(), y,
        NC, H, W,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Linear kernel (fused matmul + bias)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=4, num_warps=4),
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
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        # W is [N, K], we want W^T effectively: Y = X @ W^T
        w_tile = tl.load(W_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=(offs_n < N), other=0.0)
    acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x.contiguous(), weight.contiguous(), bias, y,
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
    # Branch 1: 1x1 conv + relu
    branch1 = triton_conv2d(x, w_1x1, b_1x1, stride=1, padding=0, apply_relu=True)

    # Branch 2: 1x1 reduce + 3x3 conv + relu
    b2_r = triton_conv2d(x, w_r3, b_r3, stride=1, padding=0, apply_relu=True)
    branch2 = triton_conv2d(b2_r, w_3x3, b_3x3, stride=1, padding=1, apply_relu=True)

    # Branch 3: 1x1 reduce + 5x5 conv + relu
    b3_r = triton_conv2d(x, w_r5, b_r5, stride=1, padding=0, apply_relu=True)
    branch3 = triton_conv2d(b3_r, w_5x5, b_5x5, stride=1, padding=2, apply_relu=True)

    # Branch 4: max pool 3x3 s1 p1 + 1x1 conv + relu
    pooled = triton_max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch4 = triton_conv2d(pooled, w_pool, b_pool, stride=1, padding=0, apply_relu=True)

    return torch.cat([branch1, branch2, branch3, branch4], dim=1)


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
    x = triton_conv2d(x, p_conv1_weight, p_conv1_bias, stride=2, padding=3, apply_relu=True)
    # maxpool1: 3x3, stride=2, pad=1
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)

    # conv2: 1x1
    x = triton_conv2d(x, p_conv2_weight, p_conv2_bias, stride=1, padding=0, apply_relu=True)
    # conv3: 3x3, pad=1
    x = triton_conv2d(x, p_conv3_weight, p_conv3_bias, stride=1, padding=1, apply_relu=True)
    # maxpool2: 3x3, stride=2, pad=1
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)

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