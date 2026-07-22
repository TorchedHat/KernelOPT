import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Triton implicit-GEMM convolution kernel
# Supports: 1x1, 3x3, 5x5, 7x7 kernels with arbitrary stride/padding
# Fuses: bias add + optional ReLU
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    M, N_out, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    DO_RELU: tl.constexpr,
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

    mask_m = offs_m < M
    mask_n = offs_n < OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K = IN_C * KERNEL_H * KERNEL_W
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Decode K -> (in_c, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (
            mask_m[:, None] &
            mask_k[None, :] &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (batch_idx[:, None] < N)
        )

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * stride_xn +
                  in_c[None, :] * stride_xc +
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> index [offs_n, in_c, kh, kw]
        w_ptrs = (W_ptr +
                  offs_n[None, :] * stride_wn +
                  in_c[:, None] * stride_wc +
                  kh[:, None] * stride_wh +
                  kw[:, None] * stride_ww)
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Optional ReLU
    if DO_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store: output shape [N, OUT_C, OUT_H, OUT_W] in NCHW
    # Linear index: batch * OUT_C*OUT_H*OUT_W + out_c * OUT_H*OUT_W + out_h * OUT_W + out_w
    out_idx = (batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
               offs_n[None, :] * (OUT_H * OUT_W) +
               out_h_idx[:, None] * OUT_W +
               out_w_idx[:, None])

    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(Y_ptr + out_idx, acc.to(tl.float32), mask=out_mask)


def conv2d_triton(x, weight, bias, stride=(1,1), padding=(0,0), do_relu=False):
    """
    """
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = weight.shape
    STRIDE_H, STRIDE_W = stride
    PAD_H, PAD_W = padding

    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    M = N * OUT_H * OUT_W
    K = IN_C * KERNEL_H * KERNEL_W

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        x, weight, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        KERNEL_H, KERNEL_W,
        M, OUT_C, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        DO_RELU=do_relu,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Max Pool 2D in Triton
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_OUT'],
)
@triton.jit
def max_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    N_OUT,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_OUT

    # Decode: offs -> (n, c, oh, ow)
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(Y_ptr + offs, max_val, mask=mask)


def max_pool2d_triton(x, kernel_size=3, stride=2, padding=1):
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding

    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1
    N_OUT = N * C * OUT_H * OUT_W

    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(N_OUT, meta['BLOCK_SIZE']),)

    max_pool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W,
        OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        N_OUT,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Average Pool 2D (global average pool to 1x1) in Triton
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
    ],
    key=['N_C'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    N_C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_C

    c = offs % C
    n = offs // C

    total = IN_H * IN_W
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    base = n * C * IN_H * IN_W + c * IN_H * IN_W

    # Loop over spatial positions
    for i in range(IN_H):
        for j in range(IN_W):
            x_idx = base + i * IN_W + j
            val = tl.load(X_ptr + x_idx, mask=mask, other=0.0)
            acc = acc + val

    acc = acc / total
    tl.store(Y_ptr + offs, acc, mask=mask)


def adaptive_avg_pool2d_triton(x, output_size=(1, 1)):
    assert output_size == (1, 1), "Only 1x1 output supported"
    N, C, IN_H, IN_W = x.shape
    N_C = N * C

    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(N_C, meta['BLOCK_SIZE']),)

    adaptive_avg_pool2d_1x1_kernel[grid](
        x, y,
        N, C, IN_H, IN_W,
        N_C,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Linear (GEMM) in Triton
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=4, num_warps=4),
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def linear_triton(x, weight, bias):
    """
    """
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# ReLU in Triton
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
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

def inception_module(
    x,
    w1x1, b1x1,
    w3x3_reduce, b3x3_reduce,
    w3x3, b3x3,
    w5x5_reduce, b5x5_reduce,
    w5x5, b5x5,
    w_pool_proj, b_pool_proj,
):
    # Branch 1: 1x1 conv + relu
    b1 = conv2d_triton(x, w1x1, b1x1, stride=(1, 1), padding=(0, 0), do_relu=True)

    # Branch 2: 1x1 reduce + relu, then 3x3 conv + relu
    b2 = conv2d_triton(x, w3x3_reduce, b3x3_reduce, stride=(1, 1), padding=(0, 0), do_relu=True)
    b2 = conv2d_triton(b2, w3x3, b3x3, stride=(1, 1), padding=(1, 1), do_relu=True)

    # Branch 3: 1x1 reduce + relu, then 5x5 conv + relu
    b3 = conv2d_triton(x, w5x5_reduce, b5x5_reduce, stride=(1, 1), padding=(0, 0), do_relu=True)
    b3 = conv2d_triton(b3, w5x5, b5x5, stride=(1, 1), padding=(2, 2), do_relu=True)

    # Branch 4: max pool 3x3 stride=1 pad=1, then 1x1 conv + relu
    b4 = max_pool2d_triton(x, kernel_size=3, stride=1, padding=1)
    b4 = conv2d_triton(b4, w_pool_proj, b_pool_proj, stride=(1, 1), padding=(0, 0), do_relu=True)

    return torch.cat([b1, b2, b3, b4], dim=1)


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
    # conv1: 7x7, stride=2, pad=3 + relu
    x = conv2d_triton(x, p_conv1_weight, p_conv1_bias, stride=(2, 2), padding=(3, 3), do_relu=True)
    # maxpool1: 3x3, stride=2, pad=1
    x = max_pool2d_triton(x, kernel_size=3, stride=2, padding=1)

    # conv2: 1x1 + relu
    x = conv2d_triton(x, p_conv2_weight, p_conv2_bias, stride=(1, 1), padding=(0,