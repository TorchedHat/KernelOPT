import torch
import triton
import triton.language as tl
import math

# ─── Fused Conv2d + Bias + ReLU kernel (implicit GEMM) ───────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    # Tensor dims
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    # Kernel params
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    # Strides for X (NHWC layout)
    sx_n, sx_h, sx_w, sx_c,
    # Strides for W (out_c, kh, kw, in_c) layout
    sw_o, sw_h, sw_w, sw_i,
    # Strides for Y (NHWC layout)
    sy_n, sy_h, sy_w, sy_c,
    # Problem size for autotuning
    M, N, K,
    # Whether to apply relu
    APPLY_RELU: tl.constexpr,
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

    TOTAL_K = IN_C * KH * KW
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c_idx = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KW
        kh_idx = offs_k // (IN_C * KW)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * sx_n +
                  in_c_idx[None, :] * sx_c)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: shape (OUT_C, KH, KW, IN_C) -> index [offs_n, kh, kw, in_c]
        valid_n = offs_n[None, :] < N
        mask_w = valid_k[:, None] & valid_n
        w_ptrs = (W_ptr +
                  offs_n[None, :] * sw_o +
                  kh_idx[:, None] * sw_h +
                  kw_idx[:, None] * sw_w +
                  in_c_idx[:, None] * sw_i)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output (NHWC)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * sy_n +
              out_h_idx[:, None] * sy_h +
              out_w_idx[:, None] * sy_w +
              offs_n[None, :] * sy_c)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_conv2d(x_nhwc, weight, bias, stride, padding, apply_relu=True):
    """
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1

    # Reorder weight to (OUT_C, KH, KW, IN_C) for better access pattern
    w_reordered = weight.permute(0, 2, 3, 1).contiguous()  # (OUT_C, KH, KW, IN_C)

    y_nhwc = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    M = N * OUT_H * OUT_W
    Nout = OUT_C
    K = IN_C * KH * KW

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Nout, meta['BLOCK_N']),
    )

    # Strides for x_nhwc
    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    # Strides for w_reordered (OUT_C, KH, KW, IN_C)
    sw_o, sw_h, sw_w, sw_i = w_reordered.stride()
    # Strides for y_nhwc
    sy_n, sy_h, sy_w, sy_c = y_nhwc.stride()

    conv2d_bias_relu_kernel[grid](
        x_nhwc, w_reordered, bias, y_nhwc,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        sx_n, sx_h, sx_w, sx_c,
        sw_o, sw_h, sw_w, sw_i,
        sy_n, sy_h, sy_w, sy_c,
        M, Nout, K,
        APPLY_RELU=apply_relu,
    )
    return y_nhwc


# ─── Max Pool 2D kernel ───────────────────────────────────────────────────────

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
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    # Strides (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    sy_n, sy_h, sy_w, sy_c,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    # Decode: N_total = BATCH * OUT_H * OUT_W * C
    c_idx = offs % C
    tmp = offs // C
    ow_idx = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh_idx = tmp2 % OUT_H
    n_idx = tmp2 // OUT_H

    result = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh_idx * STRIDE_H + kh - PAD_H
            iw = ow_idx * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + n_idx * sx_n + ih * sx_h + iw * sx_w + c_idx * sx_c
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            result = tl.maximum(result, val)

    y_ptrs = Y_ptr + n_idx * sy_n + oh_idx * sy_h + ow_idx * sy_w + c_idx * sy_c
    tl.store(y_ptrs, result, mask=mask)


def triton_maxpool2d(x_nhwc, kernel_size, stride, padding):
    N, IN_H, IN_W, C = x_nhwc.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding

    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1

    y_nhwc = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)

    N_total = N * OUT_H * OUT_W * C
    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    sy_n, sy_h, sy_w, sy_c = y_nhwc.stride()

    grid = lambda meta: (triton.cdiv(N_total, meta['BLOCK_SIZE']),)

    maxpool2d_kernel[grid](
        x_nhwc, y_nhwc,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        sx_n, sx_h, sx_w, sx_c,
        sy_n, sy_h, sy_w, sy_c,
        N_total,
    )
    return y_nhwc


# ─── Adaptive Avg Pool 2D (global) kernel ────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 16}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    X_ptr, Y_ptr,
    BATCH, C, H, W,
    HW,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # One program per (batch, channel_block)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for hw_start in range(0, HW, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < HW
        h_idx = offs_hw // W
        w_idx = offs_hw % W

        # Load X[pid_n, h, w, c] for all hw and c in block
        # Shape: (BLOCK_HW, BLOCK_C)
        x_ptrs = (X_ptr +
                  h_idx[:, None] * (W * C) +
                  w_idx[:, None] * C +
                  offs_c[None, :])
        mask_xhw = mask_hw[:, None] & mask_c[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_xhw, other=0.0)
        acc = acc + tl.sum(x_tile, axis=0)

    avg = acc / HW
    y_ptrs = Y_ptr + pid_n * C + offs_c
    tl.store(y_ptrs, avg, mask=mask_c)


def triton_adaptive_avg_pool(x_nhwc):
    N, H, W, C = x_nhwc.shape
    y = torch.empty((N, C), device=x_nhwc.device, dtype=torch.float32)
    HW = H * W

    grid = lambda meta: (N, triton.cdiv(C, meta['BLOCK_C']))
    adaptive_avg_pool_kernel[grid](
        x_nhwc, y,
        N, C, H, W, HW,
    )
    return y  # shape (N, C)


# ─── Linear kernel ────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
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

        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_k[:, None] * N + offs_n[None, :], mask=mask_w, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


def triton_linear(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    M, K = x.shape
    N = weight.shape[0]
    # weight is (N, K), we need (K, N) for matmul
    w_t = weight.t().contiguous()  # (K, N)
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
    )
    return y


# ─── Inception module helper ──────────────────────────────────────────────────

def inception_module(x_nhwc,
                     w1x1, b1x1,
                     w3x3_r, b3x3_r, w3x3, b3x3,
                     w5x5_r, b5x5_r, w5x5, b5x5,
    """
    """
    # Branch 1: 1x1 conv + relu
    b1 = triton_conv2d(x_nhwc, w1x1, b1x1, (1, 1), (0, 0), apply_relu=True)

    # Branch 2: 1x1 reduce + 3x3 conv + relu
    b2 = triton_conv2d(x_nhwc, w3x3_r, b3x3_r, (1, 1), (0, 0), apply_relu=True)
    b2 = triton_conv2d(b2, w3x3, b3x3, (1, 1), (1, 1), apply_relu=True)

    # Branch 3: 1x1 reduce + 5x5 conv + relu
    b3 = triton_conv2d(x_nhwc, w5x5_r, b5x5_r, (1, 1), (0, 0), apply_relu=True)
    b3 = triton_conv2d(b3, w5x5, b5x5, (1, 1), (2, 2), apply_relu=True)

    # Branch 4: maxpool 3x3 s1 p1 + 1x1 conv + relu
    b4 = triton_maxpool2d(x_nhwc, 3, 1, 1)
    b4 = triton_conv2d(b4, w_pool, b_pool, (1, 1), (0, 0), apply_relu=True)

    # Concatenate along channel dim (NHWC -> dim 3)
    return torch.cat([b1, b2, b3, b4], dim=3)


# ─── Main kernel_function ─────────────────────────────────────────────────────

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
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, 224, 224, 3)

    # conv1: 7x7, stride=2, pad=3, 64 out
    x_nhwc = triton_conv2d(x_nhwc, p_conv1_weight, p_conv1_bias, (2, 2), (3, 3), apply_relu=True)
    # maxpool1: 3x3, stride=2, pad=1
    x_nhwc = triton_maxpool2d(x_nhwc, 3, 2, 1)

    # conv2: 1x1, stride=1, pad=0, 64 out
    x_nhwc = triton_conv2d(x_nhwc, p_conv2_weight, p_conv2_bias, (1, 1), (0, 0), apply_relu=True)
    # conv3: 3x3, stride=1, pad=1, 192 out
    x_nhwc = triton_conv2d(x_nhwc, p_conv3_weight, p_conv3_bias, (1, 1), (1, 1), apply_relu=True)
    # maxpool2: 3x3, stride=2, pad=1
    x_nhwc = triton_maxpool2d(x_nhwc, 3, 2, 1)

    # inception3a
    x_nhwc = inception_module(x_nhwc,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,