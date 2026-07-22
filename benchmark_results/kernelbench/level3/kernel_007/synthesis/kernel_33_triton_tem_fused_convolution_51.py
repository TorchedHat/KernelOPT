import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Fused bias + ReLU pointwise kernel
# ─────────────────────────────────────────────────────────────────────────────
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
def bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Fused bias (no relu) pointwise kernel
# ─────────────────────────────────────────────────────────────────────────────
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
def bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    tl.store(out_ptr + offsets, out, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Implicit GEMM Conv2d kernel (NHWC layout internally)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    # dimensions
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    # kernel params
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    # strides for X (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (Co, Ci, KH, KW) -> flattened as (Co, Ci*KH*KW)
    sw_o, sw_k,
    # strides for Out (NHWC)
    so_n, so_h, so_w, so_c,
    # GEMM dims
    M, N, K,
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

    TOTAL_K = K  # IN_C * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (kh, kw, in_c)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KW
        kh_idx = tmp_k // KW

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < TOTAL_K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < BATCH
        mask_x = valid_m & valid_k & valid_h & valid_w & valid_batch

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: shape (OUT_C, IN_C*KH*KW) stored as (Co, Ci, KH, KW)
        # offs_k encodes (kh, kw, in_c) -> weight index = out_c * (Ci*KH*KW) + kh*KW*Ci + kw*Ci + in_c
        w_k_idx = kh_idx[:, None] * (KW * IN_C) + kw_idx[:, None] * IN_C + in_c_idx[:, None]
        valid_n = offs_n[None, :] < N
        valid_wk = offs_k[:, None] < TOTAL_K
        mask_w = valid_wk & valid_n

        w_ptrs = W_ptr + offs_n[None, :] * sw_o + w_k_idx * sw_k
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output (NHWC)
    valid_out_m = offs_m[:, None] < M
    valid_out_n = offs_n[None, :] < N
    mask_out = valid_out_m & valid_out_n

    out_ptrs = (Out_ptr
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# MaxPool2d kernel
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Out_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    # strides (NCHW)
    sx_n, sx_c, sx_h, sx_w,
    so_n, so_c, so_h, so_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements

    # Decode output index (NCHW)
    out_w_idx = offs % OUT_W
    tmp = offs // OUT_W
    out_h_idx = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c_idx = tmp2 % C
    n_idx = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h_idx * STRIDE_H + kh - PAD_H
            iw = out_w_idx * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + n_idx * sx_n + c_idx * sx_c + ih * sx_h + iw * sx_w
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.where(val > max_val, val, max_val)

    out_ptrs = Out_ptr + n_idx * so_n + c_idx * so_c + out_h_idx * so_h + out_w_idx * so_w
    tl.store(out_ptrs, max_val, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Average Pool 2d (global average pool to 1x1)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 8, 'BLOCK_HW': 256}, num_warps=8),
    ],
    key=['C', 'HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    X_ptr, Out_ptr,
    BATCH, C, HW,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    # One program per (batch, channel_block)
    pid = tl.program_id(0)
    n_c_blocks = tl.cdiv(C, BLOCK_C)
    batch_idx = pid // n_c_blocks
    c_block = pid % n_c_blocks

    offs_c = c_block * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for hw_start in range(0, HW, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < HW

        # X is NCHW: X[n, c, h, w] = X[n*C*HW + c*HW + hw]
        x_ptrs = X_ptr + batch_idx * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
        mask = mask_c[:, None] & mask_hw[None, :]
        vals = tl.load(x_ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=1)

    avg = acc / HW
    out_ptrs = Out_ptr + batch_idx * C + offs_c
    tl.store(out_ptrs, avg, mask=mask_c)


# ─────────────────────────────────────────────────────────────────────────────
# Linear (GEMM) kernel: Out[M, N] = X[M, K] @ W[N, K]^T + bias[N]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, Bias_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
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

        x_tile = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                         mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                         mask=mask_w, other=0.0)
        acc = tl.dot(x_tile, w_tile.T, acc, allow_tf32=False)

    offs_m_out = offs_m
    offs_n_out = offs_n
    mask_out = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    bias = tl.load(Bias_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = Out_ptr + offs_m_out[:, None] * stride_om + offs_n_out[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run conv2d via implicit GEMM
# ─────────────────────────────────────────────────────────────────────────────
def triton_conv2d(x_nhwc, weight, bias, stride, padding, apply_relu=True):
    """
    returns: (N, OH, OW, Co) contiguous
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = weight.shape
    STRIDE_H, STRIDE_W = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    PAD_H, PAD_W = padding if isinstance(padding, (tuple, list)) else (padding, padding)

    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1

    M = N * OUT_H * OUT_W
    K_dim = IN_C * KH * KW
    N_dim = OUT_C

    out = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    # Weight strides: (Co, Ci, KH, KW) -> treat as (Co, Ci*KH*KW)
    # sw_o = Ci*KH*KW, sw_k = 1
    sw_o = IN_C * KH * KW
    sw_k = 1

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_dim, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, weight, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
        # strides for X (NHWC): n, h, w, c
        IN_H * IN_W * IN_C, IN_W * IN_C, IN_C, 1,
        # strides for W
        sw_o, sw_k,
        # strides for Out (NHWC): n, h, w, c
        OUT_H * OUT_W * OUT_C, OUT_W * OUT_C, OUT_C, 1,
        M, N_dim, K_dim,
    )

    # Add bias + optional relu
    n_elements = N * OUT_H * OUT_W * OUT_C
    if apply_relu:
        bias_relu_kernel[lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)](
        )
    else:
        bias_kernel[lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)](
        )

    return out


def triton_maxpool2d(x_nchw, kernel_size, stride, padding):
    """x_nchw: (N, C, H, W) -> (N, C, OH, OW)"""
    N, C, IN_H, IN_W = x_nchw.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding

    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1

    out = torch.empty((N, C, OUT_H, OUT_W), device=x_nchw.device, dtype=torch.float32)
    n_elements = N * C * OUT_H * OUT_W

    # strides for x (NCHW)
    sx_n = C * IN_H * IN_W
    sx_c = IN_H * IN_W
    sx_h = IN_W
    sx_w = 1
    so_n = C * OUT_H * OUT_W
    so_c = OUT_H * OUT_W
    so_h = OUT_W
    so_w = 1

    maxpool2d_kernel[lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)](
        x_nchw, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        sx_n, sx_c, sx_h, sx_w,
        so_n, so_c, so_h, so_w,
        n_elements,
    )
    return out


def conv_block(x_nhwc, weight, bias, stride=1, padding=0, apply_relu=True):
    """Wrapper: x in NHWC, returns NHWC"""
    return triton_conv2d(x_nhwc, weight, bias, stride, padding, apply_relu)


def nchw_to_nhwc(x):
    return x.permute(0, 2, 3, 1).contiguous()

def nhwc_to_nchw(x):
    return x.permute(0, 3, 1, 2).contiguous()


def inception_module(x_nhwc,
                     w_1x1, b_1x1,
                     w_r3, b_r3, w_3x3, b_3x3,
                     w_r5, b_r5, w_5x5, b_5x5,
                     w_pool, b_pool,
    """
    """
    # Branch 1: 1x1 conv
    b1 = conv_block(x_nhwc, w_1x1, b_1x1, stride=1, padding=0, apply_relu=True)

    # Branch 2: 1x1 reduce -> 3x3
    b2 = conv_block(x_nhwc, w_r3, b_r3, stride=1, padding=0, apply_relu=True)
    b2 = conv_block(b2, w_3x3, b_3x3, stride=1, padding=1, apply_relu=True)

    # Branch 3: 1x1 reduce -> 5x5
    b3 = conv_block(x_nhwc, w_r5, b_r5, stride=1, padding=0, apply_relu=True)
    b3 = conv_block(b3, w_5x5, b_5x5, stride=1, padding=2, apply_relu=True)

    # Branch 4: maxpool -> 1x1
    # Need NCHW for maxpool
    x_nchw = nhwc_to_nchw(x_nhwc)
    mp = triton_maxpool2d(x_nchw, kernel_size=3, stride=1, padding=1)
    mp_nhwc = nchw_to_nhwc(mp)
    b4 = conv_block(mp_nhwc, w_pool, b_pool, stride=1, padding=0, apply_relu=True)

    # Cat along channel dim (NHWC -> dim=3)
    out = torch.cat([b1, b2, b3, b4], dim=3)
    return out


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