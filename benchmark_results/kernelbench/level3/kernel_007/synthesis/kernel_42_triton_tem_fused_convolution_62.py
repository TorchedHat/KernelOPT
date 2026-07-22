import torch
import triton
import triton.language as tl
import math



import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: fused conv2d (implicit GEMM) + bias + relu
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides for X (NHWC layout)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (out_c, kh, kw, in_c) layout
    sw_oc, sw_kh, sw_kw, sw_ic,
    # strides for Y (NHWC layout)
    sy_n, sy_h, sy_w, sy_c,
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

    TOTAL_K = IN_C * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KW
        kh = tmp_k // KW

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m < M
        valid_k = offs_k < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx < (M // (OUT_H * OUT_W) + 1)

        mask_x = valid_m[:, None] & valid_k[None, :] & valid_ih & valid_iw & (batch_idx[:, None] < 1000000)

        x_ptrs = X_ptr + batch_idx[:, None] * sx_n + ih * sx_h + iw * sx_w + in_c[None, :] * sx_c
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: shape (out_c, kh, kw, in_c) -> access [offs_n, kh, kw, in_c]
        mask_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * sw_oc + kh[:, None] * sw_kh + kw[:, None] * sw_kw + in_c[:, None] * sw_ic
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    mask_n = offs_n < OUT_C
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + batch_idx[:, None] * sy_n + out_h_idx[:, None] * sy_h + out_w_idx[:, None] * sy_w + offs_n[None, :] * sy_c
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_conv2d(x_nhwc, w_nhwc, bias, stride_h, stride_w, pad_h, pad_w, apply_relu=True):
    """
    returns: (N, OH, OW, OC) contiguous
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, KH, KW, _ = w_nhwc.shape

    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1

    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW

    y = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        x_nhwc, w_nhwc, bias, y,
        M, OUT_C, K,
        IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w_nhwc.stride(0), w_nhwc.stride(1), w_nhwc.stride(2), w_nhwc.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        APPLY_RELU=apply_relu,
    )
    return y


# ============================================================
# Max Pool 2D kernel
# ============================================================

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
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    # X strides (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    # Y strides (NHWC)
    sy_n, sy_h, sy_w, sy_c,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: N_TOTAL = N * OUT_H * OUT_W * C
    c_idx = offs % C
    tmp = offs // C
    ow = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh = tmp2 % OUT_H
    n_idx = tmp2 // OUT_H

    result = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + n_idx * sx_n + ih * sx_h + iw * sx_w + c_idx * sx_c
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            result = tl.maximum(result, val)

    y_ptrs = Y_ptr + n_idx * sy_n + oh * sy_h + ow * sy_w + c_idx * sy_c
    tl.store(y_ptrs, result, mask=mask)


def triton_max_pool2d(x_nhwc, kh, kw, stride_h, stride_w, pad_h, pad_w):
    N, IN_H, IN_W, C = x_nhwc.shape
    OUT_H = (IN_H + 2 * pad_h - kh) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - kw) // stride_w + 1

    y = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=x_nhwc.dtype)
    N_TOTAL = N * OUT_H * OUT_W * C

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)

    max_pool2d_kernel[grid](
        x_nhwc, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kh, kw, stride_h, stride_w, pad_h, pad_w,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        N_TOTAL,
    )
    return y


# ============================================================
# Adaptive Average Pool 2D (global) kernel
# ============================================================

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
def adaptive_avg_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, H, W,
    HW,
    sx_n, sx_h, sx_w, sx_c,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c_block)
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

        # Load X[pid_n, h_idx, w_idx, offs_c]
        x_ptrs = X_ptr + pid_n * sx_n + h_idx[:, None] * sx_h + w_idx[:, None] * sx_w + offs_c[None, :] * sx_c
        mask = mask_hw[:, None] & mask_c[None, :]
        vals = tl.load(x_ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    acc = acc / HW

    y_ptrs = Y_ptr + pid_n * C + offs_c
    tl.store(y_ptrs, acc, mask=mask_c)


def triton_adaptive_avg_pool2d(x_nhwc):
    N, H, W, C = x_nhwc.shape
    HW = H * W
    y = torch.empty((N, C), device=x_nhwc.device, dtype=torch.float32)

    grid = lambda meta: (N, triton.cdiv(C, meta['BLOCK_C']))

    adaptive_avg_pool2d_kernel[grid](
        x_nhwc, y,
        N, C, H, W, HW,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
    )
    return y


# ============================================================
# Linear kernel (GEMM: [N, K] x [K, M] -> [N, M])
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M_dim', 'N_dim', 'K_dim'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M_dim, N_dim, K_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_dim, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M_dim) & (offs_k[None, :] < K_dim)
        mask_w = (offs_k[:, None] < K_dim) & (offs_n[None, :] < N_dim)

        x_ptrs = X_ptr + offs_m[:, None] * K_dim + offs_k[None, :]
        w_ptrs = W_ptr + offs_k[:, None] * N_dim + offs_n[None, :]

        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N_dim, other=0.0)
    acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M_dim) & (offs_n[None, :] < N_dim)
    y_ptrs = Y_ptr + offs_m[:, None] * N_dim + offs_n[None, :]
    tl.store(y_ptrs, acc, mask=mask_out)


def triton_linear(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    # output: (M, N)
    M_dim, K_dim = x.shape
    N_dim = weight.shape[0]

    # weight needs to be transposed for the kernel: (K, N)
    w_t = weight.t().contiguous()
    y = torch.empty((M_dim, N_dim), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M_dim, meta['BLOCK_M']), triton.cdiv(N_dim, meta['BLOCK_N']))

    linear_kernel[grid](
        x, w_t, bias, y,
        M_dim, N_dim, K_dim,
    )
    return y


# ============================================================
# Helper: convert weight from (OC, IC, KH, KW) to (OC, KH, KW, IC)
# ============================================================

def prepare_weight(w):
    # w: (OC, IC, KH, KW) -> (OC, KH, KW, IC)
    return w.permute(0, 2, 3, 1).contiguous()


def conv_relu(x_nhwc, w_nchw, bias, stride=1, padding=0, apply_relu=True):
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    w_nhwc = prepare_weight(w_nchw)
    return triton_conv2d(x_nhwc, w_nhwc, bias, stride[0], stride[1], padding[0], padding[1], apply_relu)


def max_pool(x_nhwc, kernel_size=3, stride=2, padding=1):
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    return triton_max_pool2d(x_nhwc, kernel_size[0], kernel_size[1], stride[0], stride[1], padding[0], padding[1])


def inception_module(x_nhwc,
                     w_1x1, b_1x1,
                     w_r3, b_r3, w_3x3, b_3x3,
                     w_r5, b_r5, w_5x5, b_5x5,
    # Branch 1: 1x1 conv
    b1 = conv_relu(x_nhwc, w_1x1, b_1x1, stride=1, padding=0)

    # Branch 2: 1x1 reduce + 3x3
    b2 = conv_relu(x_nhwc, w_r3, b_r3, stride=1, padding=0)
    b2 = conv_relu(b2, w_3x3, b_3x3, stride=1, padding=1)

    # Branch 3: 1x1 reduce + 5x5
    b3 = conv_relu(x_nhwc, w_r5, b_r5, stride=1, padding=0)
    b3 = conv_relu(b3, w_5x5, b_5x5, stride=1, padding=2)

    # Branch 4: max pool + 1x1
    b4 = max_pool(x_nhwc, kernel_size=3, stride=1, padding=1)
    b4 = conv_relu(b4, w_pool, b_pool, stride=1, padding=0)

    # Concatenate along channel dim (NHWC -> dim 3)
    return torch.cat([b1, b2, b3, b4], dim=3)


# ============================================================
# Main kernel_function
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
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # conv1: 7x7, stride=2, padding=3