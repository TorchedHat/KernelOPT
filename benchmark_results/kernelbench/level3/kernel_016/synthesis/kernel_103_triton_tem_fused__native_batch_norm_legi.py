import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# ─── Fused BatchNorm + ReLU kernel ───────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    # channel index
    c = (offs // HW) % C
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var  = tl.load(var_ptr  + c, mask=mask, other=0.0)
    w    = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b    = tl.load(bias_ptr   + c, mask=mask, other=0.0)
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c = (offs // HW) % C
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var  = tl.load(var_ptr  + c, mask=mask, other=0.0)
    w    = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b    = tl.load(bias_ptr   + c, mask=mask, other=0.0)
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    tl.store(out_ptr + offs, out, mask=mask)


# ─── Implicit GEMM Conv2d kernel ─────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    # strides for X (NHWC layout)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (Co, Ci, KH, KW) -> stored as (Co, KH, KW, Ci) for NHWC
    sw_o, sw_kh, sw_kw, sw_i,
    M, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # decode M -> (n, oh, ow)
    ow = offs_m % W_out
    oh = (offs_m // W_out) % H_out
    n  = offs_m // (W_out * H_out)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # decode K -> (ci, kh, kw)
        ci  = offs_k % C_in
        kh  = (offs_k // C_in) % KH
        kw  = offs_k // (C_in * KH)

        ih = oh[:, None] * stride_h + kh[None, :] - pad_h
        iw = ow[:, None] * stride_w + kw[None, :] - pad_w

        valid_m = (offs_m < M)[:, None]
        valid_k = (offs_k < K)[None, :]
        valid_ih = (ih >= 0) & (ih < H_in)
        valid_iw = (iw >= 0) & (iw < W_in)
        mask_x = valid_m & valid_k & valid_ih & valid_iw

        x_ptrs = X_ptr + n[:, None] * sx_n + ih * sx_h + iw * sx_w + ci[None, :] * sx_c
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # W layout: (Co, KH, KW, Ci) for NHWC
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < C_out)
        w_ptrs = W_ptr + kh[:, None] * sw_kh + kw[:, None] * sw_kw + ci[:, None] * sw_i + offs_n[None, :] * sw_o
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < C_out)
    # output in NHWC: (n, oh, ow, co)
    out_ptrs = Y_ptr + n[:, None] * (H_out * W_out * C_out) + oh[:, None] * (W_out * C_out) + ow[:, None] * C_out + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# ─── Linear (GEMM) kernel ────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_b = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak, mask=mask_a, other=0.0)
        b = tl.load(B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=mask_b, other=0.0)
        acc = tl.dot(a, b, acc=acc)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_c)


# ─── Max Pool 2D kernel ───────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def maxpool2d_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    KH, KW, stride_h, stride_w, pad_h, pad_w,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    # decode: offs -> (n, c, oh, ow) in NHWC
    ow = offs % W_out
    oh = (offs // W_out) % H_out
    c  = (offs // (W_out * H_out)) % C
    n  = offs // (W_out * H_out * C)

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * stride_h + kh - pad_h
            iw = ow * stride_w + kw - pad_w
            valid = mask & (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)
            x_idx = n * (H_in * W_in * C) + ih * (W_in * C) + iw * C + c
            val = tl.load(x_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    tl.store(out_ptr + offs, max_val, mask=mask)


# ─── Avg Pool 2D kernel ───────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def avgpool2d_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    KH, KW, stride_h, stride_w, pad_h, pad_w,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    ow = offs % W_out
    oh = (offs // W_out) % H_out
    c  = (offs // (W_out * H_out)) % C
    n  = offs // (W_out * H_out * C)

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * stride_h + kh - pad_h
            iw = ow * stride_w + kw - pad_w
            valid = mask & (ih >= 0) & (ih < H_in) & (iw >= 0) & (iw < W_in)
            x_idx = n * (H_in * W_in * C) + ih * (W_in * C) + iw * C + c
            val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            acc = acc + tl.where(valid, val, 0.0)
            count = count + tl.where(valid, 1.0, 0.0)
    out = acc / count
    tl.store(out_ptr + offs, out, mask=mask)


# ─── Adaptive Avg Pool 2D (global) kernel ────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 32},  num_warps=2),
    ],
    key=['NC'],
)
@triton.jit
def adaptive_avgpool_global_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    NC,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (n, c) pair
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NC
    n = offs // C
    c = offs % C

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for hw in range(HW):
        x_idx = n * (HW * C) + hw * C + c
        val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
        acc = acc + val
    out = acc / HW
    tl.store(out_ptr + offs, out, mask=mask)


# ─── Helper functions ─────────────────────────────────────────────────────────

def triton_bn_relu(x_nhwc, mean, var, weight, bias, eps=1e-5):
    """x_nhwc: (N, H, W, C) contiguous"""
    N, H, W, C = x_nhwc.shape
    out = torch.empty_like(x_nhwc)
    numel = x_nhwc.numel()
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x_nhwc, out,
        mean, var, weight, bias,
        numel, C, HW, eps,
    )
    return out


def triton_bn(x_nhwc, mean, var, weight, bias, eps=1e-5):
    N, H, W, C = x_nhwc.shape
    out = torch.empty_like(x_nhwc)
    numel = x_nhwc.numel()
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x_nhwc, out,
        mean, var, weight, bias,
        numel, C, HW, eps,
    )
    return out


def triton_conv2d(x_nhwc, w_nchw, stride=1, padding=0):
    """
    returns: (N, H_out, W_out, C_out) contiguous
    """
    N, H_in, W_in, C_in = x_nhwc.shape
    C_out, _, KH, KW = w_nchw.shape

    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    # Convert weight to (C_out, KH, KW, C_in) layout
    w_nhwc = w_nchw.permute(0, 2, 3, 1).contiguous()  # (C_out, KH, KW, C_in)

    out = torch.empty((N, H_out, W_out, C_out), device=x_nhwc.device, dtype=x_nhwc.dtype)

    M = N * H_out * W_out
    K = C_in * KH * KW

    # strides for x_nhwc
    sx_n = H_in * W_in * C_in
    sx_h = W_in * C_in
    sx_w = C_in
    sx_c = 1

    # strides for w_nhwc (C_out, KH, KW, C_in)
    sw_o  = KH * KW * C_in
    sw_kh = KW * C_in
    sw_kw = C_in
    sw_i  = 1

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_nhwc, out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        sx_n, sx_h, sx_w, sx_c,
        sw_o, sw_kh, sw_kw, sw_i,
        M, K,
    )
    return out


def triton_maxpool2d(x_nhwc, kernel_size=3, stride=2, padding=1):
    N, H_in, W_in, C = x_nhwc.shape
    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    out = torch.empty((N, H_out, W_out, C), device=x_nhwc.device, dtype=x_nhwc.dtype)
    numel = out.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x_nhwc, out,
        N, C, H_in, W_in, H_out, W_out,
        KH, KW, stride_h, stride_w, pad_h, pad_w,
        numel,
    )
    return out


def triton_avgpool2d(x_nhwc, kernel_size=2, stride=2, padding=0):
    N, H_in, W_in, C = x_nhwc.shape
    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    out = torch.empty((N, H_out, W_out, C), device=x_nhwc.device, dtype=x_nhwc.dtype)
    numel = out.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    avgpool2d_kernel[grid](
        x_nhwc, out,
        N, C, H_in, W_in, H_out, W_out,
        KH, KW, stride_h, stride_w, pad_h, pad_w,
        numel,
    )
    return out


def triton_adaptive_avgpool_global(x_nhwc):
    """Global average pooling: (N, H, W, C) -> (N, 1, 1, C)"""
    N, H, W, C = x_nhwc.shape
    HW = H * W
    NC = N * C
    out = torch.empty((N, C), device=x_nhwc.device, dtype=x_nhwc.dtype)
    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_SIZE']),)
    adaptive_avgpool_global_kernel[grid](
        x_nhwc, out,
        N, C, HW, NC,
    )
    return out.view(N, 1, 1, C)


def triton_linear(x, weight, bias):
    """x: (M, K), weight: (N, K), bias: (N,) -> (M, N)"""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight is (N, K), so stride_bk=stride(1), stride_bn=stride(0)
    )
    return out


# ─── Main kernel_function ─────────────────────────────────────────────────────

def kernel_function(
    p_features_0_weight,
    p_features_1_weight,
    p_features_1_bias,
    p_dense_blocks_0_layers_0_0_weight,
    p_dense_blocks_0_layers_0_0_bias,
    p_dense_blocks_0_layers_0_2_weight,
    p_dense_blocks_0_layers_1_0_weight,
    p_dense_blocks_0_layers_1_0_bias,
    p_dense_blocks_0_layers_1_2_weight,
    p_dense_blocks_0_layers_2_0_weight,
    p_dense_blocks_0_layers_2_0_bias,
    p_dense_blocks_0_layers_2_2_weight,
    p_dense_blocks_0_layers_3_0_weight,
    p_dense_blocks_0_layers_3_0_bias,
    p_dense_blocks_0_layers_3_2_weight,
    p_dense_blocks_0_layers_4_0_weight,
    p_dense_blocks_0_layers_4_0_bias,
    p_dense_blocks_0_layers_4_2_weight,
    p_dense_blocks_0_layers_5_0_weight,
    p_dense_blocks_0_layers_5_0_bias,
    p_dense_blocks_0_layers_5_2_weight,
    p_transition_layers_0_transition_0_weight,
    p_transition_layers_0_transition_0_bias,
    p_transition_layers_0_transition_2_weight,
    p_dense_blocks_1_layers_0_0_weight,
    p_dense_blocks_1_layers_0_0_bias,
    p_dense_blocks_1_layers_0_2_weight,
    p_dense_blocks_1_layers_1_0_weight,
    p_dense_blocks_1_layers_1_0_bias,
    p_dense_blocks_1_layers_1_2_weight,
    p_dense_blocks_1_layers_2_0_weight,
    p_dense_blocks_1_layers_2_0_bias,
    p_dense_blocks_1_layers_2_2_weight,
    p_dense_blocks_1_layers_3_0_weight,
    p_dense_blocks_1_layers_3_0_bias,
    p_dense_blocks_1_layers_3_2_weight,
    p_dense_blocks_1_layers_4_0_weight,
    p_dense_blocks_1_layers_4_0_bias,
    p_dense_blocks_1_layers_4_2_weight,
    p_dense_blocks_1_layers_5_0_weight,
    p_dense_blocks_1_layers_5_0_bias,
    p_dense_blocks_1_layers_5_2_weight,
    p_dense_blocks_1_layers_6_0_weight,
    p_dense_blocks_1_layers_6_0_bias,
    p_dense_blocks_1_layers_6_2_weight,
    p_dense_blocks_1_layers_7_0_weight,