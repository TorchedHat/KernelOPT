import torch
import triton
import triton.language as tl
import math

# Due to the enormous complexity of this full DenseNet (200+ operations), 
# I'll implement the core Triton kernels and chain them.


import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # offs indexes into flattened (batch, C, H, W) but stored as (batch, C, H, W)
    # We need channel index to get mean/var/weight/bias
    # N = batch * C * H * W, layout is NCHW
    # channel = (offs // HW) % C
    hw_idx = offs % HW
    c_idx = (offs // HW) % C
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offs, out, mask=mask)

# ... this approach won't work for the full network easily

# Let me take a different approach - implement the full forward pass
# using a series of Triton kernels for each operation type

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),  
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW, N,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    c_idx = (offs // HW) % C
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offs, out, mask=mask)


import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton Kernels
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    c_idx = (offs // HW) % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    eps = 1e-5
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def fused_bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    c_idx = (offs // HW) % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    eps = 1e-5
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    tl.store(out_ptr + offs, out, mask=mask)


# Implicit GEMM convolution kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N_batch, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    M, N, K,
    # strides for x: (N_batch, IN_C, IN_H, IN_W)
    sx_n, sx_c, sx_h, sx_w,
    # strides for w: (OUT_C, IN_C, KH, KW)
    sw_oc, sw_ic, sw_kh, sw_kw,
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
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        ic_idx = tmp_k // KH
        
        # Input coordinates
        ih = out_h_idx[:, None] * stride_h + kh_idx[None, :] - pad_h
        iw = out_w_idx[:, None] * stride_w + kw_idx[None, :] - pad_w
        
        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_ic = ic_idx[None, :] < IN_C
        
        x_mask = valid_m & valid_k & valid_ih & valid_iw & valid_ic
        x_ptrs = x_ptr + batch_idx[:, None] * sx_n + ic_idx[None, :] * sx_c + ih * sx_h + iw * sx_w
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        valid_n = offs_n[None, :] < N
        w_mask = valid_k & valid_n
        w_ptrs = w_ptr + offs_n[None, :] * sw_oc + ic_idx[:, None] * sw_ic + kh_idx[:, None] * sw_kh + kw_idx[:, None] * sw_kw
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store output
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Output layout: (N_batch, OUT_C, OUT_H, OUT_W) -> index
    out_idx = batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h_idx[:, None] * OUT_W + out_w_idx[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
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
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        x_tile = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :], mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    N_batch, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, stride_h, stride_w, pad_h, pad_w,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Decode: N = N_batch * C * OUT_H * OUT_W
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * stride_h + kh - pad_h
            iw = ow * stride_w + kw - pad_w
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + offs, max_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def avg_pool2d_kernel(
    x_ptr, out_ptr,
    N_batch, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, stride_h, stride_w, pad_h, pad_w,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C
    
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * stride_h + kh - pad_h
            iw = ow * stride_w + kw - pad_w
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            sum_val = sum_val + tl.where(valid, val, 0.0)
            count = count + tl.where(valid, 1.0, 0.0)
    
    out = sum_val / count
    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N_batch, C, IN_H, IN_W,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Output is (N_batch, C, 1, 1) - global average pooling
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # offs indexes (batch, channel)
    c = offs % C
    n = offs // C
    
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    total = IN_H * IN_W
    
    for h in range(IN_H):
        for w in range(IN_W):
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + h * IN_W + w
            val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
            sum_val = sum_val + val
    
    out = sum_val / total
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Helper functions
# ============================================================

def triton_conv2d(x, w, stride_h=1, stride_w=1, pad_h=0, pad_w=0):
    N_batch, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1
    
    out = torch.empty(N_batch, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    M = N_batch * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KH * KW
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x, w, out,
        N_batch, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
    )
    return out


def triton_bn_relu(x, mean, var, weight, bias):
    N = x.numel()
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](x, mean, var, weight, bias, out, C, HW, N)
    return out


def triton_bn(x, mean, var, weight, bias):
    N = x.numel()
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    fused_bn_kernel[grid](x, mean, var, weight, bias, out, C, HW, N)
    return out


def triton_max_pool2d(x, kh=3, kw=3, stride_h=2, stride_w=2, pad_h=1, pad_w=1):
    N_batch, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + 2 * pad_h - kh) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - kw) // stride_w + 1
    out = torch.empty(N_batch, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    N = N_batch * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](x, out, N_batch, C, IN_H, IN_W, OUT_H, OUT_W, kh, kw, stride_h, stride_w, pad_h, pad_w, N)
    return out


def triton_avg_pool2d(x, kh=2, kw=2, stride_h=2, stride_w=2, pad_h=0, pad_w=0):
    N_batch, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + 2 * pad_h - kh) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - kw) // stride_w + 1
    out = torch.empty(N_batch, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    N = N_batch * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    avg_pool2d_kernel[grid](x, out, N_batch, C, IN_H, IN_W, OUT_H, OUT_W, kh, kw, stride_h, stride_w, pad_h, pad_w, N)
    return out


def triton_adaptive_avg_pool2d(x):
    N_batch, C, IN_H, IN_W = x.shape
    out = torch.empty(N_batch, C, 1, 1, device=x.device, dtype=x.dtype)
    N = N_batch * C
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](x, out, N_batch, C, IN_H, IN_W, N)
    return out


def triton_linear(x, w, b):
    # x: (M, K), w: (N, K), b: (N,)
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    # w needs to be transposed: w_t = w.T -> (K, N)
    w_t = w.t().contiguous()
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, w_t, b, out, M, N, K)
    return out


def kernel_function(
    p_features_0_weight, p_features_1_weight, p_features_1_bias,
    p_dense_blocks_0_layers_0_0_weight, p_dense_blocks_0_layers_0_0_bias,
    p_dense_blocks_0_layers_0_2_weight,
    p_dense_blocks_0_layers_1_0_weight, p_dense_blocks_0_layers_1_0_bias,
    p_dense_blocks_0_layers_1_2_weight,
    p_dense_blocks_0_layers_2_0_weight, p_dense_blocks_0_layers_2_0_bias,
    p_dense_blocks_0_layers_2_2_weight,
    p_dense_blocks_0_layers_3_0_weight, p_dense_blocks_0_layers_3_0_bias,
    p_dense_blocks_0_layers_3_2_weight,
    p_dense_blocks_0_layers_4_0_weight, p_dense_blocks_0_layers_4_0_bias,
    p_dense_blocks_0_layers_4_2_weight,
    p_dense_blocks_0_layers_5_0_weight, p_dense_blocks_0_layers_5_0_bias,
    p_dense_blocks_0_layers_5_2_weight,
    p_transition_layers_0_transition_0_weight, p_transition_layers_0_transition_0_bias,
    p_transition_layers_0_transition_2_weight,
    p_dense_blocks_1_layers_0_0_weight, p_dense_blocks_1_layers_0_0_bias,
    p_dense_blocks_1_layers_0_2_weight,
    p_dense_blocks_1_layers_1_0_weight, p_dense_blocks_1_layers_1_0_bias,
    p_dense_blocks_1_layers_1_2_weight,