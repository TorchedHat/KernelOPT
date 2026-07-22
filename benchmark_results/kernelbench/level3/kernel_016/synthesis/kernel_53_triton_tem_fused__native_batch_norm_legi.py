import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math
import argparse

# Fused BatchNorm (inference) + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, num_channels,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index: layout is (N, C, H, W) contiguous
    # channel = (offset / (H*W)) % C  -- but we pass spatial_size
    # Actually numel = N*C*H*W, channel index = (offset // spatial) % C
    # We'll compute it as offset // (numel // (N*C)) % C
    # Simpler: channel = (offset % (num_channels * spatial)) // spatial
    # We pass num_channels and compute channel from offset
    # offset layout: N*C*H*W, channel = (offset // (numel//(N*C_total))) % num_channels
    # Since numel = N*C*H*W, spatial = numel // (N * num_channels) -- but N is not passed
    # Let's just pass spatial_size separately
    # Actually let's use: channel_idx = (offset // spatial_size) % num_channels
    # We'll restructure to pass spatial_size
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channel index
    c = (offsets // (numel // num_channels)) % num_channels
    # Wait, numel // num_channels = N * H * W (spatial per channel across batch)
    # offset = n * C * H * W + c * H * W + h * W + w
    # c = (offset // (H*W)) % C
    # numel // num_channels = N * H * W
    # (offset // (numel // num_channels)) = (offset // (N*H*W)) -- that's wrong
    # Let's think again: numel = N*C*H*W
    # spatial_size = H*W, we need to pass it
    # c = (offset // spatial_size) % num_channels
    # But we don't have spatial_size here...
    # Let's restructure the kernel to take spatial_size
    pass

# Better version with spatial_size
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_relu_kernel_v2(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, spatial_size, num_channels,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = (offsets // spatial_size) % num_channels
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    out = tl.maximum(out, tl.zeros_like(out))
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    spatial_size = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel_v2[grid](
        x, running_mean, running_var, weight, bias,
        out,
        numel, spatial_size, C,
        eps,
    )
    return out


# Fused BatchNorm (inference) kernel (no ReLU)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, spatial_size, num_channels,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    c = (offsets // spatial_size) % num_channels
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    weight = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def triton_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    spatial_size = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        numel, spatial_size, C,
        eps,
    )
    return out


# General Conv2d using implicit GEMM in Triton
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N_out, K,
    s_xn, s_xc, s_xh, s_xw,
    s_wco, s_wci, s_wkh, s_wkw,
    s_yn, s_yc, s_yh, s_yw,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)
        
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = (batch[:, None] < N) & (offs_m[:, None] < M)
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_x = valid_m & valid_k & valid_ih & valid_iw
        
        x_ptrs = X_ptr + (
            batch[:, None] * s_xn +
            in_c[None, :] * s_xc +
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        # Weight: (OUT_C, IN_C, KH, KW) -> index [n, ci, kh, kw]
        # offs_n: output channel
        # offs_k: (kh, kw, in_c)
        w_ptrs = W_ptr + (
            offs_n[None, :] * s_wco +
            in_c[:, None] * s_wci +
            kh[:, None] * s_wkh +
            kw[:, None] * s_wkw
        )
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)
    
    # Store output
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + (
        batch[:, None] * s_yn +
        offs_n[None, :] * s_yc +
        out_h[:, None] * s_yh +
        out_w[:, None] * s_yw
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def triton_conv2d(x, w, stride=(1,1), padding=(0,0)):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    SH, SW = stride
    PH, PW = padding
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1
    
    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    M = N * OUT_H * OUT_W
    N_out = OUT_C
    K = IN_C * KH * KW
    
    s_xn, s_xc, s_xh, s_xw = x.stride()
    s_wco, s_wci, s_wkh, s_wkw = w.stride()
    s_yn, s_yc, s_yh, s_yw = y.stride()
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_out, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x, w, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        M, N_out, K,
        s_xn, s_xc, s_xh, s_xw,
        s_wco, s_wci, s_wkh, s_wkw,
        s_yn, s_yc, s_yh, s_yw,
    )
    return y


# Max pool 2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    numel,
    s_xn, s_xc, s_xh, s_xw,
    s_yn, s_yc, s_yh, s_yw,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    # Decode: offs -> (n, c, oh, ow)
    ow = offs % OUT_W
    oh = (offs // OUT_W) % OUT_H
    c = (offs // (OUT_W * OUT_H)) % C
    n = offs // (OUT_W * OUT_H * C)
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = x_ptr + n * s_xn + c * s_xc + ih * s_xh + iw * s_xw
            val = tl.load(x_ptrs, mask=valid, other=float('-inf')).to(tl.float32)
            max_val = tl.maximum(max_val, val)
    
    out_ptrs = out_ptr + n * s_yn + c * s_yc + oh * s_yh + ow * s_yw
    tl.store(out_ptrs, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1
    
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = N * C * OUT_H * OUT_W
    
    s_xn, s_xc, s_xh, s_xw = x.stride()
    s_yn, s_yc, s_yh, s_yw = out.stride()
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        numel,
        s_xn, s_xc, s_xh, s_xw,
        s_yn, s_yc, s_yh, s_yw,
    )
    return out


# Avg pool 2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    numel,
    s_xn, s_xc, s_xh, s_xw,
    s_yn, s_yc, s_yh, s_yw,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    ow = offs % OUT_W
    oh = (offs // OUT_W) % OUT_H
    c = (offs // (OUT_W * OUT_H)) % C
    n = offs // (OUT_W * OUT_H * C)
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = x_ptr + n * s_xn + c * s_xc + ih * s_xh + iw * s_xw
            val = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.float32)
            acc = acc + tl.where(valid, val, tl.zeros_like(val))
            count = count + tl.where(valid, tl.ones_like(val), tl.zeros_like(val))
    
    result = acc / tl.maximum(count, tl.ones_like(count))
    out_ptrs = out_ptr + n * s_yn + c * s_yc + oh * s_yh + ow * s_yw
    tl.store(out_ptrs, result, mask=mask)


def triton_avg_pool2d(x, kernel_size=2, stride=2, padding=0):
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1
    
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = N * C * OUT_H * OUT_W
    
    s_xn, s_xc, s_xh, s_xw = x.stride()
    s_yn, s_yc, s_yh, s_yw = out.stride()
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    avg_pool2d_kernel[grid](
        x, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        numel,
        s_xn, s_xc, s_xh, s_xw,
        s_yn, s_yc, s_yh, s_yw,
    )
    return out


# Adaptive avg pool 2d (global) kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['NC'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W,
    NC,
    s_xn, s_xc, s_xh, s_xw,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NC
    
    c = offs % C
    n = offs // C
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    total = IN_H * IN_W
    
    for h in range(IN_H):
        for w in range(IN_W):
            x_ptrs = x_ptr + n * s_xn + c * s_xc + h * s_xh + w * s_xw
            val = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
            acc = acc + val
    
    result = acc / total
    tl.store(out_ptr + offs, result, mask=mask)


def triton_adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    NC = N * C
    s_xn, s_xc, s_xh, s_xw = x.stride()
    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        NC,
        s_xn, s_xc, s_xh, s_xw,
    )
    return out


# Linear kernel (GEMM: M x K @ K x N)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    s_xm, s_xk,
    s_wn, s_wk,
    s_om, s_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + offs_m[:, None] * s_xm + offs_k[None, :] * s_xk
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # w is (N, K), we want (K, N) for the dot
        w_ptrs = w_ptr + offs_n[None, :] * s_wn + offs_k[:, None] * s_wk
        mask_w = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)
    
    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * s_om + offs_n[None, :] * s_on
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_linear(x, w, b=None):
    # x: (M, K), w: (N, K), b: (N,)
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (