import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused BatchNorm + ReLU kernel (pointwise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


# ============================================================
# Fused BatchNorm + Add + ReLU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_add_relu_kernel(
    x_ptr, residual_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b + res
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def bn_add_relu(x, residual, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_relu_kernel[grid](
        x, residual, out, mean, var, weight, bias,
    )
    return out


# ============================================================
# Channel Shuffle kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    groups, channels_per_group,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Decode: offsets = n * C*H*W + c * H*W + h * W + w
    HW = H * W
    CHW = C * HW
    
    n = offsets // CHW
    rem = offsets % CHW
    c = rem // HW
    rem2 = rem % HW
    h = rem2 // W
    w = rem2 % W
    
    # Channel shuffle: c -> (c % channels_per_group) * groups + (c // channels_per_group)
    g = c // channels_per_group
    k = c % channels_per_group
    new_c = k * groups + g
    
    src_offset = n * CHW + new_c * HW + h * W + w
    src_mask = (src_offset >= 0) & (src_offset < N * CHW) & mask
    
    val = tl.load(x_ptr + src_offset, mask=src_mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def channel_shuffle(x, groups):
    N, C, H, W = x.shape
    channels_per_group = C // groups
    numel = N * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
    )
    return out


# ============================================================
# Max Pool 2D kernel (3x3, stride=2, padding=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    OHW = OUT_H * OUT_W
    COHW = C * OHW
    
    n = offsets // COHW
    rem = offsets % COHW
    c = rem // OHW
    rem2 = rem % OHW
    oh = rem2 // OUT_W
    ow = rem2 % OUT_W
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & mask
            src = n * C * IN_H * IN_W + c * IN_H * IN_W + ih * IN_W + iw
            val = tl.load(x_ptr + src, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


def max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    OH = (H + 2 * padding - kernel_size) // stride + 1
    OW = (W + 2 * padding - kernel_size) // stride + 1
    numel = N * C * OH * OW
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out, N, C, H, W, OH, OW,
        kernel_size, kernel_size, stride, stride, padding, padding,
    )
    return out


# ============================================================
# Adaptive Average Pool 2D -> (1,1) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['NC', 'HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr, out_ptr,
    NC, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    nc_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = nc_offsets < NC
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(HW):
        val = tl.load(x_ptr + nc_offsets * HW + i, mask=mask, other=0.0).to(tl.float32)
        acc += val
    
    out = acc / HW
    tl.store(out_ptr + nc_offsets, out, mask=mask)


def adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    NC = N * C
    HW = H * W
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_SIZE']),)
    adaptive_avg_pool_kernel[grid](
    )
    return out


# ============================================================
# Linear kernel (GEMM: M x K @ K x N + bias)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def linear(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


# ============================================================
# General Conv2d via implicit GEMM
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    SH, SW, PH, PW,
    GROUP_IN_C, GROUP_OUT_C, GROUPS,
    M, N_out, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wout, stride_win, stride_wkh, stride_wkw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H
    
    # Decode N -> output channel
    out_c = offs_n  # global output channel
    group = out_c // GROUP_OUT_C
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c_local, kh, kw)
        in_c_local = offs_k % GROUP_IN_C
        kw = (offs_k // GROUP_IN_C) % KW
        kh = offs_k // (GROUP_IN_C * KW)
        
        # Input coordinates
        ih = out_h[:, None] * SH + kh[None, :] - PH
        iw = out_w[:, None] * SW + kw[None, :] - PW
        
        # Global input channel
        in_c = group[:, None] * GROUP_IN_C + in_c_local[None, :]  # This is wrong for grouped conv
        # Actually for grouped conv, group is per output channel
        # We need to handle this differently
        
        valid_m = (offs_m < M)
        valid_k = (offs_k < K)
        valid = (
            (batch[:, None] < N) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            valid_k[None, :]
        )
        
        # For simplicity, handle group=1 or depthwise separately
        # Here we assume group info is encoded in out_c
        # in_c = group_of_out_c * GROUP_IN_C + in_c_local
        # group_of_out_c = out_c // GROUP_OUT_C
        # But out_c varies per column, so we need per-column group
        group_col = offs_n // GROUP_OUT_C  # shape [BLOCK_N]
        in_c_global = group_col[None, :] * GROUP_IN_C + in_c_local[:, None]  # [BLOCK_K, BLOCK_N] - wrong shape
        
        # Let's use a simpler approach: compute x and w separately
        # x: [BLOCK_M, BLOCK_K]
        x_ptrs = (X_ptr + 
                  batch[:, None] * stride_xn + 
                  (group[None, :] * GROUP_IN_C + in_c_local[None, :]) * stride_xc +  # This mixes group per output col
        # This is getting complex for grouped conv. Let's use group=0 for now and handle groups in wrapper
        
        # Simplified: assume single group or handle in wrapper by splitting
        x_ptrs2 = (X_ptr + 
                   batch[:, None] * stride_xn + 
                   in_c_local[None, :] * stride_xc +
        
        x = tl.load(x_ptrs2, mask=valid & (offs_m[:, None] < M), other=0.0)
        
        # w: [BLOCK_K, BLOCK_N]
        w_ptrs = (W_ptr + 
                  offs_n[None, :] * stride_wout + 
                  in_c_local[:, None] * stride_win + 
                  kh[:, None] * stride_wkh + 
                  kw[:, None] * stride_wkw)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N_out)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_out)
    # Store: output is (N, OUT_C, OUT_H, OUT_W) with stride (OUT_C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1)
    out_ptrs = (Out_ptr + 
                batch[:, None] * (OUT_C * OUT_H * OUT_W) + 
                out_c[None, :] * (OUT_H * OUT_W) + 
                out_h[:, None] * OUT_W + 
                out_w[:, None])
    tl.store(out_ptrs, acc, mask=out_mask)