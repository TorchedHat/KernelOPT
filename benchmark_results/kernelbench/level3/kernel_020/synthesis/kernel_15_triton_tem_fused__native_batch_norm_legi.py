import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused BatchNorm + HardTanh (ReLU6) pointwise kernel
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
def bn_relu6_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b
    
    if apply_relu6:
        out = tl.maximum(out, 0.0)
        out = tl.minimum(out, 6.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Fused Linear (GEMM) kernel: out = x @ W^T + bias
# x: [M, K], W: [N, K], bias: [N], out: [M, N]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
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
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Implicit GEMM Conv2d kernel (NCHW input, NCHW output)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M_total', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    GROUPS,
    M_total, TOTAL_K,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (out_c, in_c, kh, kw) -> stored as (out_c, kh, kw, in_c) for NHWC
    s_wo, s_wkh, s_wkw, s_wi,
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
    
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
    
    # Which group does each output channel belong to?
    group_idx = offs_n // GROUP_OUT_C  # [BLOCK_N]
    local_out_c = offs_n % GROUP_OUT_C  # [BLOCK_N]
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(TOTAL_K, BLOCK_K)):
        k_offs = k * BLOCK_K + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c_local, kh, kw)
        in_c_local = k_offs % GROUP_IN_C
        tmp_k = k_offs // GROUP_IN_C
        kw = tmp_k % KERNEL_W
        kh = tmp_k // KERNEL_W
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        # For groups, in_c = group * GROUP_IN_C + in_c_local
        # But we need to handle groups properly
        # For simplicity with groups=1 or depthwise, handle separately
        in_c_abs = in_c_local  # group=0 case; for depthwise group=out_c
        
        valid_m = (offs_m[:, None] < M_total)
        valid_k = (k_offs[None, :] < TOTAL_K)
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        valid_batch = (batch_idx[:, None] < BATCH)
        
        x_mask = valid_m & valid_k & valid_spatial & valid_batch
        
        x_ptrs = (X_ptr + 
                  batch_idx[:, None] * s_xn +
                  in_c_abs[None, :] * s_xc)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # Weight: [OUT_C, IN_C_local, KH, KW] stored as [OUT_C, KH, KW, IN_C_local] (NHWC-like)
        w_mask = (k_offs[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_wo +
                  kh[:, None] * s_wkh +
                  kw[:, None] * s_wkw +
                  in_c_local[:, None] * s_wi)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Store output in NCHW format
    out_mask = (offs_m[:, None] < M_total) & (offs_n[None, :] < OUT_C)
    # Output index: batch * OUT_C * OUT_H * OUT_W + out_c * OUT_H * OUT_W + out_h * OUT_W + out_w
    out_ptrs = (out_ptr +
                batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
                offs_n[None, :] * (OUT_H * OUT_W) +
                out_h_idx[:, None] * OUT_W +
                out_w_idx[:, None])
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Depthwise Conv2d kernel (groups == IN_C == OUT_C)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=2, num_stages=2),
    ],
    key=['M_total', 'C', 'KH', 'KW'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, out_ptr,
    BATCH, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M_total,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """Depthwise conv: groups=C, each channel processed independently."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Loop over kernel positions
    for ki in range(KH):
        for kj in range(KW):
            ih = out_h_idx[:, None] * STRIDE_H + ki - PAD_H
            iw = out_w_idx[:, None] * STRIDE_W + kj - PAD_W
            
            valid = ((offs_m[:, None] < M_total) & 
                     (batch_idx[:, None] < BATCH) &
                     (ih >= 0) & (ih < IN_H) & 
                     (iw >= 0) & (iw < IN_W) &
                     (offs_c[None, :] < C))
            
            # X in NCHW: batch * C * IN_H * IN_W + c * IN_H * IN_W + ih * IN_W + iw
            x_ptrs = (X_ptr +
                      batch_idx[:, None] * (C * IN_H * IN_W) +
                      offs_c[None, :] * (IN_H * IN_W) +
            x_val = tl.load(x_ptrs, mask=valid, other=0.0)
            
            # W in NCHW depthwise: [C, 1, KH, KW] -> c * KH * KW + ki * KW + kj
            w_mask = offs_c < C
            w_ptrs = W_ptr + offs_c * (KH * KW) + ki * KW + kj
            w_val = tl.load(w_ptrs, mask=w_mask, other=0.0)
            
            acc += x_val * w_val[None, :]
    
    # Store output in NCHW
    out_mask = (offs_m[:, None] < M_total) & (offs_c[None, :] < C)
    out_ptrs = (out_ptr +
                batch_idx[:, None] * (C * OUT_H * OUT_W) +
                offs_c[None, :] * (OUT_H * OUT_W) +
                out_h_idx[:, None] * OUT_W +
                out_w_idx[:, None])
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Adaptive Average Pool 2d (global average pool for 1x1 output)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 32}, num_warps=2),
    ],
    key=['HW', 'C'],
)
@triton.jit
def global_avg_pool_kernel(
    x_ptr, out_ptr,
    BATCH, C, HW,
    BLOCK_HW: tl.constexpr,
):
    """Global average pooling: output shape [BATCH, C, 1, 1]"""
    pid = tl.program_id(0)  # one program per (batch, channel)
    batch_idx = pid // C
    c_idx = pid % C
    
    acc = tl.zeros((), dtype=tl.float32)
    
    base = x_ptr + batch_idx * C * HW + c_idx * HW
    
    for hw_start in range(0, HW, BLOCK_HW):
        offs = hw_start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        x = tl.load(base + offs, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    
    out_ptr_idx = batch_idx * C + c_idx
    tl.store(out_ptr + out_ptr_idx, acc / HW)


def run_bn_relu6(x, mean, var, weight, bias, apply_relu6=True, eps=1e-5):
    """Apply batch norm + optional relu6 in-place."""
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu6_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, C, HW,
        eps, apply_relu6,
    )
    return out


def run_conv2d(x, w, stride, padding, groups=1):
    """Run conv2d using implicit GEMM Triton kernel."""
    N, C_in, H, W = x.shape
    C_out, C_in_per_group, KH, KW = w.shape
    
    stride_h, stride_w = stride if isinstance(stride, (list, tuple)) else (stride, stride)
    pad_h, pad_w = padding if isinstance(padding, (list, tuple)) else (padding, padding)
    
    OUT_H = (H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (W + 2 * pad_w - KW) // stride_w + 1
    
    out = torch.zeros(N, C_out, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    if groups > 1 and C_in_per_group == 1:
        # Depthwise convolution
        M_total = N * OUT_H * OUT_W
        grid = lambda meta: (
            triton.cdiv(M_total, meta['BLOCK_M']),
            triton.cdiv(C_out, meta['BLOCK_N']),
        )
        depthwise_conv2d_kernel[grid](
            x, w, out,
            N, C_in, H, W,
            OUT_H, OUT_W,
            KH, KW,
            stride_h, stride_w,
            pad_h, pad_w,
            M_total,
        )
    else:
        # Regular conv2d using implicit GEMM
        # Convert x to NHWC for coalesced access
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        # Convert w to (OUT_C, KH, KW, IN_C_per_group) for NHWC-style access
        w_nhwc = w.permute(0, 2, 3, 1).contiguous()
        
        GROUP_IN_C = C_in // groups
        TOTAL_K = GROUP_IN_C * KH * KW
        M_total = N * OUT_H * OUT_W
        
        grid = lambda meta: (
            triton.cdiv(M_total, meta['BLOCK_M']),
            triton.cdiv(C_out, meta['BLOCK_N']),
        )
        
        conv2d_kernel[grid](
            x_nhwc, w_nhwc, out,
            N, C_in, H, W,
            C_out, OUT_H, OUT_W,
            KH, KW,
            stride_h, stride_w,
            pad_h, pad_w,
            groups,
            M_total, TOTAL_K,
            # NHWC strides for x
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            # (OUT_C, KH, KW, IN_C) strides for w
            w_nhwc.stride(0), w_nhwc.stride(1), w_nhwc.stride(2), w_nhwc.stride(3),
        )
    
    return out


def run_linear(x, w, bias):
    """Run linear layer using Triton GEMM kernel."""
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2
    
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    linear_kernel[grid](
        x, w, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
    )
    return out


def kernel_function(
    p_features_0_weight, p_features_1_weight, p_features_1_bias,
    p_features_3_0_weight, p_features_3_1_weight, p_features_3_1_bias,
    p_features_3_3_weight, p_features_3_4_weight, p_features_3_4_bias,
    p_features_4_0_weight, p_features_4_1_weight, p_features_4_1_bias,
    p_features_4_3_weight, p_features_4_4_weight, p_features_4_4_bias,
    p_features_4_6_weight, p_features_4_7_weight, p_features_4_7_bias,
    p_features_5_0_weight, p_features_5_1_weight, p_features_5_1_bias,
    p_features_5_3_weight, p_features_5_4_weight, p_features_5_4_bias,
    p_features_5_6_weight, p_features_5_7_weight, p_features_5_7_bias,
    p_features_6_0_weight, p_features_6_1_weight, p_features_6_1_bias,
    p_features_6_3_weight, p_features_6_4_weight, p_features_6_4_bias,
    p_features_6_6_weight, p_features_6_7_weight, p_features_6_7_bias,
    p_features_7_0_weight, p_features_7_1_weight, p_features_7_1_bias,
    p_features_7_3_weight, p_features_7_4_weight, p_features_7_4_bias,
    p_features_7_6_weight, p_features_7_7_weight, p_features_7_7_bias,
    p_features_8_0_weight, p_features_8_1_weight, p_features_8_1_bias,
    p_features_8_3_weight, p_features_8_4_weight, p_features_8_4_bias,
    p_features_8_6_weight, p_features_8_7_weight, p_features_8_7_bias,
    p_features_9_0_weight, p_features_9_1_weight, p_features_9_1_bias,
    p_features_9_3_weight, p_features_9_4_weight, p_features_9_4_bias,
    p_features_9_6_weight, p_features_9_7_weight, p_features_9_7_bias,
    p_features_10_0_weight, p_features_10_1_weight, p_features_10_1_bias,
    p_features_10_3_weight, p_features_10_4_weight, p_features_10_4_bias,
    p_features_10_6_weight, p_features_10_7_weight, p_features_10_7_bias,
    p_features_11_0_weight, p_features_11_1_weight, p_features_11_1_bias,
    p_features_11_3_weight, p_features_11_4_weight, p_features_11_4_bias,
    p_features_11_6_weight, p_features_11_7_weight, p_features_11_7_bias,
    p_features_12_0_weight, p_features_12_1_weight, p_features_12_1_bias,
    p_features_12_3_weight, p_features_12_4_weight, p_features_12_4_bias,
    p_features_12_6_weight, p_features_12_7_weight, p_features_12_7_bias,
    p_features_13_0_weight, p_features_13_1_weight, p_features_13_1_bias,
    p_features_13_3_weight, p_features_13_4_weight, p_features_13_4_bias,
    p_features_13_6_weight, p_features_13_7_weight, p_features_13_7_bias,
    p_features_14_0_weight, p_features_14_1_weight, p_features_14_1_bias,
    p_features_14_3_weight, p_features_14_4_weight, p_features_14_4_bias,
    p_features_14_6_weight, p_features_14_7_weight, p_features_14_7_bias,
    p_features_15_0_weight, p_features_15_1_weight, p_features_15_1_bias,
    p_features_15_3_weight, p_features_15_4_weight, p_features_15_4_bias,
    p_features_15_6_weight, p_features_15_7_weight, p_features_15_7_bias,
    p_features_16_0_weight, p_features_16_1_weight, p_features_16_1_bias,
    p_features_16_3_weight, p_features_16_4_weight, p_features_16_4_bias,
    p_features_16_6_weight, p_features_16_7_weight, p_features_16_7_bias,
    p_features_17_0_weight, p_features_17_1_weight, p_features_17_1_bias,
    p_features_17_3_weight, p_features_17_4_weight, p_features_17_4_bias,
    p_features_17_6_weight, p_features_17_7_weight, p_features_17_7_bias,
    p_features_18_0_weight, p_features_18_1_weight, p_features_18_1_bias,
    p_features_18_3_weight, p_features_18_4_weight, p_features_18_4_bias,