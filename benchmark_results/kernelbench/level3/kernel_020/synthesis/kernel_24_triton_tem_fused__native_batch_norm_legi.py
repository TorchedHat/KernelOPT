import torch
import triton
import triton.language as tl
import torch.nn as nn


import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm + optional ReLU6 (hardtanh(0,6)) kernel
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
def fused_bn_relu6_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    # Layout: N, C, H, W -> flat index: n*C*HW + c*HW + hw
    # channel = (offset // HW) % C
    hw_idx = offsets % HW
    c_idx = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * weight + bias
    
    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def bn_relu6(x, weight, bias, running_mean, running_var, eps=1e-5, apply_relu6=True):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu6_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps,
        apply_relu6,
    )
    return out


# Implicit GEMM conv2d kernel for 1x1 convolutions
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
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
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        mask_x = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        mask_w = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def conv1x1(x, w):
    # x: (N, C_in, H, W), w: (C_out, C_in, 1, 1)
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    M = N * H * W
    K = C_in
    
    # Reshape x to (M, K) - need contiguous
    x_2d = x.permute(0, 2, 3, 1).contiguous().view(M, K)
    w_2d = w.view(C_out, K)  # (N, K)
    
    out_2d = torch.empty((M, C_out), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv1x1_kernel[grid](
        x_2d, w_2d, out_2d,
        M, C_out, K,
        x_2d.stride(0), x_2d.stride(1),
        w_2d.stride(0), w_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
    )
    
    return out_2d.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


# Depthwise conv 3x3 kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 16}, num_warps=8),
    ],
    key=['M', 'C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    # Decode m -> (n, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    n = offs_m // (OUT_W * OUT_H)
    
    mask_m = offs_m < M
    mask_c = offs_c < C
    
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            
            valid = mask_m & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            
            # x: (N, C, H, W) -> index: n*C*IN_H*IN_W + c*IN_H*IN_W + ih*IN_W + iw
            x_idx = n[:, None] * (C * IN_H * IN_W) + offs_c[None, :] * (IN_H * IN_W) + ih[:, None] * IN_W + iw[:, None]
            x_mask = valid[:, None] & mask_c[None, :]
            x_val = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
            
            # w: (C, 1, 3, 3) -> index: c*9 + kh*3 + kw
            w_idx = offs_c * 9 + kh * 3 + kw
            w_val = tl.load(w_ptr + w_idx, mask=mask_c, other=0.0)
            
            acc += x_val * w_val[None, :]
    
    # Store: out (N, C, OUT_H, OUT_W)
    out_idx = n[:, None] * (C * OUT_H * OUT_W) + offs_c[None, :] * (OUT_H * OUT_W) + oh[:, None] * OUT_W + ow[:, None]
    out_mask = mask_m[:, None] & mask_c[None, :]
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=out_mask)


def depthwise_conv3x3(x, w, stride=1, padding=1):
    N, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + 2 * padding - 3) // stride + 1
    OUT_W = (IN_W + 2 * padding - 3) // stride + 1
    M = N * OUT_H * OUT_W
    
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C, meta['BLOCK_C']))
    depthwise_conv3x3_kernel[grid](
        x, w, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        stride, stride,
        padding, padding,
        M,
    )
    return out


# Standard 3x3 conv (for first layer: 3->32, stride=2)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N_batch, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m -> (n, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    n = offs_m // (OUT_W * OUT_H)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = IN_C * 3 * 3
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Decode k -> (ic, kh, kw)
        ic = offs_k % IN_C
        kh = (offs_k // IN_C) % 3
        kw = offs_k // (IN_C * 3)
        
        ih = oh[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid = mask_m[:, None] & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (offs_k[None, :] < K)
        
        x_idx = n[:, None] * (IN_C * IN_H * IN_W) + ic[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
        
        # w: (OUT_C, IN_C, 3, 3) -> w[oc, ic, kh, kw] = oc*(IN_C*9) + ic*9 + kh*3 + kw
        # But we need w[k, n] where k=(ic,kh,kw), n=oc
        # w_idx[k, n] = offs_n[n] * (IN_C*9) + ic[k]*9 + kh[k]*3 + kw[k]
        w_idx = offs_n[None, :] * K + ic[:, None] * 9 + kh[:, None] * 3 + kw[:, None]
        mask_w = (offs_k[:, None] < K) & mask_n[None, :]
        w_tile = tl.load(w_ptr + w_idx, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    out_idx = n[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + oh[:, None] * OUT_W + ow[:, None]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=out_mask)


def conv3x3(x, w, stride=1, padding=1):
    N_batch, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, _, _ = w.shape
    OUT_H = (IN_H + 2 * padding - 3) // stride + 1
    OUT_W = (IN_W + 2 * padding - 3) // stride + 1
    M = N_batch * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * 9
    
    out = torch.empty((N_batch, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    # Reorder w to (OUT_C, K) where K = IN_C * kh * kw, with kw varying fastest
    # Original w: (OUT_C, IN_C, 3, 3) with layout [oc, ic, kh, kw]
    # We need w[oc, k] where k = ic*9 + kh*3 + kw... but our kernel decodes k as:
    # ic = k % IN_C, kh = (k // IN_C) % 3, kw = k // (IN_C * 3)
    # So k = ic + IN_C * kh + IN_C * 3 * kw
    # We need to reorder w accordingly
    # w original: (OUT_C, IN_C, kh, kw) -> w[oc, ic, kh, kw]
    # We want w_flat[oc, ic + IN_C*kh + IN_C*3*kw] = w[oc, ic, kh, kw]
    # Permute: (OUT_C, kw, kh, IN_C) then reshape
    w_flat = w.permute(0, 3, 2, 1).contiguous().view(OUT_C, K)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    conv3x3_kernel[grid](
        x, w_flat, out,
        N_batch, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
        stride, stride, padding, padding,
        M, N, K,
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
    p_features_18_6_weight, p_features_18_7_weight, p_features_18_7_bias,
    p_features_19_0_weight, p_features_19_1_weight, p_features_19_1_bias,
    p_features_19_3_weight, p_features_19_4_weight, p_features_19_4_bias,
    p_features_19_6_weight, p_features_19_7_weight, p_features_19_7_bias,
    p_features_20_weight, p_features_21_weight, p_features_21_bias,
    p_classifier_1_weight, p_classifier_1_bias,
    b_features_1_running_mean, b_features_1_running_var,
    b_features_3_1_running_mean, b_features_3_1_running_var,
    b_features_3_4_running_mean, b_features_3_4_running_var,
    b_features_4_1_running_mean, b_features_4_1_running_var,
    b_features_4_4_running_mean, b_features_4_4_running_var,
    b_features_4_7_running_mean, b_features_4_7_running_var,
    b_features_5_1_running_mean, b_features_5_1_running_var,
    b_features_5_4_running_mean, b_features_5_4_running_var,
    b_features_5_7_running_mean, b_features_5_7_running_var,
    b_features_6_1_running_mean, b_features_6_1_running_var,
    b_features_6_4_running_mean, b_features_6_4_running_var,
    b_features_6_7_running_mean, b_features_6_7_running_var,
    b_features_7_1_running_mean, b_features_7_1_running_var,
    b_features_7_4_running_mean, b_features_7_4_running_var,
    b_features_7_7_running_mean, b_features_7_7_running_var,
    b_features_8_1_running_mean, b_features_8_1_running_var,
    b_features_8_4_running_mean, b_features_8_4_running_var,
    b_features_8_7_running_mean, b_features_8_7_running_var,
    b_features_9_1_running_mean, b_features_9_1_running_var,
    b_features_9_4_running_mean, b_features_9_4_running_var,
    b_features_9_7_running_mean, b_features_9_7_running_var,
    b_features_10_1_running_mean, b_features_10_1_running_var,
    b_features_10_4_running_mean, b_features_10_4_running_var,
    b_features_10_7_running_mean, b_features_10_7_running_var,
    b_features_11_1_running_mean, b_features_11_1_running_var,
    b_features_11_4_running_mean, b_features_11_4_running_var,
    b_features_11_7_running_mean, b_features_11_7_running_var,
    b_features_12_1_running_mean, b_features_12_1_running_var,