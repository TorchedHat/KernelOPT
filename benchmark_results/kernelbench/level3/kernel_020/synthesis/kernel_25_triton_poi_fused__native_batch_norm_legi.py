import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused BatchNorm + optional ReLU6 pointwise kernel
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
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def bn_relu6(x, mean, var, weight, bias, eps=1e-5, apply_relu6=True):
    out = torch.empty_like(x)
    numel = x.numel()
    N, C, H, W = x.shape
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu6_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


# ============================================================
# Implicit GEMM Conv2d Triton kernel (NHWC layout internally)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    # X shape: (N, H, W, C_in) - NHWC
    # W shape: (C_out, C_in, KH, KW) - standard
    # Out shape: (N, OH, OW, C_out) - NHWC
    BATCH, IN_H, IN_W, IN_C,
    OUT_H, OUT_W, OUT_C,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N, K,  # M=BATCH*OH*OW, N=OUT_C, K=IN_C*KH*KW
    # strides for X (NHWC)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (C_out, C_in, KH, KW)
    s_wco, s_wci, s_wkh, s_wkw,
    # strides for Out (NHWC)
    s_on, s_oh, s_ow, s_oc,
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
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KW
        kh = tmp_k // KW
        
        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = (offs_m < M)[:, None]
        valid_k = (offs_k < K)[None, :]
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_x = valid_m & valid_k & valid_ih & valid_iw
        
        x_ptrs = (X_ptr + 
                  batch[:, None] * s_xn + 
                  in_c[None, :] * s_xc)
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        # Weight: (C_out, C_in, KH, KW) -> index [offs_n, in_c, kh, kw]
        valid_n = (offs_n < N)[None, :]
        valid_wk = (offs_k < K)[:, None]
        valid_w = valid_wk & valid_n
        
        w_ptrs = (W_ptr + 
                  offs_n[None, :] * s_wco + 
                  in_c[:, None] * s_wci + 
                  kh[:, None] * s_wkh + 
                  kw[:, None] * s_wkw)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)
    
    # Store output
    valid_out_m = (offs_m < M)[:, None]
    valid_out_n = (offs_n < N)[None, :]
    out_mask = valid_out_m & valid_out_n
    
    out_ptrs = (Out_ptr + 
                batch[:, None] * s_on + 
                out_h[:, None] * s_oh + 
                out_w[:, None] * s_ow + 
                offs_n[None, :] * s_oc)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def conv2d_triton(x_nchw, weight, stride=(1,1), padding=(0,0), groups=1):
    """
    """
    N, C_in, H, W = x_nchw.shape
    C_out, C_in_per_group, KH, KW = weight.shape
    
    stride_h, stride_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    pad_h, pad_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    
    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1
    
    if groups == 1:
        # Convert to NHWC for coalesced access
        x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()
        out_nhwc = torch.empty((N, OH, OW, C_out), device=x_nchw.device, dtype=x_nchw.dtype)
        
        M = N * OH * OW
        Nk = C_out
        K = C_in * KH * KW
        
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(Nk, meta['BLOCK_N']),
        )
        
        conv2d_implicit_gemm_kernel[grid](
            x_nhwc, weight, out_nhwc,
            N, H, W, C_in,
            OH, OW, C_out,
            KH, KW,
            stride_h, stride_w,
            pad_h, pad_w,
            M, Nk, K,
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
            out_nhwc.stride(0), out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
        )
        
        return out_nhwc.permute(0, 3, 1, 2).contiguous()
    else:
        # Depthwise convolution (groups == C_in == C_out)
        return conv2d_depthwise_triton(x_nchw, weight, stride_h, stride_w, pad_h, pad_w)


# ============================================================
# Depthwise Conv2d Triton kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'KH', 'KW'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, C, IN_H, IN_W,
    OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    numel,
    s_xn, s_xc, s_xh, s_xw,
    s_wc, s_wkh, s_wkw,
    s_on, s_oc, s_oh, s_ow,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Decode: offsets -> (n, c, oh, ow)
    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            
            x_ptrs = X_ptr + n * s_xn + c * s_xc + ih * s_xh + iw * s_xw
            w_ptrs = W_ptr + c * s_wc + kh * s_wkh + kw * s_wkw
            
            x_val = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.float32)
            w_val = tl.load(w_ptrs, mask=mask, other=0.0).to(tl.float32)
            
            acc += x_val * w_val
    
    out_ptrs = Out_ptr + n * s_on + c * s_oc + oh * s_oh + ow * s_ow
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


def conv2d_depthwise_triton(x_nchw, weight, stride_h, stride_w, pad_h, pad_w):
    N, C, H, W = x_nchw.shape
    C_out, _, KH, KW = weight.shape
    
    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1
    
    out = torch.empty((N, C_out, OH, OW), device=x_nchw.device, dtype=x_nchw.dtype)
    numel = N * C_out * OH * OW
    
    # Weight for depthwise: (C, 1, KH, KW)
    w_cont = weight.contiguous()
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    depthwise_conv2d_kernel[grid](
        x_nchw, w_cont, out,
        N, C, H, W,
        OH, OW,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        numel,
        x_nchw.stride(0), x_nchw.stride(1), x_nchw.stride(2), x_nchw.stride(3),
        w_cont.stride(0), w_cont.stride(2), w_cont.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


# ============================================================
# Linear layer Triton kernel (GEMM: M x K @ K x N)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    A_ptr, B_ptr, bias_ptr, Out_ptr,
    M, N, K,
    s_am, s_ak,
    s_bn, s_bk,
    s_om, s_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        a_ptrs = A_ptr + offs_m[:, None] * s_am + offs_k[None, :] * s_ak
        b_ptrs = B_ptr + offs_n[None, :] * s_bn + offs_k[:, None] * s_bk
        
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc = tl.dot(a, b, acc=acc, allow_tf32=False)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = Out_ptr + offs_m[:, None] * s_om + offs_n[None, :] * s_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def linear_triton(x, weight, bias):
    """x: (M, K), weight: (N, K), bias: (N,) -> (M, N)"""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


# ============================================================
# Adaptive Average Pooling (global) Triton kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['NC', 'HW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    X_ptr, Out_ptr,
    NC, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles one (n, c) pair
    nc = pid
    if nc >= NC:
        return
    
    acc = tl.zeros([1], dtype=tl.float32)
    
    for hw_start in range(0, HW, BLOCK_SIZE):
        offs = hw_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        x = tl.load(X_ptr + nc * HW + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    
    tl.store(Out_ptr + nc, (acc / HW).to(tl.float32))


def adaptive_avg_pool2d_triton(x):
    """Global average pooling: (N, C, H, W) -> (N, C, 1, 1)"""
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C
    x_cont = x.contiguous()
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    
    grid = (NC,)
    adaptive_avg_pool2d_kernel[grid](
        x_cont, out.view(NC),
        NC, HW,
    )
    return out


# ============================================================
# Main kernel_function: MobileNetV2 forward pass
# ============================================================
def kernel_function(
    p_features_0_weight,
    p_features_1_weight, p_features_1_bias,
    p_features_3_0_weight,
    p_features_3_1_weight, p_features_3_1_bias,
    p_features_3_3_weight,
    p_features_3_4_weight, p_features_3_4_bias,
    p_features_4_0_weight,
    p_features_4_1_weight, p_features_4_1_bias,
    p_features_4_3_weight,
    p_features_4_4_weight, p_features_4_4_bias,
    p_features_4_6_weight,
    p_features_4_7_weight, p_features_4_7_bias,
    p_features_5_0_weight,
    p_features_5_1_weight, p_features_5_1_bias,
    p_features_5_3_weight,
    p_features_5_4_weight, p_features_5_4_bias,
    p_features_5_6_weight,
    p_features_5_7_weight, p_features_5_7_bias,
    p_features_6_0_weight,
    p_features_6_1_weight, p_features_6_1_bias,
    p_features_6_3_weight,
    p_features_6_4_weight, p_features_6_4_bias,
    p_features_6_6_weight,
    p_features_6_7_weight, p_features_6_7_bias,
    p_features_7_0_weight,
    p_features_7_1_weight, p_features_7_1_bias,
    p_features_7_3_weight,
    p_features_7_4_weight, p_features_7_4_bias,
    p_features_7_6_weight,
    p_features_7_7_weight, p_features_7_7_bias,
    p_features_8_0_weight,
    p_features_8_1_weight, p_features_8_1_bias,
    p_features_8_3_weight,
    p_features_8_4_weight, p_features_8_4_bias,
    p_features_8_6_weight,
    p_features_8_7_weight, p_features_8_7_bias,
    p_features_9_0_weight,
    p_features_9_1_weight, p_features_9_1_bias,
    p_features_9_3_weight,
    p_features_9_4_weight, p_features_9_4_bias,
    p_features_9_6_weight,
    p_features_9_7_weight, p_features_9_7_bias,
    p_features_10_0_weight,
    p_features_10_1_weight, p_features_10_1_bias,
    p_features_10_3_weight,
    p_features_10_4_weight, p_features_10_4_bias,
    p_features_10_6_weight,
    p_features_10_7_weight, p_features_10_7_bias,
    p_features_11_0_weight,
    p_features_11_1_weight, p_features_11_1_bias,
    p_features_11_3_weight,
    p_features_11_4_weight, p_features_11_4_bias,
    p_features_11_6_weight,
    p_features_11_7_weight, p_features_11_7_bias,
    p_features_12_0_weight,
    p_features_12_1_weight, p_features_12_1_bias,
    p_features_12_3_weight,
    p_features_12_4_weight, p_features_12_4_bias,
    p_features_12_6_weight,
    p_features_12_7_weight, p_features_12_7_bias,
    p_features_13_0_weight,
    p_features_13_1_weight, p_features_13_1_bias,
    p_features_13_3_weight,
    p_features_13_4_weight, p_features_13_4_bias,
    p_features_13_6_weight,
    p_features_13_7_weight, p_features_13_7_bias,
    p_features_14_0_weight,
    p_features_14_1_weight, p_features_14_1_bias,
    p_features_14_3_weight,
    p_features_14_4_weight, p_features_14_4_bias,
    p_features_14_6_weight,
    p_features_14_7_weight, p_features_14_7_bias,
    p_features_15_0_weight,
    p_features_15_1_weight, p_features_15_1_bias,
    p_features_15_3_weight,
    p_features_15_4_weight, p_features_15_4_bias,
    p_features_15_6_weight,
    p_features_15_7_weight, p_features_15_7_bias,
    p_features_16_0_weight,
    p_features_16_1_weight, p_features_16_1_bias,
    p_features_16_3_weight,
    p_features_16_4_weight, p_features_16_4_bias,
    p_features_16_6_weight,
    p_features_16_7_weight, p_features_16_7_bias,
    p_features_17_0_weight,
    p_features_17_1_weight, p_features_17_1_bias,
    p_features_17_3_weight,
    p_features_17_4_weight, p_features_17_4_bias,
    p_features_17_6_weight,
    p_features_17_7_weight, p_features_17_7_bias,
    p_features_18_0_weight,
    p_features_18_1_weight, p_features_18_1_bias,
    p_features_18_3_weight,
    p_features_18_4_weight, p_features_18_4_bias,
    p_features_18_6_weight,