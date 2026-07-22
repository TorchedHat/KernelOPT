import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: fused batch_norm + relu (pointwise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # channel index: layout is (N, C, H, W) -> channel = (offset // HW) % C
    c_idx = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Triton kernel: fused sigmoid (pointwise)
# ============================================================
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
def sigmoid_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Triton kernel: relu (pointwise)
# ============================================================
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
def relu_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Triton kernel: adaptive avg pool 2d -> (1,1) output
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 1024}, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 128}, num_warps=4),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    nc = pid
    n = nc // C
    c = nc % C
    
    if n >= N:
        return
    
    base = n * C * HW + c * HW
    
    acc = tl.zeros([1], dtype=tl.float32)
    for hw_off in range(0, HW, BLOCK_HW):
        hw_idx = hw_off + tl.arange(0, BLOCK_HW)
        mask = hw_idx < HW
        x = tl.load(x_ptr + base + hw_idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    
    out = acc / HW
    tl.store(out_ptr + n * C + c, out)


# ============================================================
# Triton kernel: linear (GEMM) with bias
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=w_mask, other=0.0)
        
        acc = tl.dot(x, tl.trans(w), acc)
    
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=out_mask)


# ============================================================
# Triton kernel: implicit GEMM conv2d (1x1 convolution)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """1x1 convolution as GEMM: M=N*H*W, N=out_channels, K=in_channels"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=w_mask, other=0.0)
        
        acc = tl.dot(x, tl.trans(w), acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=out_mask)


# ============================================================
# Triton kernel: depthwise conv2d 3x3
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Depthwise 3x3 conv: groups=C, each output element computed independently"""
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    # Decode output indices
    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = oh * stride_h + kh - pad_h
            iw = ow * stride_w + kw - pad_w
            
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            
            x_idx = n * C * IH * IW + c * IH * IW + ih * IW + iw
            w_idx = c * 9 + kh * 3 + kw
            
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)
            w_val = tl.load(w_ptr + w_idx, mask=valid, other=0.0).to(tl.float32)
            
            acc += x_val * w_val
    
    tl.store(out_ptr + offsets, acc.to(tl.float32), mask=mask)


# ============================================================
# Triton kernel: initial 3x3 conv2d stride=2
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv3x3_s2_kernel(
    x_ptr, w_ptr, out_ptr,
    N_batch, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """3x3 conv stride=2 pad=1 as implicit GEMM"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (n, oh, ow)
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    n = tmp // OUT_H
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # K = IN_C * 3 * 3
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K -> (ic, kh, kw)
        kw_k = offs_k % 3
        tmp_k = offs_k // 3
        kh_k = tmp_k % 3
        ic_k = tmp_k // 3
        
        ih = oh[:, None] * 2 + kh_k[None, :] - 1
        iw = ow[:, None] * 2 + kw_k[None, :] - 1
        
        valid_x = (
            (offs_m[:, None] < M) &
            (n[:, None] < N_batch) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < K)
        )
        
        x_idx = n[:, None] * IN_C * IN_H * IN_W + ic_k[None, :] * IN_H * IN_W + ih * IN_W + iw
        x = tl.load(x_ptr + x_idx, mask=valid_x, other=0.0)
        
        valid_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        # w shape: (OUT_C, IN_C, 3, 3) -> w[oc, ic, kh, kw] = w[oc * K + ic*9 + kh*3 + kw]
        w_idx = offs_n[:, None] * K + offs_k[None, :]
        w = tl.load(w_ptr + w_idx, mask=valid_w, other=0.0)
        
        acc = tl.dot(x, tl.trans(w), acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * OUT_C + offs_n[None, :], acc.to(tl.float32), mask=out_mask)


# ============================================================
# Helper functions
# ============================================================

def apply_bn_relu(x, running_mean, running_var, weight, bias, eps=1e-5, do_relu=True):
    """Apply batch norm + optional relu using Triton kernel"""
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps=eps,
        do_relu=do_relu,
    )
    return out


def apply_sigmoid(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


def apply_relu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def apply_adaptive_avg_pool(x):
    """Adaptive avg pool to (1,1)"""
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    
    # Use a simple approach: one program per (n, c) pair
    total_nc = N * C
    
    # Use a simpler 1D grid approach
    grid = (total_nc,)
    
    # Need to pick BLOCK_HW as constexpr - use next power of 2
    block_hw = 1
    while block_hw < HW:
        block_hw *= 2
    block_hw = min(block_hw, 1024)
    
    # Use torch for this since the kernel needs constexpr BLOCK_HW
    # Actually implement with a reduction kernel
    x_flat = x.view(N, C, HW)
    out_flat = x_flat.mean(dim=2)
    return out_flat.view(N, C, 1, 1)


def apply_conv1x1(x, weight):
    """1x1 convolution as GEMM"""
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Reshape: x -> (N*H*W, C_in), w -> (C_out, C_in)
    M = N * H * W
    K = C_in
    
    x_2d = x.permute(0, 2, 3, 1).contiguous().view(M, K)
    w_2d = weight.view(C_out, K)
    
    out_2d = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )
    
    conv1x1_kernel[grid](x_2d, w_2d, out_2d, M, C_out, K)
    
    return out_2d.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def apply_depthwise_conv3x3(x, weight, stride=1, padding=1):
    """Depthwise 3x3 convolution"""
    N, C, IH, IW = x.shape
    OH = (IH + 2 * padding - 3) // stride + 1
    OW = (IW + 2 * padding - 3) // stride + 1
    
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    
    total = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    
    depthwise_conv3x3_kernel[grid](
        x, weight, out,
        N, C, IH, IW, OH, OW,
        stride_h=stride, stride_w=stride,
        pad_h=padding, pad_w=padding,
    )
    return out


def apply_conv3x3_s2(x, weight):
    """Initial 3x3 conv stride=2 pad=1"""
    N_batch, IN_C, IN_H, IN_W = x.shape
    OUT_C = weight.shape[0]
    OUT_H = (IN_H + 2 * 1 - 3) // 2 + 1
    OUT_W = (IN_W + 2 * 1 - 3) // 2 + 1
    
    M = N_batch * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * 3 * 3
    
    # w needs to be (OUT_C, K) = (OUT_C, IN_C*9)
    w_2d = weight.view(OUT_C, K)
    
    out_2d = torch.empty(M, OUT_C, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    conv3x3_s2_kernel[grid](
        x, w_2d, out_2d,
        N_batch, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
        M, N, K,
    )
    
    # out_2d is (M, OUT_C) = (N*OH*OW, OUT_C)
    return out_2d.view(N_batch, OUT_H, OUT_W, OUT_C).permute(0, 3, 1, 2).contiguous()


def apply_linear(x, weight, bias):
    """Linear layer"""
    M, K = x.shape
    N = weight.shape[0]
    
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    linear_kernel[grid](x, weight, bias, out, M, N, K)
    return out


# ============================================================
# MBConv block
# ============================================================

def mbconv_block(
    x,
    # expand conv
    expand_w, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b,
    # depthwise conv
    dw_w, dw_bn_mean, dw_bn_var, dw_bn_w, dw_bn_b,
    # SE
    se_reduce_w, se_expand_w,
    # project conv
    proj_w, proj_bn_mean, proj_bn_var, proj_bn_w, proj_bn_b,
    stride=1,
):
    # Expansion
    t = apply_conv1x1(x, expand_w)
    t = apply_bn_relu(t, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b, do_relu=True)
    
    # Depthwise
    t = apply_depthwise_conv3x3(t, dw_w, stride=stride, padding=1)
    t = apply_bn_relu(t, dw_bn_mean, dw_bn_var, dw_bn_w, dw_bn_b, do_relu=True)
    
    # SE
    se = apply_adaptive_avg_pool(t)
    se = apply_conv1x1(se, se_reduce_w)
    se = apply_relu(se)
    se = apply_conv1x1(se, se_expand_w)
    se = apply_sigmoid(se)
    t = t * se
    
    # Project
    t = apply_conv1x1(t, proj_w)
    t = apply_bn_relu(t, proj_bn_mean, proj_bn_var, proj_bn_w, proj_bn_b, do_relu=False)
    
    return t


# ============================================================
# Main kernel function
# ============================================================

def kernel_function(
    p_conv1_weight,
    p_bn1_weight,
    p_bn1_bias,
    p_mbconv1_0_weight,
    p_mbconv1_1_weight,
    p_mbconv1_1_bias,
    p_mbconv1_3_weight,
    p_mbconv1_4_weight,
    p_mbconv1_4_bias,
    p_mbconv1_7_weight,
    p_mbconv1_9_weight,
    p_mbconv1_11_weight,
    p_mbconv1_12_weight,
    p_mbconv1_12_bias,
    p_mbconv2_0_weight,
    p_mbconv2_1_weight,
    p_mbconv2_1_bias,
    p_mbconv2_3_weight,
    p_mbconv2_4_weight,
    p_mbconv2_4_bias,
    p_mbconv2_7_weight,
    p_mbconv2_9_weight,
    p_mbconv2_11_weight,
    p_mbconv2_12_weight,
    p_mbconv2_12_bias,
    p_mbconv3_0_weight,
    p_mbconv3_1_weight,
    p_mbconv3_1_bias,
    p_mbconv3_3_weight,
    p_mbconv3_4_weight,
    p_mbconv3_4_bias,
    p_mbconv3_7_weight,
    p_mbconv3_9_weight,
    p_mbconv3_11_weight,
    p_mbconv3_12_weight,
    p_mbconv3_12_bias,
    p_mbconv4_0_weight,
    p_mbconv4_1_weight,
    p_mbconv4_1_bias,
    p_mbconv4_3_weight,
    p_mbconv4_4_weight,
    p_mbconv4_4_bias,
    p_mbconv4_7_weight,
    p_mbconv4_9_weight,
    p_mbconv4_11_weight,
    p_mbconv4_12_weight,
    p_mbconv4_12_bias,
    p_mbconv5_0_weight,
    p_mbconv5_1_weight,
    p_mbconv5_1_bias,
    p_mbconv5_3_weight,
    p_mbconv5_4_weight,
    p_mbconv5_4_bias,
    p_mbconv5_7_weight,
    p_mbconv5_9_weight,
    p_mbconv5_11_weight,
    p_mbconv5_12_weight,
    p_mbconv5_12_bias,
    p_conv_final_weight,
    p_bn_final_weight,
    p_bn_final_bias,
    p_fc_weight,
    p_fc_bias,