import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused BatchNorm + ReLU pointwise kernel
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
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
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
    x_ptr, out_ptr, numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


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
    x_ptr, out_ptr, numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


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
def mul_kernel(
    a_ptr, b_ptr, out_ptr, numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a * b, mask=mask)


# Adaptive avg pool 2d -> output (N, C, 1, 1)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 32}, num_warps=2),
    ],
    key=['HW', 'NC'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, HW, NC,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c) pair
    nc_idx = tl.program_id(0)
    if nc_idx >= NC:
        return
    
    n = nc_idx // C
    c = nc_idx % C
    
    base = n * C * HW + c * HW
    
    acc = tl.zeros([1], dtype=tl.float32)
    for hw_off in range(0, HW, BLOCK_HW):
        hw_idx = hw_off + tl.arange(0, BLOCK_HW)
        mask = hw_idx < HW
        x = tl.load(x_ptr + base + hw_idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    
    out = acc / HW
    tl.store(out_ptr + nc_idx, out)


# Linear kernel: y = x @ W^T + b
# x: (M, K), W: (N, K), b: (N,), out: (M, N)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
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
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=w_mask, other=0.0)
        
        acc = tl.dot(x, tl.trans(w), acc)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# Conv2d kernel (implicit GEMM)
# Input: (N, C_in, H, W) in NCHW format
# Weight: (C_out, C_in, KH, KW)
# Output: (N, C_out, OH, OW)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C_in, H, W,
    C_out, OH, OW,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    groups,
    M, K,  # M = N*OH*OW, K = C_in_per_group * KH * KW
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, oh, ow)
    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    batch = offs_m // (OW * OH)
    
    # Determine group
    C_in_per_group = C_in // groups
    C_out_per_group = C_out // groups
    group = offs_n // C_out_per_group
    c_out_local = offs_n % C_out_per_group
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K -> (c_in_local, kh, kw)
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        c_in_local = offs_k // (KW * KH)
        
        # Input coordinates
        ih = oh[:, None] * stride_h + kh[None, :] - pad_h
        iw = ow[:, None] * stride_w + kw[None, :] - pad_w
        c_in = group[:, None] * C_in_per_group + c_in_local[None, :]
        
        valid_m = offs_m < M
        valid_k = offs_k < K
        valid_x = (valid_m[:, None] & valid_k[None, :] & 
                   (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) &
                   (batch[:, None] < N))
        
        x_idx = batch[:, None] * (C_in * H * W) + c_in * (H * W) + ih * W + iw
        x = tl.load(x_ptr + x_idx, mask=valid_x, other=0.0)
        
        # Weight: (C_out, C_in_per_group, KH, KW)
        # w_idx for each (c_out, c_in_local, kh, kw)
        valid_w = (valid_k[:, None] & (offs_n[None, :] < C_out))
        w_idx = (offs_n[None, :] * (C_in_per_group * KH * KW) + 
                 c_in_local[:, None] * (KH * KW) + 
                 kh[:, None] * KW + kw[:, None])
        w = tl.load(w_ptr + w_idx, mask=valid_w, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < C_out)
    # Output in NCHW: out[batch, c_out, oh, ow]
    out_idx = batch[:, None] * (C_out * OH * OW) + offs_n[None, :] * (OH * OW) + oh[:, None] * OW + ow[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def triton_conv2d(x, w, stride=(1,1), padding=(0,0), groups=1):
    N, C_in, H, W = x.shape
    C_out, C_in_per_group, KH, KW = w.shape
    assert C_in_per_group * groups == C_in
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1
    
    M = N * OH * OW
    K = (C_in // groups) * KH * KW
    
    out = torch.empty(N, C_out, OH, OW, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x, w, out,
        N, C_in, H, W,
        C_out, OH, OW,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        groups,
        M, K,
    )
    return out


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    N, C, H, W = x.shape
    HW = H * W
    numel = x.numel()
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        numel, C, HW,
        eps,
        do_relu,
    )
    return out


def triton_relu(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def triton_sigmoid(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


def triton_adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    HW = H * W
    NC = N * C
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    
    grid = (NC,)
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        N, C, HW, NC,
    )
    return out


def triton_linear(x, w, b):
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_kernel[grid](x, w, b, out, M, N, K)
    return out


def mbconv_block(x, 
                 expand_w, expand_bn_w, expand_bn_b, expand_bn_mean, expand_bn_var,
                 dw_w, dw_bn_w, dw_bn_b, dw_bn_mean, dw_bn_var,
                 se_reduce_w, se_expand_w,
                 proj_w, proj_bn_w, proj_bn_b, proj_bn_mean, proj_bn_var,
                 stride, expand_ratio, eps=1e-5):
    """MBConv block forward pass"""
    identity = x
    
    # Expansion phase (if expand_ratio != 1)
    if expand_ratio != 1:
        x = triton_conv2d(x, expand_w, stride=(1,1), padding=(0,0))
        x = triton_bn_relu(x, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b, eps=eps, do_relu=True)
    
    # Depthwise conv
    N, C, H, W = x.shape
    x = triton_conv2d(x, dw_w, stride=(stride, stride), padding=(1,1), groups=C)
    x = triton_bn_relu(x, dw_bn_mean, dw_bn_var, dw_bn_w, dw_bn_b, eps=eps, do_relu=True)
    
    # Squeeze and Excitation
    se = triton_adaptive_avg_pool2d(x)
    se = triton_conv2d(se, se_reduce_w, stride=(1,1), padding=(0,0))
    se = triton_relu(se)
    se = triton_conv2d(se, se_expand_w, stride=(1,1), padding=(0,0))
    se = triton_sigmoid(se)
    
    # Scale
    x = x * se
    
    # Projection
    x = triton_conv2d(x, proj_w, stride=(1,1), padding=(0,0))
    x = triton_bn_relu(x, proj_bn_mean, proj_bn_var, proj_bn_w, proj_bn_b, eps=eps, do_relu=False)
    
    return x


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_mbconv1_0_weight, p_mbconv1_1_weight, p_mbconv1_1_bias,
    p_mbconv1_3_weight, p_mbconv1_4_weight, p_mbconv1_4_bias,
    p_mbconv1_7_weight, p_mbconv1_9_weight,
    p_mbconv1_11_weight, p_mbconv1_12_weight, p_mbconv1_12_bias,
    p_mbconv2_0_weight, p_mbconv2_1_weight, p_mbconv2_1_bias,
    p_mbconv2_3_weight, p_mbconv2_4_weight, p_mbconv2_4_bias,
    p_mbconv2_7_weight, p_mbconv2_9_weight,
    p_mbconv2_11_weight, p_mbconv2_12_weight, p_mbconv2_12_bias,
    p_mbconv3_0_weight, p_mbconv3_1_weight, p_mbconv3_1_bias,
    p_mbconv3_3_weight, p_mbconv3_4_weight, p_mbconv3_4_bias,
    p_mbconv3_7_weight, p_mbconv3_9_weight,
    p_mbconv3_11_weight, p_mbconv3_12_weight, p_mbconv3_12_bias,
    p_mbconv4_0_weight, p_mbconv4_1_weight, p_mbconv4_1_bias,
    p_mbconv4_3_weight, p_mbconv4_4_weight, p_mbconv4_4_bias,
    p_mbconv4_7_weight, p_mbconv4_9_weight,
    p_mbconv4_11_weight, p_mbconv4_12_weight, p_mbconv4_12_bias,
    p_mbconv5_0_weight, p_mbconv5_1_weight, p_mbconv5_1_bias,
    p_mbconv5_3_weight, p_mbconv5_4_weight, p_mbconv5_4_bias,
    p_mbconv5_7_weight, p_mbconv5_9_weight,
    p_mbconv5_11_weight, p_mbconv5_12_weight, p_mbconv5_12_bias,
    p_conv_final_weight, p_bn_final_weight, p_bn_final_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_mbconv1_1_running_mean, b_mbconv1_1_running_var,
    b_mbconv1_4_running_mean, b_mbconv1_4_running_var,
    b_mbconv1_12_running_mean, b_mbconv1_12_running_var,
    b_mbconv2_1_running_mean, b_mbconv2_1_running_var,
    b_mbconv2_4_running_mean, b_mbconv2_4_running_var,
    b_mbconv2_12_running_mean, b_mbconv2_12_running_var,
    b_mbconv3_1_running_mean, b_mbconv3_1_running_var,
    b_mbconv3_4_running_mean, b_mbconv3_4_running_var,
    b_mbconv3_12_running_mean, b_mbconv3_12_running_var,
    b_mbconv4_1_running_mean, b_mbconv4_1_running_var,
    b_mbconv4_4_running_mean, b_mbconv4_4_running_var,
    b_mbconv4_12_running_mean, b_mbconv4_12_running_var,
    b_mbconv5_1_running_mean, b_mbconv5_1_running_var,
    b_mbconv5_4_running_mean, b_mbconv5_4_running_var,
    b_mbconv5_12_running_mean, b_mbconv5_12_running_var,
    b_bn_final_running_mean, b_bn_final_running_var,
    x,
):
    eps = 1e-5
    
    # Initial conv + bn + relu
    x = triton_conv2d(x, p_conv1_weight, stride=(2,2), padding=(1,1))
    x = triton_bn_relu(x, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias, eps=eps, do_relu=True)
    
    # MBConv1: in=32, out=96, stride=1, expand=3 (32*3=96)
    x = mbconv_block(
        x,
        p_mbconv1_0_weight, p_mbconv1_1_weight, p_mbconv1_1_bias, b_mbconv1_1_running_mean, b_mbconv1_1_running_var,
        p_mbconv1_3_weight, p_mbconv1_4_weight, p_mbconv1_4_bias, b_mbconv1_4_running_mean, b_mbconv1_4_running_var,
        p_mbconv1_7_weight, p_mbconv1_9_weight,
        p_mbconv1_11_weight, p_mbconv1_12_weight, p_mbconv1_12_bias, b_mbconv1_12_running_mean, b_mbconv1_12_running_var,
        stride=1, expand_ratio=3, eps=eps
    )
    
    # MBConv2: in=96, out=144, stride=2, expand=6 (96*6=576)
    x = mbconv_block(
        x,
        p_mbconv2_0_weight, p_mbconv2_1_weight, p_mbconv2_1_bias, b_mbconv2_1_running_mean, b_mbconv2_1_running_var,
        p_mbconv2_3_weight, p_mbconv2_4_weight, p_mbconv2_4_bias, b_mbconv2_4_running_mean, b_mbconv2_4_running_var,
        p_mbconv2_7_weight, p_mbconv2_9_weight,
        p_mbconv2_11_weight, p_mbconv2_12_weight, p_mbconv2_12_bias, b_mbconv2_12_running_mean, b_mbconv2_12_running_var,
        stride=2, expand_ratio=6, eps=eps
    )
    
    # MBConv3: in=144, out=192, stride=2, expand=6 (144*6=864)
    x = mbconv_block(
        x,
        p_mbconv3_0_weight, p_mbconv3_1_weight, p_mbconv3_1_bias, b_mbconv3_1_running_mean, b_mbconv3_1_running_var,
        p_mbconv3_3_weight, p_mbconv3_4_weight, p_mbconv3_4_bias, b_mbconv3_4_running_mean, b_mbconv3_4_running_var,
        p_mbconv3_7_weight, p_mbconv3_9_weight,
        p_mbconv3_11_weight, p_mbconv3_12_weight, p_mbconv3_12_bias, b_mbconv3_12_running_mean, b_mbconv3_12_running_var,
        stride=2, expand_ratio=6, eps=eps
    )
    
    # MBConv4: in=192, out=288, stride=2, expand=6 (192*6=1152)
    x = mbconv_block(
        x,
        p_mbconv4_0_weight, p_mbconv4_1_weight, p_mbconv4_1_bias, b_mbconv4_1_running_mean, b_mbconv4_1_running_var,
        p_mbconv4_3_weight, p_mbconv4_4_weight, p_mbconv4_4_bias, b_mbconv4_4_running_mean, b_mbconv4_4_running_var,
        p_mbconv4_7_weight, p_mbconv4_9_weight,