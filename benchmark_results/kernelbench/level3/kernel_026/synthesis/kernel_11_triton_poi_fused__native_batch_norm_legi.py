import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Fused BatchNorm (inference) + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps: tl.constexpr,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    if apply_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def bn_relu(x, mean, var, weight, bias, eps=1e-5, relu=True):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
        numel, num_channels, spatial,
        eps=eps, apply_relu=relu,
    )
    return out


# Channel shuffle kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
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
    
    # Decode output index: (n, c_out, h, w)
    w_idx = offsets % W
    tmp = offsets // W
    h_idx = tmp % H
    tmp = tmp // H
    c_out = tmp % C
    n_idx = tmp // C
    
    # c_out = g * channels_per_group + c_in_group
    # After shuffle: c_in = c_in_group * groups + g
    g = c_out % groups
    c_in_group = c_out // groups
    c_in = c_in_group * groups + g
    
    in_offset = n_idx * C * H * W + c_in * H * W + h_idx * W + w_idx
    
    val = tl.load(x_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def channel_shuffle(x, groups):
    N, C, H, W = x.shape
    channels_per_group = C // groups
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
    )
    return out


# Fused add + BN + ReLU kernel (for residual connections)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_add_kernel(
    x_ptr, residual_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b + res
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def bn_add(x, residual, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_add_kernel[grid](
        x, residual, out, mean, var, weight, bias,
        numel, num_channels, spatial, eps=eps,
    )
    return out


# Linear kernel using tl.dot
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
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


# Max pool kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    KH, KW, SH, SW, PH, PW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    numel = N * C * OH * OW
    mask = offsets < numel
    
    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH - PH + kh
            iw = ow * SW - PW + kw
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            in_offset = n * C * IH * IW + c * IH * IW + ih * IW + iw
            val = tl.load(x_ptr + in_offset, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, IH, IW = x.shape
    OH = (IH + 2 * padding - kernel_size) // stride + 1
    OW = (IW + 2 * padding - kernel_size) // stride + 1
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    numel = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out, N, C, IH, IW, OH, OW,
        kernel_size, kernel_size, stride, stride, padding, padding,
    )
    return out


# Adaptive avg pool kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N', 'C', 'IH', 'IW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IH, IW,
    BLOCK_SIZE: tl.constexpr,
):
    # Output is (N, C, 1, 1) - global average pooling
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    numel = N * C
    mask = offsets < numel
    
    c = offsets % C
    n = offsets // C
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    spatial = IH * IW
    
    for h in range(IH):
        for w in range(IW):
            in_offset = n * C * IH * IW + c * IH * IW + h * IW + w
            val = tl.load(x_ptr + in_offset, mask=mask, other=0.0).to(tl.float32)
            acc = acc + val
    
    acc = acc / spatial.to(tl.float32)
    tl.store(out_ptr + offsets, acc, mask=mask)


def triton_adaptive_avg_pool2d(x):
    N, C, IH, IW = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    numel = N * C
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, out, N, C, IH, IW,
    )
    return out


# 1x1 Conv2d as GEMM using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_conv1x1(x, weight, groups=1):
    """1x1 grouped convolution as GEMM"""
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    
    # Reshape to (N*H*W, C_in) for GEMM
    x_flat = x.permute(0, 2, 3, 1).contiguous().view(N * H * W, C_in)
    # weight: (C_out, C_in/groups, 1, 1) -> (C_out, C_in/groups)
    w_flat = weight.view(C_out, -1)  # (C_out, C_in/groups)
    
    M = N * H * W
    K_per_group = C_in // groups
    
    out_flat = torch.empty((M, C_out), device=x.device, dtype=x.dtype)
    
    if groups == 1:
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
        conv1x1_kernel[grid](
            x_flat, w_flat, out_flat,
            M, C_out, C_in,
            x_flat.stride(0), x_flat.stride(1),
            w_flat.stride(0), w_flat.stride(1),
        )
    else:
        # Process each group
        group_in = C_in // groups
        group_out = C_out // groups
        for g in range(groups):
            x_g = x_flat[:, g*group_in:(g+1)*group_in].contiguous()
            w_g = w_flat[g*group_out:(g+1)*group_out, :].contiguous()
            out_g = out_flat[:, g*group_out:(g+1)*group_out]
            grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(group_out, meta['BLOCK_N']))
            conv1x1_kernel[grid](
                x_g, w_g, out_g,
                M, group_out, group_in,
                x_g.stride(0), x_g.stride(1),
                w_g.stride(0), w_g.stride(1),
            )
    
    return out_flat.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


# Depthwise 3x3 conv kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    stride_h, stride_w, pad_h, pad_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    numel = N * C * OH * OW
    mask = offsets < numel
    
    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = oh * stride_h - pad_h + kh
            iw = ow * stride_w - pad_w + kw
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            in_offset = n * C * IH * IW + c * IH * IW + ih * IW + iw
            w_offset = c * 9 + kh * 3 + kw
            val = tl.load(x_ptr + in_offset, mask=valid, other=0.0).to(tl.float32)
            w_val = tl.load(w_ptr + w_offset, mask=valid, other=0.0).to(tl.float32)
            acc = acc + val * w_val
    
    tl.store(out_ptr + offsets, acc, mask=mask)


def triton_depthwise_conv3x3(x, weight, stride=1, padding=1):
    N, C, IH, IW = x.shape
    OH = (IH + 2 * padding - 3) // stride + 1
    OW = (IW + 2 * padding - 3) // stride + 1
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    numel = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    depthwise_conv3x3_kernel[grid](
        x, weight, out,
        N, C, IH, IW, OH, OW,
        stride, stride, padding, padding,
    )
    return out


def shufflenet_unit(x, 
                    conv1_w, bn1_mean, bn1_var, bn1_w, bn1_b,
                    conv2_w, bn2_mean, bn2_var, bn2_w, bn2_b,
                    conv3_w, bn3_mean, bn3_var, bn3_w, bn3_b,
                    shortcut_conv_w=None, shortcut_bn_mean=None, shortcut_bn_var=None,
                    shortcut_bn_w=None, shortcut_bn_b=None,
                    groups=3):
    # conv1: 1x1 group conv
    out = triton_conv1x1(x, conv1_w, groups=groups)
    out = bn_relu(out, bn1_mean, bn1_var, bn1_w, bn1_b, relu=True)
    
    # conv2: depthwise 3x3
    out = triton_depthwise_conv3x3(out, conv2_w, stride=1, padding=1)
    out = bn_relu(out, bn2_mean, bn2_var, bn2_w, bn2_b, relu=False)
    
    # channel shuffle
    out = channel_shuffle(out, groups)
    
    # conv3: 1x1 group conv
    out = triton_conv1x1(out, conv3_w, groups=groups)
    
    # shortcut
    if shortcut_conv_w is not None:
        shortcut = triton_conv1x1(x, shortcut_conv_w, groups=1)
        shortcut = bn_relu(shortcut, shortcut_bn_mean, shortcut_bn_var, shortcut_bn_w, shortcut_bn_b, relu=False)
    else:
        shortcut = x
    
    # BN + add + relu
    out = bn_add(out, shortcut, bn3_mean, bn3_var, bn3_w, bn3_b)
    out = tl_relu(out)
    
    return out