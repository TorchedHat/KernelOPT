import torch
import triton
import triton.language as tl
import math

# [Full implementation follows]


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math

# Fused BatchNorm + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    relu: tl.constexpr,
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
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    if relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


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
def relu_kernel(
    x_ptr, out_ptr,
    numel,
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
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def add_kernel(
    a_ptr, b_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


# Channel shuffle kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'groups', 'channels_per_group', 'HW'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    numel,
    groups, channels_per_group, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # offsets: flat index into (N, C, H, W) tensor
    # C = groups * channels_per_group
    C = groups * channels_per_group
    hw_idx = offsets % HW
    c_idx = (offsets // HW) % C
    n_idx = offsets // (HW * C)
    
    # channel shuffle: c_idx = g * cpg + c_in_group
    # after shuffle: new_c = c_in_group * groups + g
    g = c_idx // channels_per_group
    c_in_group = c_idx % channels_per_group
    new_c = c_in_group * groups + g
    
    src_offset = hw_idx + new_c * HW + n_idx * C * HW
    x = tl.load(x_ptr + src_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


# Max pool 2D kernel (3x3, stride=2, padding=1)
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
    N, C, IH, IW, OH, OW,
    KH, KW, SH, SW, PH, PW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH - PH + kh
            iw = ow * SW - PW + kw
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            src = n * C * IH * IW + c * IH * IW + ih * IW + iw
            val = tl.load(x_ptr + src, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


# Adaptive avg pool (global) kernel
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
    nc_idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = nc_idx < NC
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for hw in range(HW):
        val = tl.load(x_ptr + nc_idx * HW + hw, mask=mask, other=0.0)
        acc = acc + val
    
    out = acc / HW
    tl.store(out_ptr + nc_idx, out, mask=mask)


# Linear kernel (GEMM): out = x @ W.T + b
# x: (M, K), W: (N, K), b: (N,), out: (M, N)
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
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
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
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)
    
    b = tl.load(b_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + b[None, :]
    
    tl.store(out_ptr + offs_m_out[:, None] * N + offs_n_out[None, :], acc, mask=out_mask)


# Implicit GEMM conv2d kernel for 1x1 convolutions
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """1x1 convolution as GEMM: x(M, K) @ w(K, N) -> out(M, N)
    M = N_batch * H * W, K = C_in, N = C_out
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :], mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# Group conv1x1 kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N_per_group', 'K_per_group', 'groups'],
)
@triton.jit
def group_conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N_per_group, K_per_group, groups,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Group 1x1 convolution.
    """
    pid_m = tl.program_id(0)
    pid_gn = tl.program_id(1)
    
    g = pid_gn // tl.cdiv(N_per_group, BLOCK_N)
    pid_n = pid_gn % tl.cdiv(N_per_group, BLOCK_N)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    K_total = groups * K_per_group
    N_total = groups * N_per_group
    
    for k in range(0, K_per_group, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # x is (M, K_total), we want group g's channels: g*K_per_group + offs_k
        x_col = g * K_per_group + offs_k
        x_mask = (offs_m[:, None] < M) & (x_col[None, :] < K_total) & (offs_k[None, :] < K_per_group)
        x = tl.load(x_ptr + offs_m[:, None] * K_total + x_col[None, :], mask=x_mask, other=0.0)
        
        # w is (groups*N_per_group, K_per_group), group g's rows: g*N_per_group + offs_n
        w_row = g * N_per_group + offs_n
        w_mask = (w_row[None, :] < N_total) & (offs_k[:, None] < K_per_group)
        w = tl.load(w_ptr + w_row[None, :] * K_per_group + offs_k[:, None], mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    out_col = g * N_per_group + offs_n
    out_mask = (offs_m[:, None] < M) & (out_col[None, :] < N_total)
    tl.store(out_ptr + offs_m[:, None] * N_total + out_col[None, :], acc, mask=out_mask)


# Depthwise conv 3x3 kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64}, num_warps=8),
    ],
    key=['N', 'C', 'IH', 'IW', 'OH', 'OW'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    SH, SW, PH, PW,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    """Depthwise 3x3 conv. x: (N, C, IH, IW), w: (C, 1, 3, 3), out: (N, C, OH, OW)"""
    pid_n = tl.program_id(2)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(0)
    
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    
    oh = offs_hw // OW
    ow = offs_hw % OW
    
    c_mask = offs_c < C
    hw_mask = offs_hw < (OH * OW)
    
    acc = tl.zeros([BLOCK_C, BLOCK_HW], dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = oh[None, :] * SH - PH + kh
            iw = ow[None, :] * SW - PW + kw
            valid = hw_mask[None, :] & c_mask[:, None] & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            
            x_idx = pid_n * C * IH * IW + offs_c[:, None] * IH * IW + ih * IW + iw
            x = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            
            w_idx = offs_c * 9 + kh * 3 + kw
            w = tl.load(w_ptr + w_idx, mask=c_mask, other=0.0)
            
            acc = acc + x * w[:, None]
    
    out_idx = pid_n * C * OH * OW + offs_c[:, None] * OH * OW + oh[None, :] * OW + ow[None, :]
    out_mask = c_mask[:, None] & hw_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def channel_shuffle(x, groups):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    channels_per_group = C // groups
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
    )
    return out


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, IH, IW = x.shape
    OH = (IH + 2 * padding - kernel_size) // stride + 1
    OW = (IW + 2 * padding - kernel_size) // stride + 1
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    numel = out.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out, N, C, IH, IW, OH, OW,
        kernel_size, kernel_size, stride, stride, padding, padding,
    )
    return out


def triton_adaptive_avg_pool(x):
    N, C, H, W = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    NC = N * C
    HW = H * W
    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_SIZE']),)
    adaptive_avg_pool_kernel[grid](
    )
    return out


def triton_linear(x, weight, bias):
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


def triton_conv1x1(x, weight):
    """x: (N, C_in, H, W), weight: (C_out, C_in, 1, 1)"""
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    M = N * H * W
    K = C_in
    
    # Reshape x to (M, K) - need contiguous NHWC-like layout
    x_nhwc = x.permute(0, 2, 3, 1).contiguous().view(M, K)
    w_flat = weight.view(C_out, K)  # (C_out, K)
    
    out = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv1x1_kernel[grid](
        x_nhwc, w_flat.t().contiguous(), out,
        M, C_out, K,
    )
    return out.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def triton_group_conv1x1(x, weight, groups):
    """x: (N, C_in, H, W), weight: (C_out, C_in//groups, 1, 1)"""
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    K_per_group = C_in // groups
    N_per_group = C_out // groups
    M = N * H * W
    
    x_nhwc = x.permute(0, 2, 3, 1).contiguous().view(M, C_in)
    # weight: (C_out, K_per_group, 1, 1) -> (C_out, K_per_group)
    w_flat = weight.view(C_out, K_per_group)
    
    out = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
    
    blocks_n = triton.cdiv(N_per_group, 64)  # default BLOCK_N
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), groups * triton.cdiv(N_per_group, meta['BLOCK_N']))
    group_conv1x1_kernel[grid](
        x_nhwc, w_flat, out,
        M, N_per_group, K_per_group, groups,
    )
    return out.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def triton_depthwise_conv3x3(x, weight, stride=1, padding=1):
    """x: (N, C, IH, IW), weight: (C, 1, 3, 3)"""
    N, C, IH, IW = x.shape
    OH = (IH + 2 * padding - 3) // stride + 1
    OW = (IW + 2 * padding - 3) // stride + 1
    
    x_c = x.contiguous()
    w_c = weight.contiguous().view(C, 9)
    
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)