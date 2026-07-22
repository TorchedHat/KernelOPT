import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm (inference) + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    C, spatial, eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute channel index: layout is (N, C, H, W) -> flat index
    # channel = (flat_idx // spatial) % C
    c_idx = (offsets // spatial) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std * w + b
    
    # ReLU
    out = tl.maximum(x_norm, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    spatial = H * W
    n_elements = N * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        C, spatial, eps,
        n_elements,
    )
    return out


# Fused BN + ReLU (no relu, just BN) for transition layers
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements', 'C'],
)
@triton.jit
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    C, spatial, eps,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    c_idx = (offsets // spatial) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std * w + b
    
    tl.store(out_ptr + offsets, x_norm, mask=mask)


def fused_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    spatial = H * W
    n_elements = N * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        C, spatial, eps,
        n_elements,
    )
    return out


# Implicit GEMM convolution kernel for 3x3 conv
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
def conv2d_3x3_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = N * OUT_H * OUT_W, N = OUT_C, K = IN_C * 3 * 3
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
        # Decode K -> (in_c, kh, kw) for 3x3 kernel
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % 3
        kh = offs_k // (IN_C * 3)
        
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = offs_m < M
        valid_k = offs_k < K
        valid = (
            valid_m[:, None] &
            valid_k[None, :] &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (batch[:, None] < N)
        )
        
        x_ptrs = X_ptr + (
            batch[:, None] * (IN_C * IN_H * IN_W) +
            in_c[None, :] * (IN_H * IN_W) +
        )
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        # Weight: (OUT_C, IN_C, 3, 3) -> (OUT_C, K)
        # w[oc, ic, kh, kw] = W_ptr[oc * K + ic * 9 + kh * 3 + kw]
        # But we need [K, OUT_C] for the dot
        w_k_idx = in_c[:, None] * 9 + kh[:, None] * 3 + kw[:, None]
        w_ptrs = W_ptr + offs_n[None, :] * K + w_k_idx
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Store output: Y[batch, out_c, out_h, out_w]
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + (
        batch[:, None] * (OUT_C * OUT_H * OUT_W) +
        offs_n[None, :] * (OUT_H * OUT_W) +
        out_h[:, None] * OUT_W +
        out_w[:, None]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def triton_conv2d_3x3(x, w, stride=1, padding=1):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    assert KH == 3 and KW == 3
    
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    STRIDE_H, STRIDE_W = stride
    PAD_H, PAD_W = padding
    
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    
    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW
    
    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    # Reshape weight to (OUT_C, K)
    w_flat = w.reshape(OUT_C, K)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_3x3_kernel[grid](
        x, w_flat, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M, K,
    )
    return y


# 1x1 conv as matrix multiply
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
def conv2d_1x1_kernel(
    X_ptr, W_ptr, Y_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        valid_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        w_ptrs = W_ptr + offs_k[:, None] * N + offs_n[None, :]
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def triton_conv2d_1x1(x, w):
    # x: (N, C, H, W), w: (OUT_C, C, 1, 1)
    N, C, H, W = x.shape
    OUT_C = w.shape[0]
    
    # Reshape to (N*H*W, C) x (C, OUT_C)
    x_flat = x.permute(0, 2, 3, 1).reshape(-1, C)  # (N*H*W, C)
    w_flat = w.reshape(OUT_C, C)  # (OUT_C, C)
    
    M = N * H * W
    K = C
    
    y_flat = torch.empty((M, OUT_C), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_1x1_kernel[grid](
        x_flat, w_flat.T.contiguous(), y_flat,
        M, OUT_C, K,
    )
    
    return y_flat.reshape(N, H, W, OUT_C).permute(0, 3, 1, 2).contiguous()


# Linear layer kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        valid_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        # W is (N, K), we need W^T effectively: W[n, k]
        w_ptrs = W_ptr + offs_n[None, :] * K + offs_k[:, None]
        valid_w = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        # w_tile is (BLOCK_K, BLOCK_N)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def triton_linear(x, w, b):
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, w, b, y, M, N, K)
    return y


# Max pool 3x3 stride 2 padding 1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Decode: (n, c, oh, ow)
    ow = offsets % OUT_W
    oh = (offsets // OUT_W) % OUT_H
    c = (offsets // (OUT_W * OUT_H)) % C
    n = offsets // (OUT_W * OUT_H * C)
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            x_val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, x_val)
    
    tl.store(Y_ptr + offsets, max_val, mask=mask)


def triton_maxpool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + 2 * padding - kernel_size) // stride + 1
    OUT_W = (IN_W + 2 * padding - kernel_size) // stride + 1
    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    n_elements = N * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride, padding, padding,
        n_elements,
    )
    return y


# Avg pool 2x2 stride 2
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def avgpool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    ow = offsets % OUT_W
    oh = (offsets // OUT_W) % OUT_H
    c = (offsets // (OUT_W * OUT_H)) % C
    n = offsets // (OUT_W * OUT_H * C)
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = KH * KW
    
    for kh in range(2):
        for kw in range(2):
            ih = oh * STRIDE_H + kh
            iw = ow * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            x_val = tl.load(X_ptr + x_idx, mask=valid, other=0.0)
            acc = acc + x_val
    
    avg = acc / count
    tl.store(Y_ptr + offsets, avg, mask=mask)


def triton_avgpool2d(x, kernel_size=2, stride=2):
    N, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H - kernel_size) // stride + 1
    OUT_W = (IN_W - kernel_size) // stride + 1
    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    n_elements = N * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    avgpool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride,
        n_elements,
    )
    return y


# Adaptive avg pool to 1x1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
    ],
    key=['n_elements', 'spatial'],
)
@triton.jit
def adaptive_avgpool_1x1_kernel(
    X_ptr, Y_ptr,
    N, C, spatial,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    c_idx = offsets % C
    n_idx = offsets // C
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Sum over spatial dimension
    base = n_idx * C * spatial + c_idx * spatial
    
    # We need to iterate over spatial positions
    # For small spatial (7x7=49), unroll manually
    for s in range(49):
        s_mask = mask & (s < spatial)
        x_val = tl.load(X_ptr + base + s, mask=s_mask, other=0.0)
        acc = acc + x_val
    
    avg = acc / spatial
    tl.store(Y_ptr + offsets, avg, mask=mask)


def triton_adaptive_avgpool_1x1(x):
    N, C, H, W = x.shape
    spatial = H * W
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    n_elements = N * C
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    adaptive_avgpool_1x1_kernel[grid](
        x, y.reshape(N * C),
        N, C, spatial,
        n_elements,
    )
    return y


def dense_layer(x, bn_w, bn_b, bn_mean, bn_var, conv_w):
    """BN + ReLU + Conv2d 3x3 + Dropout(0.0)"""
    x = fused_bn_relu(x, bn_w, bn_b, bn_mean, bn_var)
    x = triton_conv2d_3x3(x, conv_w, stride=1, padding=1)
    # dropout with p=0.0 is identity
    return x


def dense_block(x, layer_params):
    """Run a dense block with given layer parameters."""
    features = [x]
    for (bn_w, bn_b, bn_mean, bn_var, conv_w) in layer_params:
        new_feat = dense_layer(x, bn_w, bn_b, bn_mean, bn_var, conv_w)
        features.append(new_feat)
        x = torch