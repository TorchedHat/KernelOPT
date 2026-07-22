import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm + ReLU kernel (inference mode)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    # Layout: (N, C, H, W) -> channel = (offset / spatial) % num_channels
    chan = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=0.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    chan = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=0.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Implicit GEMM convolution kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
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
    s_wn, s_wc, s_wh, s_ww,
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
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw
        
        x_ptrs = X_ptr + (
            batch_idx[:, None] * s_xn +
            in_c[None, :] * s_xc +
        )
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # Weight: (OUT_C, IN_C, KH, KW)
        valid_k2 = offs_k[:, None] < K
        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k2 & valid_n
        
        w_ptrs = W_ptr + (
            offs_n[None, :] * s_wn +
            in_c[:, None] * s_wc +
            kh[:, None] * s_wh +
            kw[:, None] * s_ww
        )
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store output: Y is (N, OUT_C, OUT_H, OUT_W)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + (
        batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
        offs_n[None, :] * (OUT_H * OUT_W) +
        out_h_idx[:, None] * OUT_W +
        out_w_idx[:, None]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# Linear kernel (GEMM: M x K @ K x N)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=4),
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
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        x = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        w = tl.load(W_ptr + offs_k[:, None] * N + offs_n[None, :], mask=mask_w, other=0.0)
        
        acc = tl.dot(x, w, acc, allow_tf32=False)
    
    # Add bias
    mask_b = offs_n < N
    bias = tl.load(B_ptr + offs_n, mask=mask_b, other=0.0)
    acc = acc + bias[None, :]
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def triton_bn(x, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def triton_conv2d(x, w, stride=(1,1), padding=(0,0)):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    STRIDE_H, STRIDE_W = stride
    PAD_H, PAD_W = padding
    
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    
    M = N * OUT_H * OUT_W
    N_out = OUT_C
    K = IN_C * KH * KW
    
    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_out, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x, w, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M, N_out, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
    )
    return y


def triton_linear(x, w, b):
    # x: (M, K), w: (N, K) -> y: (M, N)
    M, K = x.shape
    N = w.shape[0]
    w_t = w.contiguous()  # (N, K)
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    # We need W transposed: (K, N)
    # linear: y = x @ w.T + b
    # So we need to compute x(M,K) @ w.T(K,N)
    # In our kernel, W_ptr stores row-major (K, N) after transpose
    w_t2 = w_t.T.contiguous()  # (K, N)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_kernel[grid](
        x, w_t2, b, y,
        M, N, K,
    )
    return y


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    # Use PyTorch for max pool (no Triton implementation needed for correctness)
    # But we need to implement it in Triton per rules... 
    # Actually max_pool2d is not in the banned list explicitly
    # Let me implement it simply
    N, C, H, W = x.shape
    OH = (H + 2 * padding - kernel_size) // stride + 1
    OW = (W + 2 * padding - kernel_size) // stride + 1
    
    # Unfold and max
    x_padded = torch.nn.functional.pad(x, (padding, padding, padding, padding), value=float('-inf'))
    # Use unfold
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    
    # Simple implementation using Triton
    _max_pool2d_triton(x_padded, out, N, C, H, W, OH, OW, kernel_size, stride, padding)
    return out


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
    N, C, IN_H_PAD, IN_W_PAD,
    OH, OW,
    KH, KW, STRIDE_H, STRIDE_W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    # Decode: (N, C, OH, OW)
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C
    
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh
            iw = ow * STRIDE_W + kw
            x_idx = n * (C * IN_H_PAD * IN_W_PAD) + c * (IN_H_PAD * IN_W_PAD) + ih * IN_W_PAD + iw
            val = tl.load(x_ptr + x_idx, mask=mask, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + offs, max_val, mask=mask)


def _max_pool2d_triton(x_padded, out, N, C, H, W, OH, OW, kernel_size, stride, padding):
    IN_H_PAD = x_padded.shape[2]
    IN_W_PAD = x_padded.shape[3]
    numel = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x_padded, out,
        N, C, IN_H_PAD, IN_W_PAD,
        OH, OW,
        kernel_size, kernel_size, stride, stride,
        numel,
    )


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
def avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W,
    OH, OW,
    KH, KW, STRIDE_H, STRIDE_W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = KH * KW
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh
            iw = ow * STRIDE_W + kw
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
            acc = acc + val
    
    tl.store(out_ptr + offs, acc / count, mask=mask)


def triton_avg_pool2d(x, kernel_size=2, stride=2):
    N, C, H, W = x.shape
    OH = (H - kernel_size) // stride + 1
    OW = (W - kernel_size) // stride + 1
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    numel = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    avg_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        OH, OW,
        kernel_size, kernel_size, stride, stride,
        numel,
    )
    return out


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
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    # Output is (N, C, 1, 1) - global average pooling
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    # offs indexes (N, C)
    c = offs % C
    n = offs // C
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    spatial = IN_H * IN_W
    
    # Sum over all spatial positions
    for i in range(IN_H):
        for j in range(IN_W):
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + i * IN_W + j
            val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
            acc = acc + val
    
    tl.store(out_ptr + offs, acc / spatial, mask=mask)


def triton_adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    numel = N * C
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        numel,
    )
    return out


def dense_layer(x, bn_w, bn_b, bn_mean, bn_var, conv_w, eps=1e-5):
    """BN + ReLU + Conv2d + Dropout(0.0)"""
    # BN + ReLU
    out = triton_bn_relu(x, bn_mean, bn_var, bn_w, bn_b, eps)
    # Conv2d (3x3, padding=1)
    out = triton_conv2d(out, conv_w, stride=(1,1), padding=(1,1))
    # Dropout(0.0) is identity
    return out


def transition_layer(x, bn_w, bn_b, bn_mean, bn_var, conv_w, eps=1e-5):
    """BN + ReLU + Conv2d(1x1) + AvgPool2d(2,2)"""
    out = triton_bn_relu(x, bn_mean, bn_var, bn_w, bn_b, eps)
    out = triton_conv2d(out, conv_w, stride=(1,1), padding=(0,0))
    out = triton_avg_pool2d(out, kernel_size=2, stride=2)
    return out


def kernel_function(
    p_features_0_weight,
    p_features_1_weight,
    p_features_1_bias,
    p_dense_blocks_0_layers_0_0_weight,
    p_dense_blocks_0_layers_0_0_bias,
    p_dense_blocks_0_layers_0_2_weight,
    p_dense_blocks_0_layers_1_0_weight,
    p_dense_blocks_0_layers_1_0_bias,
    p_dense_blocks_0_layers_1_2_weight,
    p_dense_blocks_0_layers_2_0_weight,
    p_dense_blocks_0_layers_2_0_bias,
    p_dense_blocks_0_layers_2_2_weight,
    p_dense_blocks_0_layers_3_0_weight,
    p_dense_blocks_0_layers_3_0_bias,
    p_dense_blocks_0_layers_3_2_weight,
    p_dense_blocks_0_layers_4_0_weight,
    p_dense_blocks_0_layers_4_0_bias,
    p_dense_blocks_0_layers_4_2_weight,
    p_dense_blocks_0_layers_5_0_weight,
    p_dense_blocks_0_layers_5_0_bias,
    p_dense_blocks_0_layers_5_2_weight,
    p_transition_layers_0_transition_0_weight,
    p_transition_layers_0_transition_0_bias,
    p_transition_layers_0_transition_2_weight,
    p_dense_blocks_1_layers_0_0_weight,
    p_dense_blocks_1_layers_0_0_bias,
    p_dense_blocks_1_layers_0_2_weight,
    p_dense_blocks_1_layers_1_0_weight,
    p_dense_blocks_1_layers_1_0_bias,
    p_dense_blocks_1_layers_1_2_weight,
    p_dense_blocks_1_layers_2_0_weight,
    p_dense_blocks_1_layers_2_0_bias,
    p_dense_blocks_1_layers_2_2_weight,
    p_dense_blocks_1_layers_3_0_weight,
    p_dense_blocks_1_layers_3_0_bias,
    p_dense_blocks_1_layers_3_2_weight,
    p_dense_blocks_1_layers_4_0_weight,
    p_dense_blocks_1_layers_4_0_bias,
    p_dense_blocks_1_layers_4_2_weight,
    p_dense_blocks_1_layers_5_0_weight,
    p_dense_blocks_1_layers_5_0_bias,
    p_dense_blocks_1_layers_5_2_weight,
    p_dense_blocks_1_layers_6_0_weight,
    p_dense_blocks_1_layers_6_0_bias,
    p_dense_blocks_1_layers_6_2_weight,
    p_dense_blocks_1_layers_7_0_weight,
    p_dense_blocks_1_layers_7_0_bias,
    p_dense_blocks_1_layers_7_2_weight,
    p_dense_blocks_1_layers_8_0_weight,
    p_dense_blocks_1_layers_8_0_bias,
    p_dense_blocks_1_layers_8_2_weight,
    p_dense_blocks_1_layers_9_0_weight,
    p_dense_blocks_1_layers_9_0_bias,
    p_dense_blocks_1_layers_9_2_weight,
    p_dense_blocks_1_layers_10_0_weight,
    p_dense_blocks_1_layers_10_0_bias,
    p_dense_blocks_1_layers_10_2_weight,
    p_dense_blocks_1_layers_11_0_weight,
    p_dense_blocks_1_layers_11_0_bias,
    p_dense_blocks_1_layers_11_2_weight,
    p_transition_layers_1_transition_0_weight,