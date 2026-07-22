import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

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
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Sigmoid pointwise kernel
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
def sigmoid_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + libdevice.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# ReLU pointwise kernel
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
def relu_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Adaptive Average Pool 2D (global) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 32}, num_warps=2),
    ],
    key=['HW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
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
# 1x1 Convolution as GEMM kernel
# Input: (N, C_in, H, W) -> treat as (N*H*W, C_in) x (C_in, C_out)
# Output: (N, C_out, H, W)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        mask_x = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        w_ptrs = w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        mask_w = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x, w, acc, allow_tf32=False)
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


# ============================================================
# 3x3 Depthwise Convolution kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 16}, num_warps=8),
    ],
    key=['M', 'C', 'H_in', 'W_in'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
    M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    # Decode m -> (n, h_out, w_out)
    w_out_idx = offs_m % W_out
    h_out_idx = (offs_m // W_out) % H_out
    n_idx = offs_m // (H_out * W_out)
    
    mask_m = offs_m < N * H_out * W_out
    mask_c = offs_c < C
    
    acc = tl.zeros([BLOCK_M, BLOCK_C], dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            h_in = h_out_idx * stride_h + kh - pad_h
            w_in = w_out_idx * stride_w + kw - pad_w
            
            valid = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
            
            # Load input: x[n, c, h_in, w_in]
            x_idx = n_idx * C * H_in * W_in + offs_c[None, :] * H_in * W_in + h_in[:, None] * W_in + w_in[:, None]
            x_mask = mask_m[:, None] & mask_c[None, :] & valid[:, None]
            x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0).to(tl.float32)
            
            # Load weight: w[c, 1, kh, kw] -> w[c, kh, kw]
            w_idx = offs_c * 9 + kh * 3 + kw
            w_mask = mask_c
            wt = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0).to(tl.float32)
            
            acc += x * wt[None, :]
    
    # Store output: out[n, c, h_out, w_out]
    out_idx = n_idx[:, None] * C * H_out * W_out + offs_c[None, :] * H_out * W_out + h_out_idx[:, None] * W_out + w_out_idx[:, None]
    out_mask = mask_m[:, None] & mask_c[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


# ============================================================
# 3x3 Standard Convolution kernel (for first conv)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N_batch, C_in, H_in, W_in,
    C_out, H_out, W_out,
    stride_h, stride_w,
    pad_h, pad_w,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m -> (n, h_out, w_out)
    w_out_idx = offs_m % W_out
    h_out_idx = (offs_m // W_out) % H_out
    n_idx = offs_m // (H_out * W_out)
    
    mask_m = offs_m < M
    mask_n = offs_n < N
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    TOTAL_K = C_in * KERNEL_H * KERNEL_W
    
    for k in range(0, TOTAL_K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Decode k -> (c_in, kh, kw)
        kw_idx = offs_k % KERNEL_W
        kh_idx = (offs_k // KERNEL_W) % KERNEL_H
        c_in_idx = offs_k // (KERNEL_H * KERNEL_W)
        
        h_in = h_out_idx[:, None] * stride_h + kh_idx[None, :] - pad_h
        w_in = w_out_idx[:, None] * stride_w + kw_idx[None, :] - pad_w
        
        valid = (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in) & (offs_k[None, :] < TOTAL_K)
        
        x_idx = n_idx[:, None] * C_in * H_in * W_in + c_in_idx[None, :] * H_in * W_in + h_in * W_in + w_in
        x_mask = mask_m[:, None] & valid
        x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
        
        # Weight: w[c_out, c_in, kh, kw]
        w_idx = offs_n[None, :] * C_in * KERNEL_H * KERNEL_W + c_in_idx[:, None] * KERNEL_H * KERNEL_W + kh_idx[:, None] * KERNEL_W + kw_idx[:, None]
        w_mask = mask_n[None, :] & (offs_k[:, None] < TOTAL_K)
        wt = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, wt, acc, allow_tf32=False)
    
    out_idx = n_idx[:, None] * C_out * H_out * W_out + offs_n[None, :] * H_out * W_out + h_out_idx[:, None] * W_out + w_out_idx[:, None]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


# ============================================================
# Linear (GEMM) kernel: out = x @ W^T + b
# x: (M, K), W: (N, K), b: (N,), out: (M, N)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
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
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        
        x_ptrs = x_ptr + offs_m[:, None] * K + k_offs[None, :]
        mask_x = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # W is (N, K), so W^T is (K, N)
        w_ptrs = w_ptr + offs_n[None, :] * K + k_offs[:, None]
        mask_w = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x, w, acc, allow_tf32=False)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += b[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=mask_out)


# ============================================================
# Helper functions
# ============================================================

def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    """Apply batch norm + optional relu"""
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def triton_sigmoid(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


def triton_relu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def triton_adaptive_avg_pool2d(x):
    """Global average pooling to (N, C, 1, 1)"""
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    grid = (N * C,)
    adaptive_avg_pool2d_kernel[grid](
        x, out.view(N, C),
    )
    return out


def triton_conv1x1(x, w):
    """1x1 convolution: x (N, C_in, H, W), w (C_out, C_in, 1, 1)"""
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    M = N * H * W
    
    # Reshape x to (M, C_in) and w to (C_in, C_out)
    x_2d = x.permute(0, 2, 3, 1).contiguous().view(M, C_in)
    w_2d = w.view(C_out, C_in).t().contiguous()  # (C_in, C_out)
    
    out_2d = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )
    
    conv1x1_kernel[grid](
        x_2d, w_2d, out_2d,
        M, C_out, C_in,
    )
    
    return out_2d.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def triton_depthwise_conv3x3(x, w, stride=1, padding=1):
    """Depthwise 3x3 conv: x (N, C, H, W), w (C, 1, 3, 3)"""
    N, C, H_in, W_in = x.shape
    H_out = (H_in + 2 * padding - 3) // stride + 1
    W_out = (W_in + 2 * padding - 3) // stride + 1
    
    out = torch.empty(N, C, H_out, W_out, device=x.device, dtype=x.dtype)
    
    M = N * H_out * W_out
    
    # Make weight contiguous: (C, 9)
    w_flat = w.view(C, 9).contiguous()
    
    BLOCK_C = 32
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C, meta['BLOCK_C']),
    )
    
    depthwise_conv3x3_kernel[grid](
        x, w_flat, out,
        N, C, H_in, W_in, H_out, W_out,
        stride, stride,
        padding, padding,
        M=M,
    )
    
    return out


def triton_conv3x3(x, w, stride=2, padding=1):
    """Standard 3x3 conv: x (N, C_in, H, W), w (C_out, C_in, 3, 3)"""
    N, C_in, H_in, W_in = x.shape
    C_out = w.shape[0]
    H_out = (H_in + 2 * padding - 3) // stride + 1
    W_out = (W_in + 2 * padding - 3) // stride + 1
    
    M = N * H_out * W_out
    K = C_in * 3 * 3
    
    out = torch.empty(N, C_out, H_out, W_out, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x, w, out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        stride, stride,
        padding, padding,
        3, 3,
        M, C_out, K,
    )
    
    return out


def triton_linear(x, w, b):
    """Linear: x (M, K), w (N, K), b (N,) -> out (M, N)"""
    M, K = x.shape
    N = w.shape[0]
    
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    linear_kernel[grid](x, w, b, out, M, N, K)
    
    return out


# ============================================================
# Main kernel function implementing EfficientNetB2 forward
# ============================================================

def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_mbconv1_0_weight, p_mbconv1_1_weight, p_mbconv1_1_bias,
    p_mbconv1_3_weight, p_mbconv1_4_weight, p_mbconv1_4_bias,
    p_mbconv1_7_weight, p_mbconv1_9_weight, p_mbconv1_11_weight,
    p_mbconv1_12_weight, p_mbconv1_12_bias,
    p_mbconv2_0_weight, p_mbconv2_1_weight, p_mbconv2_1_bias,
    p_mbconv2_3_weight, p_mbconv2_4_weight, p_mbconv2_4_bias,
    p_mbconv2_7_weight, p_mbconv2_9_weight, p_mbconv2_11_weight,
    p_mbconv2_12_weight, p_mbconv2_12_bias,
    p_mbconv3_0_weight, p_mbconv3_1_weight, p_mbconv3_1_bias,
    p_mbconv3_3_weight, p_mbconv3_4_weight, p_mbconv3_4_bias,