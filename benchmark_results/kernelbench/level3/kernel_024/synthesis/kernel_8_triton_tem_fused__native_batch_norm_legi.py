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
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if apply_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def bn_relu(x, mean, var, weight, bias, eps=1e-5, apply_relu=True):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


# ============================================================
# Fused Sigmoid pointwise kernel
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
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_sigmoid(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


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


def triton_relu(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


# ============================================================
# Adaptive Average Pool 2D (global) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
    ],
    key=['NC', 'HW'],
)
@triton.jit
def adaptive_avg_pool2d_global_kernel(
    x_ptr, out_ptr,
    NC, HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (N, C) pair
    pid = tl.program_id(0)
    if pid >= NC:
        return
    
    base = pid * HW
    acc = tl.zeros([1], dtype=tl.float32)
    
    for off in range(0, HW, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < HW
        x = tl.load(x_ptr + base + offsets, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    
    tl.store(out_ptr + pid, acc / HW)


def triton_adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    NC = N * C
    HW = H * W
    grid = (NC,)
    adaptive_avg_pool2d_global_kernel[grid](
        x.contiguous(), out, NC, HW
    )
    return out


# ============================================================
# Linear (GEMM) kernel using tl.dot
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
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
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, acc, mask=out_mask)


def triton_linear(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
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
# Conv2d as implicit GEMM with fused BN+ReLU epilogue
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M_total', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_bn_relu_kernel(
    X_ptr, W_ptr, out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    M_total, TOTAL_K,
    eps: tl.constexpr,
    apply_relu: tl.constexpr,
    has_bn: tl.constexpr,
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
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M_total
        valid_k = offs_k[None, :] < TOTAL_K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < BATCH
        x_mask = valid_m & valid_k & valid_h & valid_w & valid_batch
        
        # X in NCHW: stride = (IN_C*IN_H*IN_W, IN_H*IN_W, IN_W, 1)
        x_ptrs = X_ptr + (
            batch_idx[:, None] * (IN_C * IN_H * IN_W) +
            in_c[None, :] * (IN_H * IN_W) +
        )
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # W in (OUT_C, IN_C, KH, KW): stride = (IN_C*KH*KW, KH*KW, KW, 1)
        valid_wk = offs_k[:, None] < TOTAL_K
        valid_wn = offs_n[None, :] < OUT_C
        w_ptrs = W_ptr + (
            offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) +
            in_c[:, None] * (KERNEL_H * KERNEL_W) +
            kh[:, None] * KERNEL_W +
            kw[:, None]
        )
        w_tile = tl.load(w_ptrs, mask=valid_wk & valid_wn, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Apply BN + ReLU
    if has_bn:
        bn_mean = tl.load(mean_ptr + offs_n, mask=offs_n < OUT_C, other=0.0).to(tl.float32)
        bn_var = tl.load(var_ptr + offs_n, mask=offs_n < OUT_C, other=1.0).to(tl.float32)
        bn_gamma = tl.load(gamma_ptr + offs_n, mask=offs_n < OUT_C, other=1.0).to(tl.float32)
        bn_beta = tl.load(beta_ptr + offs_n, mask=offs_n < OUT_C, other=0.0).to(tl.float32)
        
        acc = (acc - bn_mean[None, :]) / tl.sqrt(bn_var[None, :] + eps)
        acc = acc * bn_gamma[None, :] + bn_beta[None, :]
    
    if apply_relu:
        acc = tl.maximum(acc, 0.0)
    
    # Store in NCHW: out[n, c, h, w] = out[n*OUT_C*OUT_H*OUT_W + c*OUT_H*OUT_W + h*OUT_W + w]
    out_mask = (offs_m[:, None] < M_total) & (offs_n[None, :] < OUT_C)
    out_ptrs = out_ptr + (
        batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
        offs_n[None, :] * (OUT_H * OUT_W) +
        out_h_idx[:, None] * OUT_W +
        out_w_idx[:, None]
    )
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_conv2d_bn_relu(x, weight, mean, var, gamma, beta, stride, padding, eps=1e-5, apply_relu=True, has_bn=True):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    SH, SW = stride if isinstance(stride, (list, tuple)) else (stride, stride)
    PH, PW = padding if isinstance(padding, (list, tuple)) else (padding, padding)
    
    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1
    
    M_total = N * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW
    
    out = torch.empty(N, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M_total, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    
    if mean is None:
        mean = torch.zeros(OUT_C, device=x.device, dtype=x.dtype)
        var = torch.ones(OUT_C, device=x.device, dtype=x.dtype)
        gamma = torch.ones(OUT_C, device=x.device, dtype=x.dtype)
        beta = torch.zeros(OUT_C, device=x.device, dtype=x.dtype)
    
    conv2d_bn_relu_kernel[grid](
        x.contiguous(), weight.contiguous(), out,
        mean, var, gamma, beta,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        M_total, TOTAL_K,
        eps, apply_relu, has_bn,
    )
    return out


# ============================================================
# Depthwise Conv2d with fused BN+ReLU
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 32}, num_warps=8),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 128}, num_warps=4),
    ],
    key=['C', 'OUT_HW', 'KH', 'KW'],
)
@triton.jit
def depthwise_conv2d_bn_relu_kernel(
    X_ptr, W_ptr, out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    OUT_HW,
    eps: tl.constexpr,
    apply_relu: tl.constexpr,
    has_bn: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_c = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    
    out_w_idx = offs_hw % OUT_W
    out_h_idx = offs_hw // OUT_W
    
    c_mask = offs_c < C
    hw_mask = offs_hw < OUT_HW
    
    # Load BN params
    if has_bn:
        bn_mean = tl.load(mean_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
        bn_var = tl.load(var_ptr + offs_c, mask=c_mask, other=1.0).to(tl.float32)
        bn_gamma = tl.load(gamma_ptr + offs_c, mask=c_mask, other=1.0).to(tl.float32)
        bn_beta = tl.load(beta_ptr + offs_c, mask=c_mask, other=0.0).to(tl.float32)
    
    # For each output position and channel, compute depthwise conv
    for c_off in range(BLOCK_C):
        c_idx = pid_c * BLOCK_C + c_off
        if c_idx >= C:
            break
        
        acc = tl.zeros([BLOCK_HW], dtype=tl.float32)
        
        for kh in range(KH):
            for kw in range(KW):
                ih = out_h_idx * SH + kh - PH
                iw = out_w_idx * SW + kw - PW
                
                valid = hw_mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
                
                x_ptrs = X_ptr + (
                )
                x_val = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.float32)
                
                w_val = tl.load(W_ptr + c_idx * (KH * KW) + kh * KW + kw).to(tl.float32)
                acc += x_val * w_val
        
        # Apply BN
        if has_bn:
            m = tl.load(mean_ptr + c_idx).to(tl.float32)
            v = tl.load(var_ptr + c_idx).to(tl.float32)
            g = tl.load(gamma_ptr + c_idx).to(tl.float32)
            bt = tl.load(beta_ptr + c_idx).to(tl.float32)
            acc = (acc - m) / tl.sqrt(v + eps)
            acc = acc * g + bt
        
        if apply_relu:
            acc = tl.maximum(acc, 0.0)
        
        # Store
        out_ptrs = out_ptr + (
        )
        tl.store(out_ptrs, acc, mask=hw_mask)


def triton_depthwise_conv2d_bn_relu(x, weight, mean, var, gamma, beta, stride, padding, eps=1e-5, apply_relu=True, has_bn=True):
    N, C, IN_H, IN_W = x.shape
    _, _, KH, KW = weight.shape
    SH, SW = stride if isinstance(stride, (list, tuple)) else (stride, stride)
    PH, PW = padding if isinstance(padding, (list, tuple)) else (padding, padding)
    
    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1
    OUT_HW = OUT_H * OUT_W
    
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    if mean is None:
        mean = torch.zeros(C, device=x.device, dtype=x.dtype)
        var = torch.ones(C, device=x.device, dtype=x.dtype)
        gamma = torch.ones(C, device=x.device, dtype=x.dtype)
        beta = torch.zeros(C, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(C, meta['BLOCK_C']),
        triton.cdiv(OUT_HW, meta['BLOCK_HW']),
    )
    
    depthwise_conv2d_bn_relu_kernel[grid](
        x.contiguous(), weight.contiguous(), out,
        mean, var, gamma, beta,
        N, C, IN_H, IN_W,
        OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        OUT_HW,
        eps, apply_relu, has_bn,
    )
    return out


# ============================================================
# Main kernel function implementing EfficientNetB2 forward
# ============================================================
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