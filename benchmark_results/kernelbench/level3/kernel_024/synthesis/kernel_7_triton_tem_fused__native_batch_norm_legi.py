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
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # channel index: layout is (N, C, H, W) -> channel = (offset / spatial) % num_channels
    chan = (offsets // spatial) % num_channels
    
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + chan, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    if apply_relu:
        out = tl.where(out > 0.0, out, 0.0)
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Fused BatchNorm (no relu) pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    chan = (offsets // spatial) % num_channels
    
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + chan, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


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
def relu_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0.0, x, tl.zeros_like(x))
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
def sigmoid_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + libdevice.exp(-x))
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Elementwise multiply kernel (for SE attention)
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
def mul_kernel(
    a_ptr, b_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a * b, mask=mask)


# ============================================================
# Linear (GEMM) kernel: out = x @ W^T + bias
# M x K @ K x N -> M x N
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=out_mask)


# ============================================================
# Implicit GEMM Conv2d kernel (for 1x1 convolutions)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
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
    """1x1 convolution as matrix multiplication: (N*H*W, Cin) x (Cin, Cout)"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        # x: (M, K) - input reshaped
        x_ptrs = x_ptr + offs_m[:, None] * K + k_offs[None, :]
        # w: (N, K) -> access as (K, N) transposed
        w_ptrs = w_ptr + offs_n[None, :] * K + k_offs[:, None]
        
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=out_mask)


# ============================================================
# Depthwise Conv2d kernel (groups=channels)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 16}, num_warps=8),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    """Depthwise convolution: each channel has its own filter"""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    # M = N * OUT_H * OUT_W
    M = N * OUT_H * OUT_W
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    # Decode m -> (n, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    n = offs_m // (OUT_W * OUT_H)
    
    m_mask = offs_m < M
    c_mask = offs_c < C
    
    # For each (m, c), compute the depthwise conv
    # Output shape: (N, C, OUT_H, OUT_W)
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            
            valid_h = (ih >= 0) & (ih < IN_H)
            valid_w = (iw >= 0) & (iw < IN_W)
            valid_m = m_mask & valid_h & valid_w
            
            # x: (N, C, IN_H, IN_W) -> index: n*C*IN_H*IN_W + c*IN_H*IN_W + ih*IN_W + iw
            x_idx = (n[:, None] * C * IN_H * IN_W + 
                     offs_c[None, :] * IN_H * IN_W + 
                     ih[:, None] * IN_W + 
                     iw[:, None])
            x_mask = valid_m[:, None] & c_mask[None, :]
            x_val = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
            
            # w: (C, 1, KH, KW) -> index: c*KH*KW + kh*KW + kw
            w_idx = offs_c * KH * KW + kh * KW + kw
            w_val = tl.load(w_ptr + w_idx, mask=c_mask, other=0.0)
            
            acc += x_val * w_val[None, :]
    
    # Store: out[n, c, oh, ow] = acc
    out_idx = (n[:, None] * C * OUT_H * OUT_W + 
               offs_c[None, :] * OUT_H * OUT_W + 
               oh[:, None] * OUT_W + 
               ow[:, None])
    out_mask = m_mask[:, None] & c_mask[None, :]
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Adaptive Average Pool 2d -> (1,1) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 16}, num_warps=2),
        triton.Config({'BLOCK_C': 128}, num_warps=8),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    BLOCK_C: tl.constexpr,
):
    """Adaptive average pool to (1,1): average over H*W for each (n, c)"""
    pid = tl.program_id(0)
    offs_nc = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    nc_mask = offs_nc < N * C
    
    n = offs_nc // C
    c = offs_nc % C
    
    HW = H * W
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    
    for hw in range(HW):
        h_idx = hw // W
        w_idx = hw % W
        x_idx = n * C * H * W + c * H * W + h_idx * W + w_idx
        x_val = tl.load(x_ptr + x_idx, mask=nc_mask, other=0.0).to(tl.float32)
        acc += x_val
    
    acc = acc / HW.to(tl.float32)
    tl.store(out_ptr + offs_nc, acc.to(tl.float32), mask=nc_mask)


# ============================================================
# Helper functions
# ============================================================

def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5, apply_relu=True):
    """Apply batch norm + optional relu using Triton kernel"""
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    spatial = x.numel() // (x.shape[0] * x.shape[1])
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def triton_bn(x, mean, var, weight, bias, eps=1e-5):
    """Apply batch norm using Triton kernel"""
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    spatial = x.numel() // (x.shape[0] * x.shape[1])
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def triton_relu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def triton_sigmoid(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


def triton_adaptive_avg_pool2d_1x1(x):
    """Adaptive average pool to (1,1)"""
    N, C, H, W = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(N * C, meta['BLOCK_C']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x, out.view(N * C),
    )
    return out


def triton_conv1x1(x, w):
    """1x1 convolution as matrix multiplication"""
    N, Cin, H, W = x.shape
    Cout = w.shape[0]
    
    # Reshape: (N*H*W, Cin) x (Cin, Cout) -> (N*H*W, Cout)
    M = N * H * W
    K = Cin
    
    x_flat = x.permute(0, 2, 3, 1).contiguous().view(M, K)
    w_flat = w.view(Cout, K)  # (Cout, Cin)
    
    out_flat = torch.empty(M, Cout, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(Cout, meta['BLOCK_N']))
    conv1x1_kernel[grid](
        x_flat, w_flat, out_flat,
    )
    
    return out_flat.view(N, H, W, Cout).permute(0, 3, 1, 2).contiguous()


def triton_depthwise_conv2d(x, w, stride=1, padding=1):
    """Depthwise convolution"""
    N, C, IN_H, IN_W = x.shape
    KH, KW = w.shape[2], w.shape[3]
    
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1
    
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    M = N * OUT_H * OUT_W
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C, meta['BLOCK_C'])
    )
    
    depthwise_conv2d_kernel[grid](
        x, w.view(C, KH * KW), out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
    )
    
    return out


def triton_linear(x, w, bias):
    """Linear layer: x @ w.T + bias"""
    M, K = x.shape
    N = w.shape[0]
    
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, w, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1)
    )
    
    return out


# ============================================================
# Main kernel function implementing EfficientNetB2 forward pass
# ============================================================

def mbconv_block(x, 
                 expand_w, expand_bn_w, expand_bn_b, expand_bn_mean, expand_bn_var,
                 dw_w, dw_bn_w, dw_bn_b, dw_bn_mean, dw_bn_var,
                 se_reduce_w, se_expand_w,
                 proj_w, proj_bn_w, proj_bn_b, proj_bn_mean, proj_bn_var,
                 stride=1, eps=1e-5):
    """MBConv block with SE"""
    # Expansion
    x = triton_conv1x1(x, expand_w)
    x = triton_bn_relu(x, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b, eps=eps, apply_relu=True)
    
    # Depthwise conv
    x = triton_depthwise_conv2d(x, dw_w, stride=stride, padding=1)
    x = triton_bn_relu(x, dw_bn_mean, dw_bn_var, dw_bn_w, dw_bn_b, eps=eps, apply_relu=True)
    
    # SE
    se = triton_adaptive_avg_pool2d_1x1(x)
    se = triton_conv1x1(se, se_reduce_w)
    se = triton_relu(se)
    se = triton_conv1x1(se, se_expand_w)
    se = triton_sigmoid(se)
    x = x * se
    
    # Projection
    x = triton_conv1x1(x, proj_w)
    x = triton_bn(x, proj_bn_mean, proj_bn_var, proj_bn_w, proj_bn_b, eps=eps)
    
    return x


def kernel_function(
    p_conv1_weight,
    p_bn1_weight, p_bn1_bias,
    p_mbconv1_0_weight,
    p_mbconv1_1_weight, p_mbconv1_1_bias,
    p_mbconv1_3_weight,
    p_mbconv1_4_weight, p_mbconv1_4_bias,
    p_mbconv1_7_weight,
    p_mbconv1_9_weight,
    p_mbconv1_11_weight,
    p_mbconv1_12_weight, p_mbconv1_12_bias,
    p_mbconv2_0_weight,
    p_mbconv2_1_weight, p_mbconv2_1_bias,
    p_mbconv2_3_weight,
    p_mbconv2_4_weight, p_mbconv2_4_bias,
    p_mbconv2_7_weight,
    p_mbconv2_9_weight,
    p_mbconv2_11_weight,
    p_mbconv2_12_weight, p_mbconv2_12_bias,
    p_mbconv3_0_weight,
    p_mbconv3_1_weight, p_mbconv3_1_bias,
    p_mbconv3_3_weight,
    p_mbconv3_4_weight, p_mbconv3_4_bias,
    p_mbconv3_7_weight,
    p_mbconv3_9_weight,
    p_mbconv3_11_weight,
    p_mbconv3_12_weight, p_mbconv3_12_bias,
    p_mbconv4_0_weight,
    p_mbconv4_1_weight, p_mbconv4_1_bias,
    p_mbconv4_3_weight,
    p_mbconv4_4_weight, p_mbconv4_4_bias,
    p_mbconv4_7_weight,
    p_mbconv4_9_weight,
    p_mbconv4_11_weight,
    p_mbconv4_12_weight, p_mbconv4_12_bias,
    p_mbconv5_0_weight,
    p_mbconv5_1_weight, p_mbconv5_1_bias,
    p_mbconv5_3_weight,
    p_mbconv5_4_weight, p_mbconv5_4_bias,
    p_mbconv5_7_weight,
    p_mbconv5_9_weight,
    p_mbconv5_11_weight,
    p_mbconv5_12_weight, p_mbconv5_12_bias,
    p_conv_final_weight,
    p_bn_final_weight, p_bn_final_bias,