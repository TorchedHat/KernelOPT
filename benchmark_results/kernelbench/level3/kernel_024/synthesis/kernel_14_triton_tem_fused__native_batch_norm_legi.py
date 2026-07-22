import torch
import triton
import triton.language as tl



import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused BatchNorm + ReLU (pointwise)
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
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    # Compute channel index
    c = (offs // HW) % C
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        numel, C, HW,
        eps, do_relu,
    )
    return out


# ============================================================
# Sigmoid (pointwise)
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
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, out, mask=mask)


def triton_sigmoid(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


# ============================================================
# ReLU (pointwise)
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
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


def triton_relu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


# ============================================================
# Adaptive Average Pool 2D -> (1,1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 8}, num_warps=4),
        triton.Config({'BLOCK_C': 4}, num_warps=4),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    if n >= N:
        return
    
    base = n * C * HW + c * HW
    acc = tl.zeros((), dtype=tl.float32)
    
    # Loop over spatial positions
    for i in range(0, HW, BLOCK_C):
        offs = i + tl.arange(0, BLOCK_C)
        mask = offs < HW
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    
    out = acc / HW
    tl.store(out_ptr + n * C + c, out)


def triton_adaptive_avg_pool(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    grid = (N * C,)
    adaptive_avg_pool_kernel[grid](
        x, out.view(N * C),
        N, C, HW,
    )
    return out


# ============================================================
# Conv2d as implicit GEMM
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    # Tensor dims
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    GROUPS,
    # Strides for X (NCHW)
    sx_n, sx_c, sx_h, sx_w,
    # Strides for W (Co, Ci, Kh, Kw)
    sw_o, sw_i, sw_h, sw_w,
    # Strides for Out (NCHW)
    so_n, so_c, so_h, so_w,
    # GEMM dims
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
    
    # Group handling
    GROUP_OUT_C = OUT_C // GROUPS
    GROUP_IN_C = IN_C // GROUPS
    
    group_idx = offs_n // GROUP_OUT_C
    local_out_c = offs_n % GROUP_OUT_C
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = GROUP_IN_C * KERNEL_H * KERNEL_W
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c_local, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c_local = tmp_k // KERNEL_H
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        in_c_global = group_idx[None, :] * GROUP_IN_C + in_c_local[None, :]  # wrong shape, fix below
        
        # Actually need to handle groups properly
        # For X: in_c = group * GROUP_IN_C + in_c_local
        # But group depends on output channel (offs_n), not K
        # For simplicity with groups=1 or depthwise, handle separately
        
        # For standard conv (groups=1):
        in_c_abs = in_c_local  # shape [BLOCK_K]
        
        x_ptrs = (X_ptr + 
                  batch_idx[:, None] * sx_n +
                  in_c_abs[None, :] * sx_c +
        
        valid_x = (
            (offs_m[:, None] < M) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < K)
        )
        
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        # W: shape (OUT_C, IN_C, KH, KW) -> (OUT_C, K)
        # offs_k encodes (in_c_local, kh, kw)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * sw_o +
                  in_c_local[:, None] * sw_i +
                  kh[:, None] * sw_h +
                  kw[:, None] * sw_w)
        
        valid_w = (
            (offs_k[:, None] < K) &
            (offs_n[None, :] < OUT_C)
        )
        
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store output
    out_ptrs = (Out_ptr +
                batch_idx[:, None] * so_n +
                offs_n[None, :] * so_c +
                out_h_idx[:, None] * so_h +
                out_w_idx[:, None] * so_w)
    
    valid_out = (
        (offs_m[:, None] < M) &
        (offs_n[None, :] < OUT_C)
    )
    
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


def triton_conv2d(x, w, stride=(1,1), padding=(0,0), groups=1):
    """Standard conv2d using implicit GEMM (groups=1 or depthwise handled separately)"""
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    
    SH, SW = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    PH, PW = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1
    
    GROUP_IN_C = IN_C // groups
    K = GROUP_IN_C * KH * KW
    M = N * OUT_H * OUT_W
    
    out = torch.zeros(N, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    if groups > 1 and groups == IN_C and groups == OUT_C:
        # Depthwise convolution - handle specially
        return triton_depthwise_conv2d(x, w, stride=(SH, SW), padding=(PH, PW))
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_implicit_gemm_kernel[grid](
        x, w, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        groups,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        M, OUT_C, K,
    )
    return out


# ============================================================
# Depthwise Conv2d
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_C': 32}, num_warps=4),
    ],
    key=['M', 'C', 'KH', 'KW'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    SH, SW,
    PH, PW,
    sx_n, sx_c, sx_h, sx_w,
    sw_c, sw_h, sw_w,
    so_n, so_c, so_h, so_w,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = out_h_idx[:, None] * SH + kh - PH
            iw = out_w_idx[:, None] * SW + kw - PW
            
            x_ptrs = (X_ptr +
                      batch_idx[:, None] * sx_n +
                      offs_c[None, :] * sx_c +
            
            valid_x = (
                (offs_m[:, None] < M) &
                (offs_c[None, :] < C) &
                (ih >= 0) & (ih < IN_H) &
                (iw >= 0) & (iw < IN_W)
            )
            
            x_val = tl.load(x_ptrs, mask=valid_x, other=0.0).to(tl.float32)
            
            w_ptrs = W_ptr + offs_c * sw_c + kh * sw_h + kw * sw_w
            valid_w = offs_c < C
            w_val = tl.load(w_ptrs, mask=valid_w, other=0.0).to(tl.float32)
            
            acc += x_val * w_val[None, :]
    
    out_ptrs = (Out_ptr +
                batch_idx[:, None] * so_n +
                offs_c[None, :] * so_c +
                out_h_idx[:, None] * so_h +
                out_w_idx[:, None] * so_w)
    
    valid_out = (offs_m[:, None] < M) & (offs_c[None, :] < C)
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


def triton_depthwise_conv2d(x, w, stride=(1,1), padding=(0,0)):
    N, C, IN_H, IN_W = x.shape
    _, _, KH, KW = w.shape
    SH, SW = stride
    PH, PW = padding
    
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1
    M = N * OUT_H * OUT_W
    
    out = torch.zeros(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    # w shape: (C, 1, KH, KW) -> strides
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C, meta['BLOCK_C']),
    )
    
    depthwise_conv2d_kernel[grid](
        x, w.view(C, KH, KW), out,
        N, C, IN_H, IN_W,
        OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.view(C, KH, KW).stride(0), w.view(C, KH, KW).stride(1), w.view(C, KH, KW).stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        M,
    )
    return out


# ============================================================
# Linear layer
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        valid_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        # W is (N, K), we need (K, N) for the dot
        w_ptrs = W_ptr + offs_n[None, :] * K + offs_k[:, None]
        valid_w = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += b[None, :]
    
    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


def triton_linear(x, w, b):
    # x: (M, K), w: (N, K), b: (N,)
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, w, b, out, M, N, K)
    return out


# ============================================================
# Multiply (for SE block: x * sigmoid_scale)
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
def channel_mul_kernel(
    x_ptr, scale_ptr, out_ptr,
    numel, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    c = (offs // HW) % C
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # scale is (N, C, 1, 1) -> index as n*C + c
    n = offs // (C * HW)
    scale = tl.load(scale_ptr + n * C + c, mask=mask, other=1.0).to(tl.float32)
    
    out = x * scale
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def triton_channel_mul(x, scale):
    """x: (N,C,H,W), scale: (N,C,1,1)"""
    N, C, H, W = x.shape
    numel = x.numel()
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_mul_kernel[grid](x, scale.contiguous(), out, numel, C, HW)
    return out


# ============================================================
# Main kernel_function: EfficientNetB2 forward pass
# ============================================================
def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_mbconv1_0_weight, p_mbconv1_1_weight, p_mbconv1_1_bias,
    p_mbconv1_3_weight, p_mbconv1_4_weight, p_mbconv1_4_bias,
    p_mbconv1_7_weight, p_mbconv1_9_weight,
    p_mbconv1_11_weight, p_mbconv1_12_weight, p_mbconv1_12_bias,
    p_mbconv2_0_weight, p_mbconv2_1_weight, p_mbconv2_1_bias,