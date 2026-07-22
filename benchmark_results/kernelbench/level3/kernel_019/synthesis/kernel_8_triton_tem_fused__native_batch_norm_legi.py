import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused BatchNorm + ReLU kernel (pointwise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Determine channel index
    # Layout: N, C, H, W -> index = n*C*HW + c*HW + hw
    c_idx = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Fused AvgPool2d (global, kernel=7) + flatten kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 256}, num_warps=8),
        triton.Config({'BLOCK_C': 32}, num_warps=2),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def avgpool_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one (n, c_block) pair
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C
    
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    
    for hw in range(HW):
        x = tl.load(x_ptr + pid_n * C * HW + c_offsets * HW + hw, mask=c_mask, other=0.0)
        acc += x
    
    acc = acc / HW
    tl.store(out_ptr + pid_n * C + c_offsets, acc, mask=c_mask)


# ============================================================
# Linear (GEMM) kernel: out[m, n] = sum_k(x[m,k] * w[n,k]) + b[n]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
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
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ============================================================
# Implicit GEMM Conv2d kernel (standard conv, NCHW)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    M, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    M = N * OUT_H * OUT_W
    N_out = OUT_C
    K = IN_C * KH * KW
    """
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
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        in_c_idx = tmp_k // KH
        
        # Input coordinates
        in_h = out_h_idx[:, None] * stride_h + kh_idx[None, :] - pad_h
        in_w = out_w_idx[:, None] * stride_w + kw_idx[None, :] - pad_w
        
        valid = (
            (offs_m[:, None] < M) &
            (batch_idx[:, None] < N) &
            (in_h >= 0) & (in_h < IN_H) &
            (in_w >= 0) & (in_w < IN_W) &
            (offs_k[None, :] < K)
        )
        
        x_ptrs = (x_ptr + 
                  batch_idx[:, None] * IN_C * IN_H * IN_W +
                  in_c_idx[None, :] * IN_H * IN_W +
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        # Weight: [OUT_C, IN_C, KH, KW]
        w_ptrs = (w_ptr +
                  offs_n[None, :] * IN_C * KH * KW +
                  in_c_idx[:, None] * KH * KW +
                  kh_idx[:, None] * KW +
                  kw_idx[:, None])
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    out_ptrs = (out_ptr +
                batch_idx[:, None] * OUT_C * OUT_H * OUT_W +
                offs_n[None, :] * OUT_H * OUT_W +
                out_h_idx[:, None] * OUT_W +
                out_w_idx[:, None])
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Depthwise Conv2d kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 16}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 64}, num_warps=4),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W', 'KH', 'KW'],
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    """
    Depthwise convolution: groups=C, each channel convolved independently.
    """
    pid_m = tl.program_id(0)  # spatial tile
    pid_c = tl.program_id(1)  # channel tile
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    M_total = N * OUT_H * OUT_W
    
    acc = tl.zeros([BLOCK_M, BLOCK_C], dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            in_h = out_h_idx[:, None] * stride_h + kh - pad_h
            in_w = out_w_idx[:, None] * stride_w + kw - pad_w
            
            valid = (
                (offs_m[:, None] < M_total) &
                (in_h >= 0) & (in_h < IN_H) &
                (in_w >= 0) & (in_w < IN_W) &
                (offs_c[None, :] < C)
            )
            
            x_ptrs = (x_ptr +
                      batch_idx[:, None] * C * IN_H * IN_W +
                      offs_c[None, :] * IN_H * IN_W +
            x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
            
            # Weight: [C, 1, KH, KW] -> [C, KH, KW]
            w_ptrs = w_ptr + offs_c * KH * KW + kh * KW + kw
            w_tile = tl.load(w_ptrs, mask=offs_c < C, other=0.0)
            
            acc += x_tile * w_tile[None, :]
    
    out_mask = (offs_m[:, None] < M_total) & (offs_c[None, :] < C)
    out_ptrs = (out_ptr +
                batch_idx[:, None] * C * OUT_H * OUT_W +
                offs_c[None, :] * OUT_H * OUT_W +
                out_h_idx[:, None] * OUT_W +
                out_w_idx[:, None])
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Pointwise Conv2d (1x1) kernel - use GEMM approach
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def pointwise_conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = N * C * H * W
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](x, out, mean, var, weight, bias, numel, C, HW, eps)
    return out


def triton_conv2d(x, w, stride=1, padding=1):
    """Standard conv2d using implicit GEMM."""
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1
    
    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW
    
    out = torch.empty(N, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    conv2d_kernel[grid](
        x, w, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        M, K,
    )
    return out


def triton_depthwise_conv2d(x, w, stride=1, padding=1):
    """Depthwise conv2d."""
    N, C, IN_H, IN_W = x.shape
    KH, KW = w.shape[2], w.shape[3]
    
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1
    
    M_total = N * OUT_H * OUT_W
    
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M_total, meta['BLOCK_M']), triton.cdiv(C, meta['BLOCK_C']))
    depthwise_conv2d_kernel[grid](
        x, w, out,
        N, C, IN_H, IN_W,
        OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
    )
    return out


def triton_pointwise_conv2d(x, w):
    """1x1 conv as GEMM."""
    N, IN_C, H, W = x.shape
    OUT_C = w.shape[0]
    
    M = N * H * W
    K = IN_C
    
    # Reshape x to [M, K]
    x_2d = x.permute(0, 2, 3, 1).contiguous().view(M, K)
    w_2d = w.view(OUT_C, K)
    
    out_2d = torch.empty(M, OUT_C, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    pointwise_conv2d_kernel[grid](x_2d, w_2d, out_2d, M, OUT_C, K)
    
    # Reshape back to [N, OUT_C, H, W]
    out = out_2d.view(N, H, W, OUT_C).permute(0, 3, 1, 2).contiguous()
    return out


def triton_avgpool(x, kernel_size=7):
    """Average pooling."""
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, device=x.device, dtype=x.dtype)
    grid = lambda meta: (N, triton.cdiv(C, meta['BLOCK_C']))
    avgpool_kernel[grid](x, out, N, C, HW)
    return out.view(N, C, 1, 1)


def triton_linear(x, w, b):
    """Linear layer."""
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, w, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
    )
    return out


def conv_bn_relu(x, conv_w, bn_mean, bn_var, bn_w, bn_b, stride=1, padding=1):
    x = triton_conv2d(x, conv_w, stride=stride, padding=padding)
    x = bn_relu(x, bn_mean, bn_var, bn_w, bn_b)
    return x


def conv_dw_bn_relu(x, dw_w, bn_mean, bn_var, bn_w, bn_b, stride=1):
    x = triton_depthwise_conv2d(x, dw_w, stride=stride, padding=1)
    x = bn_relu(x, bn_mean, bn_var, bn_w, bn_b)
    return x


def conv_pw_bn_relu(x, pw_w, bn_mean, bn_var, bn_w, bn_b):
    x = triton_pointwise_conv2d(x, pw_w)
    x = bn_relu(x, bn_mean, bn_var, bn_w, bn_b)
    return x


def kernel_function(
    p_model_0_0_weight, p_model_0_1_weight, p_model_0_1_bias,
    p_model_1_0_weight, p_model_1_1_weight, p_model_1_1_bias,
    p_model_1_3_weight, p_model_1_4_weight, p_model_1_4_bias,
    p_model_2_0_weight, p_model_2_1_weight, p_model_2_1_bias,
    p_model_2_3_weight, p_model_2_4_weight, p_model_2_4_bias,
    p_model_3_0_weight, p_model_3_1_weight, p_model_3_1_bias,
    p_model_3_3_weight, p_model_3_4_weight, p_model_3_4_bias,
    p_model_4_0_weight, p_model_4_1_weight, p_model_4_1_bias,
    p_model_4_3_weight, p_model_4_4_weight, p_model_4_4_bias,
    p_model_5_0_weight, p_model_5_1_weight, p_model_5_1_bias,
    p_model_5_3_weight, p_model_5_4_weight, p_model_5_4_bias,
    p_model_6_0_weight, p_model_6_1_weight, p_model_6_1_bias,
    p_model_6_3_weight, p_model_6_4_weight, p_model_6_4_bias,
    p_model_7_0_weight, p_model_7_1_weight, p_model_7_1_bias,
    p_model_7_3_weight, p_model_7_4_weight, p_model_7_4_bias,
    p_model_8_0_weight, p_model_8_1_weight, p_model_8_1_bias,
    p_model_8_3_weight, p_model_8_4_weight, p_model_8_4_bias,
    p_model_9_0_weight, p_model_9_1_weight, p_model_9_1_bias,
    p_model_9_3_weight, p_model_9_4_weight, p_model_9_4_bias,
    p_model_10_0_weight, p_model_10_1_weight, p_model_10_1_bias,
    p_model_10_3_weight, p_model_10_4_weight, p_model_10_4_bias,
    p_model_11_0_weight, p_model_11_1_weight, p_model_11_1_bias,
    p_model_11_3_weight, p_model_11_4_weight, p_model_11_4_bias,
    p_model_12_0_weight, p_model_12_1_weight, p_model_12_1_bias,
    p_model_12_3_weight, p_model_12_4_weight, p_model_12_4_bias,
    p_model_13_0_weight, p_model_13_1_weight, p_model_13_1_bias,
    p_model_13_3_weight, p_model_13_4_weight, p_model_13_4_bias,
    p_fc_weight, p_fc_bias,
    b_model_0_1_running_mean, b_model_0_1_running_var,
    b_model_1_1_running_mean, b_model_1_1_running_var,
    b_model_1_4_running_mean, b_model_1_4_running_var,
    b_model_2_1_running_mean, b_model_2_1_running_var,
    b_model_2_4_running_mean, b_model_2_4_running_var,
    b_model_3_1_running_mean, b_model_3_1_running_var,
    b_model_3_4_running_mean, b_model_3_4_running_var,
    b_model_4_1_running_mean, b_model_4_1_running_var,