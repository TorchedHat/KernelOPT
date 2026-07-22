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
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps,
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
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    if apply_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Sigmoid pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
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
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
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
# Linear kernel (GEMM): y = x @ W^T + b
# x: [M, K], W: [N, K], b: [N], y: [M, N]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
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
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ============================================================
# Adaptive Average Pool 2D kernel (global average pooling)
# Input: [N, C, H, W] -> Output: [N, C, 1, 1]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_HW': 64}, num_warps=2),
    ],
    key=['HW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    base = n * C * HW + c * HW
    
    acc = tl.zeros((), dtype=tl.float32)
    for hw_start in range(0, HW, BLOCK_HW):
        offs = hw_start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    
    out = acc / HW
    tl.store(out_ptr + pid, out)


# ============================================================
# Conv2d implicit GEMM kernel (1x1 convolution)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
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
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ============================================================
# Depthwise Conv2d kernel (groups=C)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 32}, num_warps=2),
    ],
    key=['M_total', 'C'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M_total,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    # Decode m -> (n, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    n = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    
    # 3x3 kernel
    for kh in range(3):
        for kw in range(3):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            
            valid_m = (offs_m < M_total) & (n < N) & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            valid_c = offs_c < C
            
            # x: [BLOCK_M, BLOCK_C]
            x_idx = n[:, None] * C * IN_H * IN_W + offs_c[None, :] * IN_H * IN_W + ih[:, None] * IN_W + iw[:, None]
            x_mask = valid_m[:, None] & valid_c[None, :]
            x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0).to(tl.float32)
            
            # w: [BLOCK_C] (depthwise: each channel has its own 3x3 filter)
            w_idx = offs_c * 9 + kh * 3 + kw
            w_mask = valid_c
            w = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0).to(tl.float32)
            
            acc += x * w[None, :]
    
    # Store output
    out_idx = n[:, None] * C * OUT_H * OUT_W + offs_c[None, :] * OUT_H * OUT_W + oh[:, None] * OUT_W + ow[:, None]
    out_mask = (offs_m[:, None] < M_total) & (offs_c[None, :] < C)
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


# ============================================================
# Conv2d 3x3 (standard, non-depthwise) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M_total', 'OUT_C', 'IN_C'],
)
@triton.jit
def conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M_total,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m -> (n, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    n = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * 9  # 3x3 kernel
    
    for k_start in range(0, TOTAL_K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        
        # Decode k -> (ic, kh, kw)
        ic = k_offs % IN_C
        kh = (k_offs // IN_C) % 3
        kw = k_offs // (IN_C * 3)
        
        ih = oh[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = (offs_m < M_total) & (n < N)
        valid_k = k_offs < TOTAL_K
        valid_hw = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        
        x_idx = n[:, None] * IN_C * IN_H * IN_W + ic[None, :] * IN_H * IN_W + ih * IN_W + iw
        x_mask = valid_m[:, None] & valid_k[None, :] & valid_hw
        x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
        
        # w: [OUT_C, IN_C, 3, 3] -> w[oc, ic, kh, kw]
        # w_idx for [BLOCK_K, BLOCK_N]: w[offs_n, ic, kh, kw]
        w_idx = offs_n[None, :] * IN_C * 9 + ic[:, None] * 9 + kh[:, None] * 3 + kw[:, None]
        w_mask = (k_offs[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    out_idx = n[:, None] * OUT_C * OUT_H * OUT_W + offs_n[None, :] * OUT_H * OUT_W + oh[:, None] * OUT_W + ow[:, None]
    out_mask = (offs_m[:, None] < M_total) & (offs_n[None, :] < OUT_C)
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


# ============================================================
# Helper functions
# ============================================================

def run_bn_relu(x, mean, var, weight, bias, eps=1e-5, apply_relu=True):
    """Apply batch norm + optional relu"""
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias, out,
        numel, C, HW, eps,
        apply_relu,
    )
    return out


def run_sigmoid(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


def run_relu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def run_adaptive_avg_pool2d(x):
    """Global average pooling: [N, C, H, W] -> [N, C, 1, 1]"""
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    
    grid = (N * C,)
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        N, C, HW,
    )
    return out


def run_conv1x1(x, w):
    """1x1 convolution: x [N, C_in, H, W], w [C_out, C_in, 1, 1]"""
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    
    # Reshape to [N*H*W, C_in] x [C_in, C_out]
    M = N * H * W
    K = C_in
    
    x_2d = x.permute(0, 2, 3, 1).contiguous().view(M, K)
    w_2d = w.view(C_out, K)  # [C_out, K]
    
    out_2d = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv1x1_kernel[grid](
        x_2d, w_2d, out_2d,
        M, C_out, K,
        x_2d.stride(0), x_2d.stride(1),
        w_2d.stride(0), w_2d.stride(1),
    )
    
    return out_2d.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def run_conv3x3(x, w, stride=1, padding=1):
    """3x3 convolution"""
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C = w.shape[0]
    
    OUT_H = (IN_H + 2 * padding - 3) // stride + 1
    OUT_W = (IN_W + 2 * padding - 3) // stride + 1
    M_total = N * OUT_H * OUT_W
    
    out = torch.empty(N, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M_total, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    conv3x3_kernel[grid](
        x, w, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        stride, stride,
        padding, padding,
        M_total,
    )
    return out


def run_depthwise_conv3x3(x, w, stride=1, padding=1):
    """Depthwise 3x3 convolution"""
    N, C, IN_H, IN_W = x.shape
    
    OUT_H = (IN_H + 2 * padding - 3) // stride + 1
    OUT_W = (IN_W + 2 * padding - 3) // stride + 1
    M_total = N * OUT_H * OUT_W
    
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M_total, meta['BLOCK_M']), triton.cdiv(C, meta['BLOCK_C']))
    depthwise_conv3x3_kernel[grid](
        x, w, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        stride, stride,
        padding, padding,
        M_total,
    )
    return out


def run_linear(x, w, b):
    """Linear: x [M, K], w [N, K], b [N] -> [M, N]"""
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


def mbconv_block(x, expand_w, expand_bn_w, expand_bn_b, expand_bn_mean, expand_bn_var,
                 dw_w, dw_bn_w, dw_bn_b, dw_bn_mean, dw_bn_var,
                 se_reduce_w, se_expand_w,
                 proj_w, proj_bn_w, proj_bn_b, proj_bn_mean, proj_bn_var,
    """MBConv block"""
    # Expansion
    if expand_ratio != 1:
        x = run_conv1x1(x, expand_w)
        x = run_bn_relu(x, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b, apply_relu=True)
    
    # Depthwise conv
    x = run_depthwise_conv3x3(x, dw_w, stride=stride, padding=1)
    x = run_bn_relu(x, dw_bn_mean, dw_bn_var, dw_bn_w, dw_bn_b, apply_relu=True)
    
    # Squeeze and Excitation
    se = run_adaptive_avg_pool2d(x)
    se = run_conv1x1(se, se_reduce_w)
    se = run_relu(se)
    se = run_conv1x1(se, se_expand_w)
    se = run_sigmoid(se)
    x = x * se
    
    # Projection
    x = run_conv1x1(x, proj_w)
    x = run_bn_relu(x, proj_bn_mean, proj_bn_var, proj_bn_w, proj_bn_b, apply_relu=False)
    
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