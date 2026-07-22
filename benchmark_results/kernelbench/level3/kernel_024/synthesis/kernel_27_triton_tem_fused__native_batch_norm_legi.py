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
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, spatial,
    eps,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // spatial) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


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
def sigmoid_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = 1.0 / (1.0 + tl.exp(-x))
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


# ============================================================
# Adaptive Average Pool 2D (global) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 8}, num_warps=4),
        triton.Config({'BLOCK_C': 4}, num_warps=4),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def adaptive_avg_pool2d_global_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    if n >= N:
        return
    
    spatial = H * W
    base = n * C * spatial + c * spatial
    
    acc = tl.zeros((), dtype=tl.float32)
    for i in range(spatial):
        val = tl.load(x_ptr + base + i)
        acc += val
    
    out = acc / spatial
    tl.store(out_ptr + n * C + c, out)


# ============================================================
# Implicit GEMM Conv2d kernel (1x1 convolution)
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
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
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
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Implicit GEMM Conv2d kernel (3x3 depthwise convolution)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_C': 32}, num_warps=4),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W', 'IN_H', 'IN_W'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    # Each program handles BLOCK_M output spatial positions for BLOCK_C channels
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    out_w_idx = offs_m % OUT_W
    out_h_idx = offs_m // OUT_W
    
    m_mask = offs_m < (OUT_H * OUT_W)
    c_mask = offs_c < C
    
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = out_h_idx * STRIDE_H + kh - PAD_H
            iw = out_w_idx * STRIDE_W + kw - PAD_W
            
            valid_h = (ih >= 0) & (ih < IN_H)
            valid_w = (iw >= 0) & (iw < IN_W)
            valid = valid_h & valid_w
            
            x_idx = pid_n * C * IN_H * IN_W + offs_c[None, :] * IN_H * IN_W + ih[:, None] * IN_W + iw[:, None]
            x_mask = m_mask[:, None] & c_mask[None, :] & valid[:, None]
            x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)
            
            w_idx = offs_c * 9 + kh * 3 + kw
            w_mask = c_mask
            w = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)
            
            acc += x * w[None, :]
    
    out_idx = pid_n * C * OUT_H * OUT_W + offs_c[None, :] * OUT_H * OUT_W + out_h_idx[:, None] * OUT_W + out_w_idx[:, None]
    out_mask = m_mask[:, None] & c_mask[None, :]
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=out_mask)


# ============================================================
# General Conv2d kernel (3x3 with groups=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_general_kernel(
    x_ptr, w_ptr, out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N, K,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    out_w_idx = offs_m % OUT_W
    out_h_idx = (offs_m // OUT_W) % OUT_H
    batch_idx = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K
    
    for k_block in range(BLOCK_K_COUNT):
        k_offs = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        
        in_c = k_offs % IN_C
        kw = (k_offs // IN_C) % KERNEL_W
        kh = k_offs // (IN_C * KERNEL_W)
        
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (k_offs[None, :] < TOTAL_K) & (offs_m[:, None] < M)
        
        x_ptrs = x_ptr + batch_idx[:, None] * IN_C * IN_H * IN_W + in_c[None, :] * IN_H * IN_W + ih * IN_W + iw
        x = tl.load(x_ptrs, mask=valid, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * IN_C * KERNEL_H * KERNEL_W + in_c[:, None] * KERNEL_H * KERNEL_W + kh[:, None] * KERNEL_W + kw[:, None]
        w_mask = (k_offs[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    out_ptrs = out_ptr + batch_idx[:, None] * OUT_C * OUT_H * OUT_W + offs_n[None, :] * OUT_H * OUT_W + out_h_idx[:, None] * OUT_W + out_w_idx[:, None]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Linear (GEMM) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
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
        
        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    b_ptrs = b_ptr + offs_n
    b_mask = offs_n < N
    bias = tl.load(b_ptrs, mask=b_mask, other=0.0)
    
    out = acc + bias[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out.to(tl.float32), mask=out_mask)


# ============================================================
# Helper functions
# ============================================================

def triton_conv1x1(x, w):
    """1x1 convolution using GEMM: x is (N, C, H, W), w is (Co, Ci, 1, 1)"""
    N, Ci, H, W = x.shape
    Co = w.shape[0]
    
    # Reshape to (N*H*W, Ci) x (Ci, Co) -> (N*H*W, Co)
    x_flat = x.permute(0, 2, 3, 1).contiguous().view(N * H * W, Ci)
    w_flat = w.view(Co, Ci)
    
    M = N * H * W
    out = torch.empty((M, Co), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(Co, meta['BLOCK_N']))
    conv1x1_kernel[grid](
        x_flat, w_flat, out,
        M, Co, Ci,
    )
    
    return out.view(N, H, W, Co).permute(0, 3, 1, 2).contiguous()


def triton_depthwise_conv3x3(x, w, stride=1, padding=1):
    """Depthwise 3x3 convolution"""
    N, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + 2 * padding - 3) // stride + 1
    OUT_W = (IN_W + 2 * padding - 3) // stride + 1
    
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    BLOCK_M = 32
    BLOCK_C = 32
    
    grid = (triton.cdiv(OUT_H * OUT_W, BLOCK_M), triton.cdiv(C, BLOCK_C), N)
    depthwise_conv3x3_kernel[grid](
        x, w, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        stride, stride,
        padding, padding,
        BLOCK_M=BLOCK_M, BLOCK_C=BLOCK_C,
    )
    
    return out


def triton_conv3x3(x, w, stride=2, padding=1):
    """General 3x3 convolution"""
    N, Ci, IN_H, IN_W = x.shape
    Co = w.shape[0]
    OUT_H = (IN_H + 2 * padding - 3) // stride + 1
    OUT_W = (IN_W + 2 * padding - 3) // stride + 1
    
    M = N * OUT_H * OUT_W
    out = torch.empty((N, Co, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(Co, meta['BLOCK_N']))
    conv2d_general_kernel[grid](
        x, w, out,
        N, Ci, IN_H, IN_W,
        Co, OUT_H, OUT_W,
        stride, stride,
        padding, padding,
        M, Co, Ci * 3 * 3,
        KERNEL_H=3, KERNEL_W=3,
    )
    
    return out


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    """Fused BatchNorm + optional ReLU"""
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, C, spatial,
        eps,
        do_relu,
    )
    
    return out


def triton_sigmoid(x):
    """Sigmoid activation"""
    numel = x.numel()
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    
    return out


def triton_relu(x):
    """ReLU activation"""
    numel = x.numel()
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    
    return out


def triton_adaptive_avg_pool2d_global(x):
    """Global adaptive average pooling to (1, 1)"""
    N, C, H, W = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    
    # Use a simple approach: one thread per (n, c) pair
    grid = (N * C,)
    adaptive_avg_pool2d_global_kernel[grid](
        x, out.view(N, C),
        N, C, H, W,
        BLOCK_C=1,
    )
    
    return out


def triton_linear(x, w, b):
    """Linear layer: x @ w.T + b"""
    M, K = x.shape
    N = w.shape[0]
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, w, b, out,
        M, N, K,
    )
    
    return out


# ============================================================
# MBConv block helper
# ============================================================

def mbconv_block(x, 
                 expand_w, expand_bn_w, expand_bn_b, expand_bn_mean, expand_bn_var,
                 dw_w, dw_bn_w, dw_bn_b, dw_bn_mean, dw_bn_var,
                 se_reduce_w, se_expand_w,
                 proj_w, proj_bn_w, proj_bn_b, proj_bn_mean, proj_bn_var,
                 stride, expand_ratio, eps=1e-5):
    """MBConv block forward pass"""
    
    # Expansion phase
    if expand_ratio != 1:
        x = triton_conv1x1(x, expand_w)
        x = triton_bn_relu(x, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b, eps=eps, do_relu=True)
    
    # Depthwise convolution
    x = triton_depthwise_conv3x3(x, dw_w, stride=stride, padding=1)
    x = triton_bn_relu(x, dw_bn_mean, dw_bn_var, dw_bn_w, dw_bn_b, eps=eps, do_relu=True)
    
    # Squeeze and Excitation
    se = triton_adaptive_avg_pool2d_global(x)
    se = triton_conv1x1(se, se_reduce_w)
    se = triton_relu(se)
    se = triton_conv1x1(se, se_expand_w)
    se = triton_sigmoid(se)
    x = x * se
    
    # Projection
    x = triton_conv1x1(x, proj_w)
    x =