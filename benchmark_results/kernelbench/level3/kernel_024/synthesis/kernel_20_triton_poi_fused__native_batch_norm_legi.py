import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm + ReLU pointwise kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, num_channels, spatial,
    eps,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if apply_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def relu_kernel(
    x_ptr, out_ptr, numel,
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
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def sigmoid_kernel(
    x_ptr, out_ptr, numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 64}, num_warps=4),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    if n >= N:
        return
    
    # Compute mean over HW dimension
    acc = tl.zeros([1], dtype=tl.float32)
    base = n * C * HW + c * HW
    
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        mask = hw_offsets < HW
        x = tl.load(x_ptr + base + hw_offsets, mask=mask, other=0.0)
        acc += tl.sum(x, axis=0)
    
    out = acc / HW
    tl.store(out_ptr + n * C + c, out)


# Linear kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 2, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 2, 'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    m_offsets = m_start + tl.arange(0, BLOCK_M)
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offsets = k_start + tl.arange(0, BLOCK_K)
        
        x_mask = (m_offsets[:, None] < M) & (k_offsets[None, :] < K)
        w_mask = (n_offsets[:, None] < N) & (k_offsets[None, :] < K)
        
        x = tl.load(x_ptr + m_offsets[:, None] * K + k_offsets[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + n_offsets[:, None] * K + k_offsets[None, :], mask=w_mask, other=0.0)
        
        acc += tl.dot(x, tl.trans(w))
    
    # Add bias
    bias = tl.load(bias_ptr + n_offsets, mask=n_offsets < N, other=0.0)
    acc += bias[None, :]
    
    out_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
    tl.store(out_ptr + m_offsets[:, None] * N + n_offsets[None, :], acc, mask=out_mask)


# Conv2d kernel using implicit GEMM
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    GROUPS,
    M_TOTAL, TOTAL_K,
    s_xn, s_xc, s_xh, s_xw,
    s_wn, s_wc, s_wh, s_ww,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    
    # Decode m -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
    
    # Determine group for each output channel
    group_idx = offs_n // GROUP_OUT_C
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode k -> (in_c_local, kh, kw)
        in_c_local = offs_k % GROUP_IN_C
        tmp_k = offs_k // GROUP_IN_C
        kw = tmp_k % KERNEL_W
        kh = tmp_k // KERNEL_W
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        # For grouped conv, in_c depends on group
        # group for output channel offs_n
        g_n = offs_n // GROUP_OUT_C  # [BLOCK_N]
        in_c_abs = in_c_local[None, :] + g_n[:, None] * GROUP_IN_C  # [BLOCK_N, BLOCK_K] - not right
        
        # Actually for standard (groups=1) or depthwise (groups=C):
        # For groups=1: in_c_abs = in_c_local
        # For depthwise: each output channel c maps to input channel c
        
        # Simple approach: handle groups=1 and depthwise separately
        # For groups=1:
        in_c_abs_x = in_c_local  # [BLOCK_K]
        
        valid_x = (
            (batch_idx[:, None] < BATCH) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )
        
        x_ptrs = (X_ptr + 
                  batch_idx[:, None] * s_xn + 
                  in_c_abs_x[None, :] * s_xc + 
        
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        
        # Weight: [OUT_C, IN_C/groups, KH, KW] -> [OUT_C, TOTAL_K]
        # For groups=1: w_ptr[oc, ic, kh, kw]
        w_ptrs = (W_ptr + 
                  offs_n[None, :] * s_wn + 
                  in_c_local[:, None] * s_wc + 
                  kh[:, None] * s_wh + 
                  kw[:, None] * s_ww)
        
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc += tl.dot(x_tile, w_tile)
    
    out_mask = (
        (batch_idx[:, None] < BATCH) &
        (out_h_idx[:, None] < OUT_H) &
        (out_w_idx[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )
    
    # Output: [BATCH, OUT_C, OUT_H, OUT_W]
    out_ptrs = (out_ptr + 
                batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) + 
                offs_n[None, :] * (OUT_H * OUT_W) + 
                out_h_idx[:, None] * OUT_W + 
                out_w_idx[:, None])
    
    tl.store(out_ptrs, acc, mask=out_mask)


# Depthwise conv2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
    ],
    key=['M_TOTAL', 'C'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, out_ptr,
    BATCH, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M_TOTAL,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    m_start = pid_m * BLOCK_M
    n_start = pid_n * BLOCK_N
    
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)  # channel indices
    
    # Decode m -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            ih = out_h_idx[:, None] * STRIDE_H + kh - PAD_H
            iw = out_w_idx[:, None] * STRIDE_W + kw - PAD_W
            
            valid_x = (
                (batch_idx[:, None] < BATCH) &
                (ih >= 0) & (ih < IN_H) &
                (iw >= 0) & (iw < IN_W) &
                (offs_n[None, :] < C)
            )
            
            x_ptrs = (X_ptr + 
                      batch_idx[:, None] * (C * IN_H * IN_W) + 
                      offs_n[None, :] * (IN_H * IN_W) + 
            
            x_vals = tl.load(x_ptrs, mask=valid_x, other=0.0)
            
            valid_w = offs_n < C
            w_ptrs = W_ptr + offs_n * (KERNEL_H * KERNEL_W) + kh * KERNEL_W + kw
            w_vals = tl.load(w_ptrs, mask=valid_w, other=0.0)
            
            acc += x_vals * w_vals[None, :]
    
    out_mask = (
        (batch_idx[:, None] < BATCH) &
        (out_h_idx[:, None] < OUT_H) &
        (out_w_idx[:, None] < OUT_W) &
        (offs_n[None, :] < C)
    )
    
    out_ptrs = (out_ptr + 
                batch_idx[:, None] * (C * OUT_H * OUT_W) + 
                offs_n[None, :] * (OUT_H * OUT_W) + 
                out_h_idx[:, None] * OUT_W + 
                out_w_idx[:, None])
    
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_conv2d(x, w, stride=(1,1), padding=(0,0), groups=1):
    """Triton-based conv2d"""
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    SH, SW = stride
    PH, PW = padding
    
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1
    
    out = torch.empty(BATCH, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    if groups == IN_C and groups == OUT_C:
        # Depthwise convolution
        M_TOTAL = BATCH * OUT_H * OUT_W
        C = IN_C
        
        grid = lambda meta: (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(C, meta['BLOCK_N']),
        )
        
        # Reshape weight for depthwise: [C, 1, KH, KW] -> [C, KH, KW]
        w_dw = w.squeeze(1)
        
        depthwise_conv2d_kernel[grid](
            x, w_dw, out,
            BATCH, C, IN_H, IN_W,
            OUT_H, OUT_W,
            KH, KW,
            SH, SW,
            PH, PW,
            M_TOTAL,
        )
    else:
        # Standard conv2d (groups=1)
        GROUP_IN_C = IN_C // groups
        TOTAL_K = GROUP_IN_C * KH * KW
        M_TOTAL = BATCH * OUT_H * OUT_W
        
        s_xn, s_xc, s_xh, s_xw = x.stride()
        s_wn, s_wc, s_wh, s_ww = w.stride()
        
        grid = lambda meta: (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )
        
        conv2d_kernel[grid](
            x, w, out,
            BATCH, IN_C, IN_H, IN_W,
            OUT_C, OUT_H, OUT_W,
            KH, KW,
            SH, SW,
            PH, PW,
            groups,
            M_TOTAL, TOTAL_K,
            s_xn, s_xc, s_xh, s_xw,
            s_wn, s_wc, s_wh, s_ww,
        )
    
    return out


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5, apply_relu=True):
    """Fused BatchNorm + ReLU"""
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    
    fused_bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        numel, num_channels, spatial,
        eps,
        apply_relu,
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


def triton_adaptive_avg_pool2d(x):
    """Adaptive avg pool to (1,1)"""
    BATCH, C, H, W = x.shape
    HW = H * W
    out = torch.empty(BATCH, C, 1, 1, device=x.device, dtype=x.dtype)
    
    # Use a simple kernel: one program per (n, c)
    N_total = BATCH * C
    
    # Use a simple reduction approach
    x_flat = x.reshape(BATCH, C, HW)
    out_flat = out.reshape(BATCH * C)
    
    grid = lambda meta: (N_total,)
    
    # Simple kernel: each thread block handles one (n,c) pair
    @triton.jit
    def pool_kernel(x_ptr, out_ptr, HW_val, N_total_val, BLOCK_HW: tl.constexpr):
        pid = tl.program_id(0)
        if pid >= N_total_val:
            return
        base = pid * HW_val
        acc = tl.zeros([1], dtype=tl.float32)
        for hw_start in range(0, HW_val, BLOCK_HW):
            hw_offs = hw_start + tl.arange(0, BLOCK_HW)
            mask = hw_offs < HW_val
            vals = tl.load(x_ptr + base + hw_offs, mask=mask, other=0.0)
            acc += tl.sum(vals, axis=0)
        tl.store(out_ptr + pid, acc / HW_val)
    
    # Can't define kernel inside function, use pre-defined one
    # Use the adaptive_avg_pool2d_kernel
    grid2 = (N_total,)
    adaptive_avg_pool2d_kernel[grid2](
        x_flat, out_flat,
        BATCH, C, HW,
    )
    
    return out


def triton_linear(x, w, bias):
    """Linear layer: x @ w.T + bias"""
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2
    
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    linear_kernel[grid](
        x, w, bias, out,
        M, N, K,
    )
    return out


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_mbconv1_0_weight, p_mbconv1_1_weight, p_mbconv1_1_bias,
    p_mbconv1_3_weight, p_mbconv1_4_weight, p_mbconv1_4_bias,