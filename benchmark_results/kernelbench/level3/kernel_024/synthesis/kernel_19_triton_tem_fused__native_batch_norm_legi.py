import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

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
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    if apply_relu:
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
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
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
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Adaptive Average Pool 2D (global) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_warps=2),
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
    nc = pid
    n = nc // C
    c = nc % C
    
    if n >= N:
        return
    
    acc = tl.zeros((), dtype=tl.float32)
    base = n * C * HW + c * HW
    
    for hw_off in range(0, HW, BLOCK_HW):
        hw_idx = hw_off + tl.arange(0, BLOCK_HW)
        mask = hw_idx < HW
        x = tl.load(x_ptr + base + hw_idx, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x, axis=0)
    
    out = acc / HW
    tl.store(out_ptr + n * C + c, out)


# ============================================================
# Linear (GEMM) kernel: y = x @ W^T + b
# x: [M, K], W: [N, K], b: [N], y: [M, N]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
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
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc=acc)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + b[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Conv2d implicit GEMM kernel (1x1 convolution)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
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
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc=acc)
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Conv2d 3x3 depthwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 64}, num_warps=8),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    
    M_TOTAL = N * OUT_H * OUT_W
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    n_idx = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)
    
    for kh in range(3):
        for kw in range(3):
            ih = out_h_idx * STRIDE_H + kh - PAD_H
            iw = out_w_idx * STRIDE_W + kw - PAD_W
            
            valid_m = (offs_m < M_TOTAL) & (n_idx < N) & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            valid_c = offs_c < C
            
            x_ptrs = x_ptr + n_idx[:, None] * C * IN_H * IN_W + offs_c[None, :] * IN_H * IN_W + ih[:, None] * IN_W + iw[:, None]
            x_mask = valid_m[:, None] & valid_c[None, :]
            x = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)
            
            w_ptrs = w_ptr + offs_c * 9 + kh * 3 + kw
            w_mask = valid_c
            w = tl.load(w_ptrs, mask=w_mask, other=0.0).to(tl.float32)
            
            acc += x * w[None, :]
    
    out_ptrs = out_ptr + n_idx[:, None] * C * OUT_H * OUT_W + offs_c[None, :] * OUT_H * OUT_W + out_h_idx[:, None] * OUT_W + out_w_idx[:, None]
    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_c[None, :] < C)
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Conv2d 3x3 standard kernel (for initial conv)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    BATCH, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    M_TOTAL = BATCH * OUT_H * OUT_W
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    n_idx = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * 9
    
    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % 3
        kh = offs_k // (IN_C * 3)
        
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid = (offs_m[:, None] < M_TOTAL) & (n_idx[:, None] < BATCH) & \
                (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & \
                (offs_k[None, :] < TOTAL_K)
        
        x_ptrs = x_ptr + n_idx[:, None] * IN_C * IN_H * IN_W + in_c[None, :] * IN_H * IN_W + ih * IN_W + iw
        x = tl.load(x_ptrs, mask=valid, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * IN_C * 9 + in_c[:, None] * 9 + kh[:, None] * 3 + kw[:, None]
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc=acc)
    
    out_ptrs = out_ptr + n_idx[:, None] * OUT_C * OUT_H * OUT_W + offs_n[None, :] * OUT_H * OUT_W + out_h_idx[:, None] * OUT_W + out_w_idx[:, None]
    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Multiply (SE attention) kernel: out = x * se_weight
# x: [N, C, H, W], se: [N, C, 1, 1]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'HW'],
)
@triton.jit
def se_multiply_kernel(
    x_ptr, se_ptr, out_ptr,
    numel, N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    hw = offsets % HW
    c = (offsets // HW) % C
    n = offsets // (C * HW)
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    se = tl.load(se_ptr + n * C + c, mask=mask, other=0.0).to(tl.float32)
    
    out = x * se
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5, apply_relu=True):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def triton_relu(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def triton_sigmoid(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


def triton_adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    grid = (N * C,)
    adaptive_avg_pool2d_kernel[grid](
        x, out.view(N, C),
    )
    return out


def triton_conv1x1(x, w):
    # x: [N, C_in, H, W] -> reshape to [N*H*W, C_in]
    # w: [C_out, C_in, 1, 1] -> reshape to [C_out, C_in]
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    M = N * H * W
    K = C_in
    
    x_flat = x.permute(0, 2, 3, 1).contiguous().view(M, K)
    w_flat = w.view(C_out, K)
    
    out_flat = torch.empty(M, C_out, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv1x1_kernel[grid](
        x_flat, w_flat, out_flat,
    )
    
    return out_flat.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def triton_depthwise_conv3x3(x, w, stride=1, padding=1):
    N, C, H, W = x.shape
    OUT_H = (H + 2 * padding - 3) // stride + 1
    OUT_W = (W + 2 * padding - 3) // stride + 1
    
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    M_TOTAL = N * OUT_H * OUT_W
    
    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(C, meta['BLOCK_C'])
    )
    
    depthwise_conv3x3_kernel[grid](
        x, w, out,
        N, C, H, W, OUT_H, OUT_W,
    )
    
    return out


def triton_conv3x3(x, w, stride=2, padding=1):
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    OUT_H = (H + 2 * padding - 3) // stride + 1
    OUT_W = (W + 2 * padding - 3) // stride + 1
    
    out = torch.empty(N, C_out, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    M_TOTAL = N * OUT_H * OUT_W
    
    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N'])
    )
    
    conv3x3_kernel[grid](
        x, w, out,
        N, C_in, H, W, C_out, OUT_H, OUT_W,
    )
    
    return out


def triton_linear(x, w, b):
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, w, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1)
    )
    return out


def triton_se_multiply(x, se):
    N, C, H, W = x.shape
    numel = x.numel()
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    se_multiply_kernel[grid](
        x, se.view(N, C), out,
    )
    return out


def mbconv_block(x, 
                 expand_w, expand_bn_w, expand_bn_b, expand_bn_mean, expand_bn_var,
                 dw_w, dw_bn_w, dw_bn_b, dw_bn_mean, dw_bn_var,
                 se_reduce_w, se_expand_w,
                 proj_w, proj_bn_w, proj_bn_b, proj_bn_mean, proj_bn_var,
                 stride, expand_ratio, eps=1e-5):
    # Expansion
    if expand_ratio != 1:
        x = triton_conv1x1(x, expand_w)
        x = triton_bn_relu(x, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b, eps, apply_relu=True)
    
    # Depthwise conv
    x = triton_depthwise_conv3x3(x, dw