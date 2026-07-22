import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Fused BatchNorm + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_elements, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    n_elements = N * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        n_elements, C, HW,
        eps,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements', 'C'],
)
@triton.jit
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_elements, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    tl.store(out_ptr + offsets, out, mask=mask)


def bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    n_elements = N * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        n_elements, C, HW,
        eps,
    )
    return out


# Implicit GEMM convolution kernel for 3x3 conv with padding=1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_3x3_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    M, N_out, K,
    STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = IN_C * 3 * 3
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % 3
        kh = offs_k // (IN_C * 3)
        
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = (batch[:, None] < N) & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        valid_k = offs_k[None, :] < K
        valid = valid_m & valid_k
        
        x_ptrs = X_ptr + batch[:, None] * stride_xn + in_c[None, :] * stride_xc + ih * stride_xh + iw * stride_xw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * stride_wco + in_c[:, None] * stride_wci + kh[:, None] * stride_wkh + kw[:, None] * stride_wkw
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + batch[:, None] * stride_yn + out_h[:, None] * stride_yh + out_w[:, None] * stride_yw + offs_n[None, :] * stride_yc
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_3x3(x, w, stride=1, padding=1):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1
    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW
    
    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    conv2d_3x3_kernel[grid](
        x, w, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        M, OUT_C, K,
        stride, stride, padding, padding,
    )
    return y


# 1x1 conv kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N_out', 'K'],
)
@triton.jit
def conv2d_1x1_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    M, N_out, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # For 1x1 conv: X reshaped to (N*H*W, IN_C), W reshaped to (IN_C, OUT_C)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        valid_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_ptrs = X_ptr + offs_m[:, None] * IN_C + offs_k[None, :]
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_k[:, None] * OUT_C + offs_n[None, :]
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + offs_m[:, None] * OUT_C + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_1x1(x, w):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C = w.shape[0]
    OUT_H, OUT_W = IN_H, IN_W
    M = N * IN_H * IN_W
    K = IN_C
    
    # Reshape x to (M, K) - need contiguous NHWC-like layout
    # x is (N, C, H, W), we need (N*H*W, C)
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
    x_flat = x_nhwc.reshape(M, K)
    w_flat = w.reshape(OUT_C, K).t().contiguous()  # (K, OUT_C)
    
    y_flat = torch.empty((M, OUT_C), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    conv2d_1x1_kernel[grid](
        x_flat, w_flat, y_flat,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        M, OUT_C, K,
    )
    
    y = y_flat.reshape(N, IN_H, IN_W, OUT_C).permute(0, 3, 1, 2).contiguous()
    return y


# 7x7 conv kernel (for initial conv)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N_out', 'K'],
)
@triton.jit
def conv2d_7x7_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    M, N_out, K,
    STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    KH: tl.constexpr, KW: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)
        
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = (batch[:, None] < N) & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        valid_k = offs_k[None, :] < K
        valid = valid_m & valid_k
        
        x_ptrs = X_ptr + batch[:, None] * stride_xn + in_c[None, :] * stride_xc + ih * stride_xh + iw * stride_xw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * stride_wco + in_c[:, None] * stride_wci + kh[:, None] * stride_wkh + kw[:, None] * stride_wkw
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + batch[:, None] * stride_yn + out_h[:, None] * stride_yh + out_w[:, None] * stride_yw + offs_n[None, :] * stride_yc
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_7x7(x, w, stride=2, padding=3):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1
    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW
    
    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    conv2d_7x7_kernel[grid](
        x, w, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        M, OUT_C, K,
        stride, stride, padding, padding,
        KH, KW,
    )
    return y


# Max pool 3x3 stride 2 padding 1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def max_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    # Decode: offs -> (n, c, oh, ow)
    ow = offs % OUT_W
    oh = (offs // OUT_W) % OUT_H
    c = (offs // (OUT_W * OUT_H)) % C
    n = offs // (OUT_W * OUT_H * C)
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_idx = n * C * IN_H * IN_W + c * IN_H * IN_W + ih * IN_W + iw
            x_val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, x_val)
    
    tl.store(Y_ptr + offs, max_val, mask=mask)


def max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + 2 * padding - kernel_size) // stride + 1
    OUT_W = (IN_W + 2 * padding - kernel_size) // stride + 1
    n_elements = N * C * OUT_H * OUT_W
    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride, padding, padding,
        n_elements,
    )
    return y


# Avg pool 2x2 stride 2
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def avg_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    
    ow = offs % OUT_W
    oh = (offs // OUT_W) % OUT_H
    c = (offs // (OUT_W * OUT_H)) % C
    n = offs // (OUT_W * OUT_H * C)
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = KH * KW
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh
            iw = ow * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_idx = n * C * IN_H * IN_W + c * IN_H * IN_W + ih * IN_W + iw
            x_val = tl.load(X_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)
            acc = acc + x_val
    
    out = acc / count
    tl.store(Y_ptr + offs, out, mask=mask)


def avg_pool2d(x, kernel_size=2, stride=2):
    N, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H - kernel_size) // stride + 1
    OUT_W = (IN_W - kernel_size) // stride + 1
    n_elements = N * C * OUT_H * OUT_W
    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    avg_pool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride,
        n_elements,
    )
    return y


# Adaptive avg pool to 1x1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
    ],
    key=['NC'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    X_ptr, Y_ptr,
    N, C, HW,
    NC,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NC
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(HW):
        x_idx = offs * HW + i
        x_val = tl.load