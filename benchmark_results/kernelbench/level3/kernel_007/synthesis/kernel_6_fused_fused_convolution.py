import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused Conv2d + Bias + ReLU kernel (implicit GEMM)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    M, N_out, K,
    apply_relu: tl.constexpr,
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
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = k_offs % IN_C
        kw_idx = (k_offs // IN_C) % KW
        kh_idx = k_offs // (IN_C * KW)
        
        ih = out_h[:, None] * SH + kh_idx[None, :] - PH
        iw = out_w[:, None] * SW + kw_idx[None, :] - PW
        
        valid_x = (
            (batch[:, None] < N) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (k_offs[None, :] < K)
        )
        
        x_ptrs = (X_ptr + 
                  batch[:, None] * stride_xn + 
                  in_c[None, :] * stride_xc + 
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        valid_w = (k_offs[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr + 
                  offs_n[None, :] * stride_wn + 
                  in_c[:, None] * stride_wc + 
                  kh_idx[:, None] * stride_wh + 
                  kw_idx[:, None] * stride_ww)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]
    
    if apply_relu:
        acc = tl.maximum(acc, 0.0)
    
    # Store output: Y is (N, OUT_C, OUT_H, OUT_W) in NCHW
    valid_out = (
        (batch[:, None] < N) &
        (out_h[:, None] < OUT_H) &
        (out_w[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )
    
    y_ptrs = (Y_ptr + 
              batch[:, None] * (OUT_C * OUT_H * OUT_W) + 
              offs_n[None, :] * (OUT_H * OUT_W) + 
              out_h[:, None] * OUT_W + 
              out_w[:, None])
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def conv2d_triton(x, weight, bias, stride=(1,1), padding=(0,0), apply_relu=False):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding
    
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1
    
    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW
    
    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_bias_relu_kernel[grid](
        x, weight, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        M, OUT_C, K,
        apply_relu,
    )
    return y


# ============================================================
# Max Pool2d kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['total_elements'],
)
@triton.jit
def max_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements
    
    # Decode: offs -> (n, c, oh, ow)
    ow = offs % OUT_W
    oh = (offs // OUT_W) % OUT_H
    c = (offs // (OUT_W * OUT_H)) % C
    n = offs // (OUT_W * OUT_H * C)
    
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(Y_ptr + offs, max_val, mask=mask)


def max_pool2d_triton(x, kernel_size=3, stride=2, padding=1):
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1
    
    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    total = N * C * OUT_H * OUT_W
    
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        total,
    )
    return y


# ============================================================
# Adaptive Average Pool2d kernel (global average pool to 1x1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'IN_H', 'IN_W'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C
    
    if n >= N:
        return
    
    total = IN_H * IN_W
    acc = tl.zeros((1,), dtype=tl.float32)
    
    base = X_ptr + n * (C * IN_H * IN_W) + c * (IN_H * IN_W)
    
    for start in range(0, total, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        valid = offs < total
        val = tl.load(base + offs, mask=valid, other=0.0)
        acc += tl.sum(val, axis=0)
    
    tl.store(Y_ptr + n * C + c, acc[0] / total)


def adaptive_avg_pool2d_triton(x):
    N, C, IN_H, IN_W = x.shape
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    
    grid = (N * C,)
    adaptive_avg_pool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W,
    )
    return y


# ============================================================
# Linear kernel (using tl.dot)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        
        x_ptrs = X_ptr + offs_m[:, None] * K + k_offs[None, :]
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # W is (N, K), we need W^T effectively: Y = X @ W^T
        w_ptrs = W_ptr + offs_n[None, :] * K + k_offs[:, None]
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)
    
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def linear_triton(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    M, K = x.shape
    N = weight.shape[0]
    
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    linear_kernel[grid](
        x, weight, bias, y,
        M, N, K,
    )
    return y


# ============================================================
# ReLU kernel (standalone, for cases where conv doesn't fuse)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(X_ptr, Y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < n_elements
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offs, y, mask=mask)


def relu_triton(x):
    y = torch.empty_like(x)
    n = x.numel()
    grid = lambda meta: (triton.cdiv(n, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, y, n)
    return y


# ============================================================
# Inception module helper
# ============================================================
def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    # Branch 1x1
    branch1 = conv2d_triton(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), apply_relu=True)
    
    # Branch 3x3
    branch3_r = conv2d_triton(x, w_3x3_r, b_3x3_r, stride=(1,1), padding=(0,0), apply_relu=True)
    branch3 = conv2d_triton(branch3_r, w_3x3, b_3x3, stride=(1,1), padding=(1,1), apply_relu=True)
    
    # Branch 5x5
    branch5_r = conv2d_triton(x, w_5x5_r, b_5x5_r, stride=(1,1), padding=(0,0), apply_relu=True)
    branch5 = conv2d_triton(branch5_r, w_5x5, b_5x5, stride=(1,1), padding=(2,2), apply_relu=True)
    
    # Branch pool: max_pool then 1x1 conv
    pool = max_pool2d_triton(x, kernel_size=3, stride=1, padding=1)
    branch_pool = conv2d_triton(pool, w_pool, b_pool, stride=(1,1), padding=(0,0), apply_relu=True)
    
    return torch.cat([branch1, branch3, branch5, branch_pool], dim=1)


# ============================================================
# Main kernel_function
# ============================================================
def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_conv3_weight, p_conv3_bias,
    p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
    p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
    p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
    p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
    p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
    p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
    p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
    p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
    p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
    p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
    p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
    p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
    p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
    p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
    p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
    p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
    p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
    p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
    p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
    p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
    p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
    p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
    p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
    p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
    p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
    p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias,
    p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
    p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
    p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
    p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
    p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
    p_inception4d_branch_pool_1_weight, p_inception4d_branch_pool_1_bias,
    p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
    p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
    p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
    p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
    p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
    p_inception4e_branch_pool_1_weight, p_inception4e_branch_pool_1_bias,
    p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
    p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
    p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
    p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
    p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
    p_inception5a_branch_pool_1_weight, p_inception5a_branch_pool_1_bias,
    p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
    p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
    p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
    p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
    p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
    p_inception5b_branch_pool_1_weight, p_inception5b_branch_pool_1_bias,
    p_fc_weight, p_fc_bias,
    x,
):
    # Initial convolutions
    x = conv2d_triton(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), apply_relu=True)
    x = max_pool2d_triton(x, kernel_size=3, stride=2, padding=1)
    x = conv2d_triton(x, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), apply_relu=True)
    x = conv2d_triton(x, p_conv3_weight, p_conv3_bias, stride=(1,1), padding=(1,1), apply_relu=True)
    x = max_pool2d_triton(x, kernel_size=3, stride=2, padding=1)
    
    # Inception 3a
    x = inception_module(x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
        p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    )
    
    # Inception 3b
    x = inception_module(x,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
        p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
        p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
        p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
        p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    )
    
    x = max_pool2d_triton(x, kernel_size=3, stride=2, padding=1)
    
    # Inception 4a
    x = inception_module(x,
        p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
        p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
        p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
        p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
        p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
        p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    )
    
    # Inception 4b
    x = inception_module(x,
        p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
        p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
        p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
        p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,