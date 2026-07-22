import torch
import triton
import triton.language as tl
import math

# [Full implementation follows]


import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused Conv2d + Bias + ReLU (implicit GEMM style)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
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
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    M, N_out, K,
    has_bias: tl.constexpr,
    apply_relu: tl.constexpr,
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
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = IN_C * KH * KW
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)
        
        ih = out_h_idx[:, None] * SH + kh[None, :] - PH
        iw = out_w_idx[:, None] * SW + kw[None, :] - PW
        
        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < N
        mask_x = valid_m & valid_k & valid_ih & valid_iw & valid_batch
        
        x_ptrs = X_ptr + (batch_idx[:, None] * stride_xn + 
                          in_c[None, :] * stride_xc + 
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n
        w_ptrs = W_ptr + (offs_n[None, :] * stride_wco + 
                          in_c[:, None] * stride_wci + 
                          kh[:, None] * stride_wkh + 
                          kw[:, None] * stride_wkw)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    if has_bias:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + bias[None, :]
    
    if apply_relu:
        acc = tl.maximum(acc, 0.0)
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    # Output in NCHW: Y[n, c, h, w] = Y[n * OUT_C * OUT_H * OUT_W + c * OUT_H * OUT_W + h * OUT_W + w]
    y_ptrs = Y_ptr + (batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) + 
                      offs_n[None, :] * (OUT_H * OUT_W) + 
                      out_h_idx[:, None] * OUT_W + 
                      out_w_idx[:, None])
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


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
    
    has_bias = bias is not None
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    
    conv2d_bias_relu_kernel[grid](
        y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        M, OUT_C, K,
        has_bias, apply_relu,
    )
    return y


# ============================================================
# Max Pool 2D kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_out'],
)
@triton.jit
def max_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    N_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_out
    
    # Decode: offs -> (n, c, oh, ow)
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C
    
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (n < N)
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
    N_out = N * C * OUT_H * OUT_W
    
    grid = lambda meta: (triton.cdiv(N_out, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W,
        OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        N_out,
    )
    return y


# ============================================================
# Adaptive Average Pool 2D (global average pool to 1x1)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N * C
    
    c = offs % C
    n = offs // C
    
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    total = IN_H * IN_W
    
    for h in range(IN_H):
        for w in range(IN_W):
            x_ptrs = X_ptr + n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + h * IN_W + w
            val = tl.load(x_ptrs, mask=mask, other=0.0)
            acc = acc + val
    
    acc = acc / total
    tl.store(Y_ptr + n * C + c, acc, mask=mask)


def adaptive_avg_pool2d_triton(x, output_size=(1, 1)):
    N, C, IN_H, IN_W = x.shape
    assert output_size == (1, 1), "Only 1x1 output supported"
    y = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(N * C, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x, y,
        N, C, IN_H, IN_W,
    )
    return y


# ============================================================
# Linear kernel (GEMM: Y = X @ W^T + b)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    has_bias: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        
        x_tile = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                         mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                         mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile.T, acc, allow_tf32=False)
    
    if has_bias:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


def linear_triton(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    has_bias = bias is not None
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    
    linear_kernel[grid](
        x, weight, bias if has_bias else x,
        y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        has_bias,
    )
    return y


# ============================================================
# ReLU kernel (pointwise)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def relu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offs, y, mask=mask)


def relu_triton(x):
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, y, N)
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
    
    # Branch pool
    pooled = max_pool2d_triton(x, kernel_size=3, stride=1, padding=1)
    branch_pool = conv2d_triton(pooled, w_pool, b_pool, stride=(1,1), padding=(0,0), apply_relu=True)
    
    return torch.cat([branch1, branch3, branch5, branch_pool], dim=1)


# ============================================================
# Main kernel function
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
    # Make sure inputs are contiguous
    x = x.contiguous()
    
    # conv1: 3->64, 7x7, stride=2, pad=3
    out = conv2d_triton(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), apply_relu=True)
    # maxpool1: 3x3, stride=2, pad=1
    out = max_pool2d_triton(out, kernel_size=3, stride=2, padding=1)
    # conv2: 64->64, 1x1
    out = conv2d_triton(out, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), apply_relu=True)
    # conv3: 64->192, 3x3, pad=1
    out = conv2d_triton(out, p_conv3_weight, p_conv3_bias, stride=(1,1), padding=(1,1), apply_relu=True)
    # maxpool2: 3x3, stride=2, pad=1
    out = max_pool2d_triton(out, kernel_size=3, stride=2, padding=1)
    
    # inception3a: in=192, out=256 (64+128+32+32)
    out = inception_module(out,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
        p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    )
    
    # inception3b: in=256, out=480 (128+192+96+64)
    out = inception_module(out,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
        p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,