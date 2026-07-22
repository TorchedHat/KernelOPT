import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm + ReLU pointwise kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
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
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
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


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    """Apply fused BatchNorm + optional ReLU using Triton."""
    out = torch.empty_like(x)
    numel = x.numel()
    N, C, H, W = x.shape
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        numel, C, HW,
        eps,
        do_relu,
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


# Implicit GEMM Conv2d kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, K,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (OUT_C, IN_C, KH, KW) -> flattened
    s_wn,
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
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KW
        kh = tmp_k // KW
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw
        
        x_ptrs = X_ptr + (
            batch_idx[:, None] * s_xn +
            in_c[None, :] * s_xc
        )
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # Weight: shape (OUT_C, K) where K = IN_C * KH * KW
        # W stored as (OUT_C, IN_C, KH, KW), stride s_wn per output channel
        w_ptrs = W_ptr + offs_n[None, :] * s_wn + offs_k[:, None]
        mask_w = (offs_n[None, :] < OUT_C) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store output in NCHW layout
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + (
        batch_idx[:, None] * OUT_C * OUT_H * OUT_W +
        offs_n[None, :] * OUT_H * OUT_W +
        out_h_idx[:, None] * OUT_W +
        out_w_idx[:, None]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_conv2d(x, weight, stride=(1,1), padding=(0,0), groups=1):
    """
    """
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, IN_C_per_group, KH, KW = weight.shape
    
    STRIDE_H, STRIDE_W = stride
    PAD_H, PAD_W = padding
    
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    
    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    if groups == 1:
        # Convert x to NHWC for coalesced access
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        # Flatten weight to (OUT_C, IN_C * KH * KW)
        w_flat = weight.reshape(OUT_C, -1).contiguous()
        
        M = N * OUT_H * OUT_W
        K = IN_C * KH * KW
        
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )
        
        conv2d_implicit_gemm_kernel[grid](
            x_nhwc, w_flat, y,
            N, IN_C, IN_H, IN_W,
            OUT_C, OUT_H, OUT_W,
            KH, KW,
            STRIDE_H, STRIDE_W,
            PAD_H, PAD_W,
            M, K,
            x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
            w_flat.stride(0),
        )
    else:
        # Depthwise convolution (groups == IN_C == OUT_C)
        # Handle each group separately or use specialized kernel
        # For depthwise: groups = IN_C, IN_C_per_group = 1
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        w_flat = weight.reshape(OUT_C, -1).contiguous()  # (OUT_C, KH*KW)
        
        M = N * OUT_H * OUT_W
        K_per_group = IN_C_per_group * KH * KW  # = KH * KW for depthwise
        
        # For depthwise, we process each output channel independently
        # Use a specialized depthwise kernel
        _triton_depthwise_conv2d(x_nhwc, w_flat, y, N, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W, KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W, groups)
    
    return y


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'OUT_C', 'KH', 'KW'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M,
    s_xn, s_xh, s_xw, s_xc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Depthwise conv2d: groups == IN_C == OUT_C, IN_C_per_group == 1"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H
    
    mask_m = offs_m < M
    mask_n = offs_n < OUT_C
    
    # For depthwise: each output channel c uses weight[c, 0, :, :] and input channel c
    # We iterate over KH * KW
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = out_h_idx[:, None] * STRIDE_H + kh - PAD_H
            iw = out_w_idx[:, None] * STRIDE_W + kw - PAD_W
            
            valid_ih = (ih >= 0) & (ih < IN_H)
            valid_iw = (iw >= 0) & (iw < IN_W)
            mask_x = mask_m[:, None] & valid_ih & valid_iw
            
            # Load input: x[batch, ih, iw, c] for each c in offs_n
            x_ptrs = X_ptr + (
                batch_idx[:, None] * s_xn +
                offs_n[None, :] * s_xc
            )
            x_vals = tl.load(x_ptrs, mask=mask_x & mask_n[None, :], other=0.0)
            
            # Load weight: w[c, kh*KW + kw] for each c in offs_n
            w_ptrs = W_ptr + offs_n * (KH * KW) + kh * KW + kw
            w_vals = tl.load(w_ptrs, mask=mask_n, other=0.0)
            
            acc += x_vals * w_vals[None, :]
    
    # Store output
    mask_out = mask_m[:, None] & mask_n[None, :]
    y_ptrs = Y_ptr + (
        batch_idx[:, None] * OUT_C * OUT_H * OUT_W +
        offs_n[None, :] * OUT_H * OUT_W +
        out_h_idx[:, None] * OUT_W +
        out_w_idx[:, None]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def _triton_depthwise_conv2d(x_nhwc, w_flat, y, N, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W, KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W, groups):
    M = N * OUT_H * OUT_W
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    depthwise_conv2d_kernel[grid](
        x_nhwc, w_flat, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
    )


# Linear layer using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
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
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        w_ptrs = W_ptr + offs_n[None, :] * K + offs_k[:, None]
        mask_w = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_linear(x, weight, bias):
    """x: (M, K), weight: (N, K), bias: (N,) -> (M, N)"""
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, weight, bias, y, M, N, K)
    return y


# Adaptive avg pool 2d -> (N, C, 1, 1)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    """Reduce spatial dims to 1x1."""
    pid = tl.program_id(0)
    # Each program handles one (batch, channel) pair
    nc = pid
    if nc >= N * C:
        return
    
    n = nc // C
    c = nc % C
    
    base = n * C * HW + c * HW
    
    acc = tl.zeros((), dtype=tl.float32)
    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)
    
    tl.store(out_ptr + nc, acc / HW)


def triton_adaptive_avg_pool2d(x):
    """x: (N, C, H, W) -> (N, C, 1, 1)"""
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    adaptive_avg_pool2d_kernel[grid](x, out, N, C, HW)
    return out


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
    p_mbconv3_0_weight, p_mbconv3_1_weight, p_mbconv3_1_bias,
    p_mbconv3_3_weight, p_mbconv3_4_weight, p_mbconv3_4_bias,
    p_mbconv3_7_weight, p_mbconv3_9_weight,
    p_mbconv3_11_weight, p_mbconv3_12_weight, p_mbconv3_12_bias,
    p_mbconv4_0_weight, p_mbconv4_1_weight, p_mbconv4_1_bias,
    p_mbconv4_3_weight, p_mbconv4_4_weight, p_mbconv4_4_bias,
    p_mbconv4_7_weight, p_mbconv4_9_weight,
    p_mbconv4_11_weight, p_mbconv4_12_weight, p_mbconv4_12_bias,
    p_mbconv5_0_weight, p_mbconv5_1_weight, p_mbconv5_1_bias,
    p_mbconv5_3_weight, p_mbconv5_4_weight, p_mbconv5_4_bias,
    p_mbconv5_7_weight, p_mbconv5_9_weight,
    p_mbconv5_11_weight, p_mbconv5_12_weight, p_mbconv5_12_bias,
    p_conv_final_weight, p_bn_final_weight, p_bn_final_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_mbconv1_1_running_mean, b_mbconv1_1_running_var,
    b_mbconv1_4_running_mean, b_mbconv1_4_running_var,