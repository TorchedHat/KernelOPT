import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = N * C * H * W
    spatial = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial,
        eps,
    )
    return out


# Fused BatchNorm (no ReLU) kernel  
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def fused_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = N * C * H * W
    spatial = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial,
        eps,
    )
    return out


# Implicit GEMM convolution kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def implicit_gemm_conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N_out, K,
    s_xn, s_xc, s_xh, s_xw,
    s_wn, s_wc, s_wh, s_ww,
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
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KW
        kh_idx = offs_k // (IN_C * KW)
        
        ih = out_h[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W
        
        valid_x = (
            (batch[:, None] < N) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < K)
        )
        
        x_ptrs = X_ptr + (
            batch[:, None] * s_xn +
            in_c[None, :] * s_xc +
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + (
            offs_n[None, :] * s_wn +
            in_c[:, None] * s_wc +
            kh_idx[:, None] * s_wh +
            kw_idx[:, None] * s_ww
        )
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Store output
    valid_out = (
        (offs_m[:, None] < M) &
        (offs_n[None, :] < OUT_C)
    )
    
    y_ptrs = Y_ptr + (
        batch[:, None] * (OUT_C * OUT_H * OUT_W) +
        offs_n[None, :] * (OUT_H * OUT_W) +
        out_h[:, None] * OUT_W +
        out_w[:, None]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def triton_conv2d(x, w, stride=(1,1), padding=(0,0)):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
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
    
    implicit_gemm_conv2d_kernel[grid](
        x, w, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        M, OUT_C, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
    )
    return y


# Max pool kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Decode output index
    ow = offsets % OUT_W
    oh = (offsets // OUT_W) % OUT_H
    c = (offsets // (OUT_W * OUT_H)) % C
    n = offsets // (OUT_W * OUT_H * C)
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (n < N)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = N * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        numel,
    )
    return out


# Avg pool kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, SH, SW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    ow = offsets % OUT_W
    oh = (offsets // OUT_W) % OUT_H
    c = (offsets // (OUT_W * OUT_H)) % C
    n = offsets // (OUT_W * OUT_H * C)
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = KH * KW
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh
            iw = ow * SW + kw
            valid = mask & (ih < IN_H) & (iw < IN_W) & (n < N)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            acc = acc + val
    
    out_val = acc / count
    tl.store(out_ptr + offsets, out_val.to(tl.float32), mask=mask)


def triton_avg_pool2d(x, kernel_size=2, stride=2):
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    OUT_H = (IN_H - KH) // SH + 1
    OUT_W = (IN_W - KW) // SW + 1
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = N * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    avg_pool2d_kernel[grid](
        x, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW,
        numel,
    )
    return out


# Adaptive avg pool kernel (global)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['numel', 'spatial'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    numel, spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = offsets % C
    n = offsets // C
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for h in range(H):
        for w in range(W):
            x_idx = n * (C * H * W) + c * (H * W) + h * W + w
            val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
            acc = acc + val
    
    out_val = acc / spatial
    tl.store(out_ptr + offsets, out_val.to(tl.float32), mask=mask)


def triton_adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    numel = N * C
    spatial = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        numel, spatial,
    )
    return out


# Linear kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_linear(x, w, b):
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, w, b, out,
        M, N, K,
    )
    return out


def kernel_function(
    p_features_0_weight,
    p_features_1_weight,
    p_features_1_bias,
    p_dense_blocks_0_layers_0_0_weight,
    p_dense_blocks_0_layers_0_0_bias,
    p_dense_blocks_0_layers_0_2_weight,
    p_dense_blocks_0_layers_1_0_weight,
    p_dense_blocks_0_layers_1_0_bias,
    p_dense_blocks_0_layers_1_2_weight,
    p_dense_blocks_0_layers_2_0_weight,
    p_dense_blocks_0_layers_2_0_bias,
    p_dense_blocks_0_layers_2_2_weight,
    p_dense_blocks_0_layers_3_0_weight,
    p_dense_blocks_0_layers_3_0_bias,
    p_dense_blocks_0_layers_3_2_weight,
    p_dense_blocks_0_layers_4_0_weight,
    p_dense_blocks_0_layers_4_0_bias,
    p_dense_blocks_0_layers_4_2_weight,
    p_dense_blocks_0_layers_5_0_weight,
    p_dense_blocks_0_layers_5_0_bias,
    p_dense_blocks_0_layers_5_2_weight,
    p_transition_layers_0_transition_0_weight,
    p_transition_layers_0_transition_0_bias,
    p_transition_layers_0_transition_2_weight,
    p_dense_blocks_1_layers_0_0_weight,
    p_dense_blocks_1_layers_0_0_bias,
    p_dense_blocks_1_layers_0_2_weight,
    p_dense_blocks_1_layers_1_0_weight,
    p_dense_blocks_1_layers_1_0_bias,
    p_dense_blocks_1_layers_1_2_weight,
    p_dense_blocks_1_layers_2_0_weight,
    p_dense_blocks_1_layers_2_0_bias,
    p_dense_blocks_1_layers_2_2_weight,
    p_dense_blocks_1_layers_3_0_weight,
    p_dense_blocks_1_layers_3_0_bias,
    p_dense_blocks_1_layers_3_2_weight,
    p_dense_blocks_1_layers_4_0_weight,
    p_dense_blocks_1_layers_4_0_bias,
    p_dense_blocks_1_layers_4_2_weight,
    p_dense_blocks_1_layers_5_0_weight,
    p_dense_blocks_1_layers_5_0_bias,
    p_dense_blocks_1_layers_5_2_weight,
    p_dense_blocks_1_layers_6_0_weight,
    p_dense_blocks_1_layers_6_0_bias,
    p_dense_blocks_1_layers_6_2_weight,
    p_dense_blocks_1_layers_7_0_weight,
    p_dense_blocks_1_layers_7_0_bias,
    p_dense_blocks_1_layers_7_2_weight,
    p_dense_blocks_1_layers_8_0_weight,
    p_dense_blocks_1_layers_8_0_bias,
    p_dense_blocks_1_layers_8_2_weight,
    p_dense_blocks_1_layers_9_0_weight,
    p_dense_blocks_1_layers_9_0_bias,
    p_dense_blocks_1_layers_9_2_weight,
    p_dense_blocks_1_layers_10_0_weight,
    p_dense_blocks_1_layers_10_0_bias,
    p_dense_blocks_1_layers_10_2_weight,
    p_dense_blocks_1_layers_11_0_weight,
    p_dense_blocks_1_layers_11_0_bias,
    p_dense_blocks_1_layers_11_2_weight,
    p_transition_layers_1_transition_0_weight,
    p_transition_layers_1_transition_0_bias,
    p_transition_layers_1_transition_2_weight,
    p_dense_blocks_2_layers_0_0_weight,
    p_dense_blocks_2_layers_0_0_bias,
    p_dense_blocks_2_layers_0_2_weight,
    p_dense_blocks_2_layers_1_0_weight,
    p_dense_blocks_2_layers_1_0_bias,
    p_dense_blocks_2_layers_1_2_weight,
    p_dense_blocks_2_layers_2_0_weight,
    p_dense_blocks_2_layers_2_0_bias,
    p_dense_blocks_2_layers_2_2_weight,
    p_dense_blocks_2_layers_3_0_weight,
    p_dense_blocks_2_layers_3_0_bias,
    p_dense_blocks_2_layers_3_2_weight,
    p_dense_blocks_2_layers_4_0_weight,
    p_dense_blocks_2_layers_4_0_bias,
    p_dense_blocks_2_layers_4_2_weight,
    p_dense_blocks_2_layers_5_0_weight,
    p_dense_blocks_2_layers_5_0_bias,
    p_dense_blocks_2_layers_5_2_weight,