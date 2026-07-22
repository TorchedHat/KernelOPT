import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice



import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
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
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
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
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    tl.store(out_ptr + offsets, out, mask=mask)


# General Conv2D kernel using implicit GEMM
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = N * OUT_H * OUT_W, N = OUT_C, K = IN_C * KH * KW
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
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        in_c_idx = tmp_k // KH
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_x = valid_m & valid_k & valid_ih & valid_iw & (batch_idx[:, None] < N)
        
        x_idx = batch_idx[:, None] * (IN_C * IN_H * IN_W) + in_c_idx[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(x_ptr + x_idx, mask=valid_x, other=0.0)
        
        # Weight: [OUT_C, IN_C, KH, KW]
        valid_n = offs_n[None, :] < OUT_C
        valid_w = valid_k[:, None] & valid_n
        w_idx = offs_n[None, :] * K + offs_k[:, None]
        w_tile = tl.load(w_ptr + w_idx, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Store output
    valid_out_m = offs_m[:, None] < M
    valid_out_n = offs_n[None, :] < OUT_C
    valid_out = valid_out_m & valid_out_n & (batch_idx[:, None] < N)
    
    out_idx = batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h_idx[:, None] * OUT_W + out_w_idx[:, None]
    tl.store(out_ptr + out_idx, acc, mask=valid_out)


# MaxPool2D kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def maxpool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Decode output index
    ow = offsets % OUT_W
    tmp = offsets // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, x_val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


# AvgPool2D kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def avgpool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    ow = offsets % OUT_W
    tmp = offsets // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C
    
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = KH * KW
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh
            iw = ow * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            sum_val = sum_val + x_val
    
    tl.store(out_ptr + offsets, sum_val / count, mask=mask)


# Adaptive AvgPool2D (global) kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['numel', 'IN_H', 'IN_W'],
)
@triton.jit
def adaptive_avgpool2d_kernel(
    x_ptr, out_ptr,
    N, C, IN_H, IN_W,
    numel,
    IN_H_c: tl.constexpr, IN_W_c: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = offsets % C
    n = offsets // C
    
    sum_val = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for h in range(IN_H_c):
        for w in range(IN_W_c):
            x_idx = n * (C * IN_H_c * IN_W_c) + c * (IN_H_c * IN_W_c) + h * IN_W_c + w
            x_val = tl.load(x_ptr + x_idx, mask=mask, other=0.0)
            sum_val = sum_val + x_val
    
    tl.store(out_ptr + offsets, sum_val / (IN_H_c * IN_W_c), mask=mask)


# Linear kernel (GEMM: [M, K] x [K, N] -> [M, N])
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        x_tile = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :], mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](x, out, mean, var, weight, bias, numel, num_channels, spatial, eps)
    return out


def triton_bn(x, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](x, out, mean, var, weight, bias, numel, num_channels, spatial, eps)
    return out


def triton_conv2d(x, w, stride=(1,1), padding=(0,0)):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    STRIDE_H, STRIDE_W = stride
    PAD_H, PAD_W = padding
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    
    out = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW
    
    # Reshape weight to [OUT_C, K]
    w_flat = w.reshape(OUT_C, K).contiguous()
    x_cont = x.contiguous()
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    conv2d_kernel[grid](
        x_cont, w_flat, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M, K,
    )
    return out


def triton_maxpool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    STRIDE_H = STRIDE_W = stride
    PAD_H = PAD_W = padding
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = out.numel()
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
        numel,
    )
    return out


def triton_avgpool2d(x, kernel_size=2, stride=2):
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    STRIDE_H = STRIDE_W = stride
    OUT_H = (IN_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W - KW) // STRIDE_W + 1
    
    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = out.numel()
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    avgpool2d_kernel[grid](
        x, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, STRIDE_H, STRIDE_W,
        numel,
    )
    return out


def triton_adaptive_avgpool2d(x, output_size=(1, 1)):
    N, C, IN_H, IN_W = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    numel = N * C
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avgpool2d_kernel[grid](
        x, out,
        N, C, IN_H, IN_W,
        numel,
        IN_H, IN_W,
    )
    return out


def triton_linear(x, w, bias):
    # x: [M, K], w: [N, K], bias: [N]
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    w_t = w.contiguous()  # [N, K]
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, w_t, bias, out,
        M, N, K,
    )
    return out


def kernel_function(*args):
    # Unpack all parameters
    idx = 0
    p_features_0_weight = args[idx]; idx+=1
    p_features_1_weight = args[idx]; idx+=1
    p_features_1_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_0_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_0_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_0_2_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_1_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_1_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_1_2_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_2_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_2_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_2_2_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_3_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_3_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_3_2_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_4_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_4_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_4_2_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_5_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_5_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_5_2_weight = args[idx]; idx+=1
    p_transition_layers_0_transition_0_weight = args[idx]; idx+=1
    p_transition_layers_0_transition_0_bias = args[idx]; idx+=1
    p_transition_layers_0_transition_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_0_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_0_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_0_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_1_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_1_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_1_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_2_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_2_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_2_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_3_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_3_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_3_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_4_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_4_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_4_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_5_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_5_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_5_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_6_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_6_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_6_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_7_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_7_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_7_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_8_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_8_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_8_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_9_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_9_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_9_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_10_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_10_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_10_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_11_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_11_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_11_2_weight = args[idx]; idx+=1
    p_transition_layers_1_transition_0_weight = args[idx]; idx+=1
    p_transition_layers_1_transition_0_bias = args[idx]; idx+=1
    p_transition_layers_1_transition_2_weight = args[idx]; idx+=1
    
    # Dense block 2 (48 layers)
    db2_weights = []
    db2_biases = []
    db2_conv_weights = []
    for i in range(48):
        db2_weights.append(args[idx]); idx