import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Fused BatchNorm + ReLU6 kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu6_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr, out_ptr,
    numel, C, spatial,
    eps: tl.constexpr,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // spatial) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu6(x, running_mean, running_var, weight, bias, apply_relu6=True, eps=1e-5):
    N, C, H, W = x.shape
    spatial = H * W
    numel = N * C * H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu6_kernel[grid](
        x, running_mean, running_var, weight, bias, out,
        numel, C, spatial,
        eps=eps,
        apply_relu6=apply_relu6,
    )
    return out


# General Conv2d Triton kernel (implicit GEMM)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    GROUPS,
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
    out_w_idx = offs_m % OUT_W
    out_h_idx = (offs_m // OUT_W) % OUT_H
    batch_idx = offs_m // (OUT_W * OUT_H)
    
    # Group info
    GROUP_OUT_C = OUT_C // GROUPS
    GROUP_IN_C = IN_C // GROUPS
    group_idx = offs_n // GROUP_OUT_C
    local_out_c = offs_n % GROUP_OUT_C
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = GROUP_IN_C * KH * KW
    
    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c_local, kh, kw)
        in_c_local = offs_k % GROUP_IN_C
        kw_idx = (offs_k // GROUP_IN_C) % KW
        kh_idx = offs_k // (GROUP_IN_C * KW)
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W
        
        # Global in_c
        in_c_global = in_c_local + group_idx[None, :] * GROUP_IN_C  # shape issues - need to handle groups
        # Actually for simplicity, handle groups=1 or depthwise separately
        # For groups=1: in_c_global = in_c_local
        # For depthwise (groups=IN_C): each output channel corresponds to one input channel
        
        # Let's handle the general case:
        # For each output channel offs_n, group = offs_n // GROUP_OUT_C
        # input channels for that group: [group*GROUP_IN_C, (group+1)*GROUP_IN_C)
        # But BLOCK_N may span multiple groups - this is complex
        # For simplicity, assume single group or depthwise
        
        # X load: shape [BLOCK_M, BLOCK_K]
        # For group g, input channel = g * GROUP_IN_C + in_c_local
        # We need per-output-channel group info
        # group_idx has shape [BLOCK_N], in_c_local has shape [BLOCK_K]
        # ih, iw have shape [BLOCK_M, BLOCK_K]
        
        # For the X pointer: batch_idx[BLOCK_M], in_c_global needs group from output side
        # This is tricky with mixed groups in a tile
        # Simplification: assume GROUPS=1 or GROUPS=IN_C (depthwise)
        
        # For GROUPS=1:
        in_c_g = in_c_local  # shape [BLOCK_K]
        
        valid_k = offs_k < TOTAL_K
        valid_m = (batch_idx < N) & (ih[:, 0] >= -PAD_H)  # rough check
        
        x_valid = (
            (batch_idx[:, None] < N) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            valid_k[None, :]
        )
        
        x_ptrs = (X_ptr + 
                  batch_idx[:, None] * s_xn + 
                  in_c_g[None, :] * s_xc + 
        
        x_tile = tl.load(x_ptrs, mask=x_valid, other=0.0)
        
        # W load: shape [BLOCK_K, BLOCK_N]
        # w[out_c, in_c_local, kh, kw]
        w_valid = (
            valid_k[:, None] &
            (offs_n[None, :] < OUT_C)
        )
        
        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_wn +
                  in_c_local[:, None] * s_wc +
                  kh_idx[:, None] * s_wh +
                  kw_idx[:, None] * s_ww)
        
        w_tile = tl.load(w_ptrs, mask=w_valid, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store
    out_mask = (
        (batch_idx[:, None] < N) &
        (out_h_idx[:, None] < OUT_H) &
        (out_w_idx[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )
    
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
              offs_n[None, :] * (OUT_H * OUT_W) +
              out_h_idx[:, None] * OUT_W +
              out_w_idx[:, None])
    
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_conv2d(x, w, stride=(1,1), padding=(0,0), groups=1):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, W_IN_C, KH, KW = w.shape
    
    stride_h, stride_w = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    pad_h, pad_w = padding if isinstance(padding, (tuple, list)) else (padding, padding)
    
    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1
    
    GROUP_IN_C = IN_C // groups
    GROUP_OUT_C = OUT_C // groups
    K = GROUP_IN_C * KH * KW
    M = N * OUT_H * OUT_W
    
    y = torch.zeros(N, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    if groups > 1 and groups == IN_C:
        # Depthwise convolution - handle specially
        return triton_depthwise_conv2d(x, w, stride=(stride_h, stride_w), padding=(pad_h, pad_w))
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x, w, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        groups,
        M, OUT_C, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
    )
    return y


# Depthwise conv2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W', 'KH', 'KW'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OUT_H * OUT_W
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    # Decode: offsets -> (n, c, oh, ow)
    ow = offsets % OUT_W
    oh = (offsets // OUT_W) % OUT_H
    c = (offsets // (OUT_W * OUT_H)) % C
    n = offsets // (OUT_W * OUT_H * C)
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            w_idx = c * (KH * KW) + kh * KW + kw
            
            x_val = tl.load(X_ptr + x_idx, mask=valid, other=0.0)
            w_val = tl.load(W_ptr + w_idx, mask=mask, other=0.0)
            
            acc += x_val * w_val
    
    tl.store(Y_ptr + offsets, acc, mask=mask)


def triton_depthwise_conv2d(x, w, stride=(1,1), padding=(0,0)):
    N, C, IN_H, IN_W = x.shape
    _, _, KH, KW = w.shape
    
    stride_h, stride_w = stride
    pad_h, pad_w = padding
    
    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1
    
    y = torch.zeros(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    
    total = N * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    
    depthwise_conv2d_kernel[grid](
        x, w, y,
        N, C, IN_H, IN_W,
        OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
    )
    return y


# Linear kernel (GEMM)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
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
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w_tile = tl.load(W_ptr + offs_k[:, None] * N + offs_n[None, :], mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def triton_linear(x, w, b):
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2
    
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, w, b, y, M, N, K)
    return y


# Adaptive avg pool kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'IN_H', 'IN_W'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OUT_H * OUT_W
    
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    ow = offsets % OUT_W
    oh = (offsets // OUT_W) % OUT_H
    c = (offsets // (OUT_W * OUT_H)) % C
    n = offsets // (OUT_W * OUT_H * C)
    
    # Compute input range for this output position
    ih_start = oh * IN_H // OUT_H
    ih_end = (oh + 1) * IN_H // OUT_H
    iw_start = ow * IN_W // OUT_W
    iw_end = (ow + 1) * IN_W // OUT_W
    
    pool_h = ih_end - ih_start
    pool_w = iw_end - iw_start
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = pool_h * pool_w
    
    for dh in range(7):  # max pool size
        for dw in range(7):
            ih = ih_start + dh
            iw = iw_start + dw
            valid = mask & (dh < pool_h) & (dw < pool_w)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            x_val = tl.load(X_ptr + x_idx, mask=valid, other=0.0)
            acc += x_val
    
    out = acc / count.to(tl.float32)
    tl.store(Y_ptr + offsets, out, mask=mask)


def triton_adaptive_avg_pool2d(x, output_size=(1, 1)):
    N, C, IN_H, IN_W = x.shape
    OUT_H, OUT_W = output_size
    
    y = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    total = N * C * OUT_H * OUT_W
    
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](x, y, N, C, IN_H, IN_W, OUT_H, OUT_W)
    return y


def kernel_function(
    p_features_0_weight, p_features_1_weight, p_features_1_bias,
    p_features_3_0_weight, p_features_3_1_weight, p_features_3_1_bias,
    p_features_3_3_weight, p_features_3_4_weight, p_features_3_4_bias,
    p_features_4_0_weight, p_features_4_1_weight, p_features_4_1_bias,
    p_features_4_3_weight, p_features_4_4_weight, p_features_4_4_bias,
    p_features_4_6_weight, p_features_4_7_weight, p_features_4_7_bias,
    p_features_5_0_weight, p_features_5_1_weight, p_features_5_1_bias,
    p_features_5_3_weight, p_features_5_4_weight, p_features_5_4_bias,
    p_features_5_6_weight, p_features_5_7_weight, p_features_5_7_bias,
    p_features_6_0_weight, p_features_6_1_weight, p_features_6_1_bias,
    p_features_6_3_weight, p_features_6_4_weight, p_features_6_4_bias,
    p_features_6_6_weight, p_features_6_7_weight, p_features_6_7_bias,
    p_features_7_0_weight, p_features_7_1_weight, p_features_7_1_bias,
    p_features_7_3_weight, p_features_7_4_weight, p_features_7_4_bias,
    p_features_7_6_weight, p_features_7_7_weight, p_features_7_7_bias,
    p_features_8_0_weight, p_features_8_1_weight, p_features_8_1_bias,
    p_features_8_3_weight, p_features_8_4_weight, p_features_8_4_bias,
    p_features_8_6_weight, p_features_8_7_weight, p_features_8_7_bias,
    p_features_9_0_weight, p_features_9_1_weight, p_features_9_1_bias,
    p_features_9_3_weight, p_features_9_4_weight, p_features_9_4_bias,
    p_features_9_6_weight, p_features_9_7_weight, p_features_9_7_bias,
    p_features_10_0_weight, p_features_10_1_weight, p_features_10_1_bias,
    p_features_10_3_weight, p_features_10_4_weight, p_features_10_4_bias,
    p_features_10_6_weight, p_features_10_7_weight, p_features_10_7_bias,
    p_features_11_0_weight, p_features_11_1_weight, p_features_11_1_bias,
    p_features_11_3_weight, p_features_11_4_weight, p_features_11_4_bias,
    p_features_11_6_weight, p_features_11_7_weight, p_features_11_7_bias,
    p_features_12_0_weight, p_features_12_1_weight, p_features_12_1_bias,
    p_features_12_3_weight, p_features_12_4_weight, p_features_12_4_bias,
    p_features_12_6_weight, p_features_12_7_weight, p_features_12_7_bias,
    p_features_13_0_weight, p_features_13_1_weight, p_features_13_1_bias,
    p_features_13_3_weight, p_features_13_4_weight, p_features_13_4_bias,
    p_features_13_6_weight, p_features_13_7_weight, p_features_13_7_bias,
    p_features_14_0_weight, p_features_14_1_weight, p_features_14_1_bias,
    p_features_14_3_weight, p_features_14_4_weight, p_features_14_4_bias,
    p_features_14_6_weight, p_features_14_7_weight, p_features_14_7_bias,
    p_features_15_0_weight, p_features_15_1_weight, p_features_15_1_bias,
    p_features_15_3_weight, p_features_15_4_weight, p_features_15_4_bias,
    p_features_15_6_weight, p_features_15_7_weight, p_features_15_7_bias,
    p_features_16_0_weight, p_features_16_1_weight, p_features_16_1_bias,
    p_features_16_3_weight, p_features_16_4_weight, p_features_16_4_bias,
    p_features_16_6_weight, p_features_16_7_weight, p_features_16_7_bias,
    p_features_17_0_weight, p_features_17_1_weight, p_features_17_1_bias,
    p_features_17_3_weight, p_features_17_4_weight, p_features_17_4_bias,
    p_features_17_6_weight, p_features_17_7_weight, p_features_17_7_bias,
    p_features_18_0_weight, p_features_18_1_weight, p_features_18_1_bias,
    p_features_18_3_weight, p_features_18_4_weight, p_features_18_4_bias,
    p_features_18_6_weight, p_features_18_7_weight, p_features_18_7_bias,
    p_features_19_0_weight, p_features_19_1_weight, p_features_19_1_bias,
    p_features_19_3_weight, p_features_19_4_weight, p_features_19_4_bias,
    p_features_19_6_weight, p_features_19_7_weight, p_features_19_7_bias,
    p_features_20_weight, p_features_21_weight, p_features_21_bias,
    p_classifier_1_weight, p_classifier_1_bias,
    b_features_1_running_mean, b_features_1_running_var,
    b_features_3_1_running_mean, b_features_3_1_running_var,
    b_features_3_4_running_mean, b_features_3_4_running_var,