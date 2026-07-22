import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_relu6_kernel(
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
    
    # Batch norm
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_scaled = x_norm * w + b
    
    # ReLU6 (hardtanh with min=0, max=6)
    x_out = tl.minimum(tl.maximum(x_scaled, 0.0), 6.0)
    
    tl.store(out_ptr + offsets, x_out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
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
    
    # Compute channel index
    c = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # Batch norm
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_scaled = x_norm * w + b
    
    tl.store(out_ptr + offsets, x_scaled, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
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
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def apply_bn_relu6(x, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    
    # Make x contiguous in channel-last or standard format
    # x is NCHW, so channel index = (flat_idx // spatial) % C
    x_cont = x.contiguous()
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu6_kernel[grid](
        x_cont, out,
        mean, var, weight, bias,
        numel, num_channels, spatial,
        eps,
    )
    return out


def apply_bn(x, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    
    x_cont = x.contiguous()
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x_cont, out,
        mean, var, weight, bias,
        numel, num_channels, spatial,
        eps,
    )
    return out


def conv2d_bn_relu6(x, conv_w, bn_mean, bn_var, bn_w, bn_b,
                    stride=1, padding=0, groups=1, eps=1e-5):
    # Use cuDNN for conv
    out = F.conv2d(x, conv_w, bias=None, stride=stride, padding=padding, groups=groups)
    # Fused BN + ReLU6
    out = apply_bn_relu6(out, bn_mean, bn_var, bn_w, bn_b, eps)
    return out


def conv2d_bn(x, conv_w, bn_mean, bn_var, bn_w, bn_b,
              stride=1, padding=0, groups=1, eps=1e-5):
    # Use cuDNN for conv
    out = F.conv2d(x, conv_w, bias=None, stride=stride, padding=padding, groups=groups)
    # Fused BN (no activation)
    out = apply_bn(out, bn_mean, bn_var, bn_w, bn_b, eps)
    return out


def kernel_function(
    p_features_0_weight,
    p_features_1_weight,
    p_features_1_bias,
    p_features_3_0_weight,
    p_features_3_1_weight,
    p_features_3_1_bias,
    p_features_3_3_weight,
    p_features_3_4_weight,
    p_features_3_4_bias,
    p_features_4_0_weight,
    p_features_4_1_weight,
    p_features_4_1_bias,
    p_features_4_3_weight,
    p_features_4_4_weight,
    p_features_4_4_bias,
    p_features_4_6_weight,
    p_features_4_7_weight,
    p_features_4_7_bias,
    p_features_5_0_weight,
    p_features_5_1_weight,
    p_features_5_1_bias,
    p_features_5_3_weight,
    p_features_5_4_weight,
    p_features_5_4_bias,
    p_features_5_6_weight,
    p_features_5_7_weight,
    p_features_5_7_bias,
    p_features_6_0_weight,
    p_features_6_1_weight,
    p_features_6_1_bias,
    p_features_6_3_weight,
    p_features_6_4_weight,
    p_features_6_4_bias,
    p_features_6_6_weight,
    p_features_6_7_weight,
    p_features_6_7_bias,
    p_features_7_0_weight,
    p_features_7_1_weight,
    p_features_7_1_bias,
    p_features_7_3_weight,
    p_features_7_4_weight,
    p_features_7_4_bias,
    p_features_7_6_weight,
    p_features_7_7_weight,
    p_features_7_7_bias,
    p_features_8_0_weight,
    p_features_8_1_weight,
    p_features_8_1_bias,
    p_features_8_3_weight,
    p_features_8_4_weight,
    p_features_8_4_bias,
    p_features_8_6_weight,
    p_features_8_7_weight,
    p_features_8_7_bias,
    p_features_9_0_weight,
    p_features_9_1_weight,
    p_features_9_1_bias,
    p_features_9_3_weight,
    p_features_9_4_weight,
    p_features_9_4_bias,
    p_features_9_6_weight,
    p_features_9_7_weight,
    p_features_9_7_bias,
    p_features_10_0_weight,
    p_features_10_1_weight,
    p_features_10_1_bias,
    p_features_10_3_weight,
    p_features_10_4_weight,
    p_features_10_4_bias,
    p_features_10_6_weight,
    p_features_10_7_weight,
    p_features_10_7_bias,
    p_features_11_0_weight,
    p_features_11_1_weight,
    p_features_11_1_bias,
    p_features_11_3_weight,
    p_features_11_4_weight,
    p_features_11_4_bias,
    p_features_11_6_weight,
    p_features_11_7_weight,
    p_features_11_7_bias,
    p_features_12_0_weight,
    p_features_12_1_weight,
    p_features_12_1_bias,
    p_features_12_3_weight,
    p_features_12_4_weight,
    p_features_12_4_bias,
    p_features_12_6_weight,
    p_features_12_7_weight,
    p_features_12_7_bias,
    p_features_13_0_weight,
    p_features_13_1_weight,
    p_features_13_1_bias,
    p_features_13_3_weight,
    p_features_13_4_weight,
    p_features_13_4_bias,
    p_features_13_6_weight,
    p_features_13_7_weight,
    p_features_13_7_bias,
    p_features_14_0_weight,
    p_features_14_1_weight,
    p_features_14_1_bias,
    p_features_14_3_weight,
    p_features_14_4_weight,
    p_features_14_4_bias,
    p_features_14_6_weight,
    p_features_14_7_weight,
    p_features_14_7_bias,
    p_features_15_0_weight,
    p_features_15_1_weight,
    p_features_15_1_bias,
    p_features_15_3_weight,
    p_features_15_4_weight,
    p_features_15_4_bias,
    p_features_15_6_weight,
    p_features_15_7_weight,
    p_features_15_7_bias,
    p_features_16_0_weight,
    p_features_16_1_weight,
    p_features_16_1_bias,
    p_features_16_3_weight,
    p_features_16_4_weight,
    p_features_16_4_bias,
    p_features_16_6_weight,
    p_features_16_7_weight,
    p_features_16_7_bias,
    p_features_17_0_weight,
    p_features_17_1_weight,
    p_features_17_1_bias,
    p_features_17_3_weight,
    p_features_17_4_weight,
    p_features_17_4_bias,
    p_features_17_6_weight,
    p_features_17_7_weight,
    p_features_17_7_bias,
    p_features_18_0_weight,
    p_features_18_1_weight,
    p_features_18_1_bias,
    p_features_18_3_weight,
    p_features_18_4_weight,
    p_features_18_4_bias,
    p_features_18_6_weight,
    p_features_18_7_weight,
    p_features_18_7_bias,
    p_features_19_0_weight,
    p_features_19_1_weight,
    p_features_19_1_bias,
    p_features_19_3_weight,
    p_features_19_4_weight,
    p_features_19_4_bias,
    p_features_19_6_weight,
    p_features_19_7_weight,
    p_features_19_7_bias,
    p_features_20_weight,
    p_features_21_weight,
    p_features_21_bias,
    p_classifier_1_weight,
    p_classifier_1_bias,
    b_features_1_running_mean,
    b_features_1_running_var,
    b_features_3_1_running_mean,
    b_features_3_1_running_var,
    b_features_3_4_running_mean,
    b_features_3_4_running_var,
    b_features_4_1_running_mean,
    b_features_4_1_running_var,
    b_features_4_4_running_mean,
    b_features_4_4_running_var,
    b_features_4_7_running_mean,
    b_features_4_7_running_var,
    b_features_5_1_running_mean,
    b_features_5_1_running_var,
    b_features_5_4_running_mean,
    b_features_5_4_running_var,
    b_features_5_7_running_mean,
    b_features_5_7_running_var,
    b_features_6_1_running_mean,
    b_features_6_1_running_var,
    b_features_6_4_running_mean,
    b_features_6_4_running_var,
    b_features_6_7_running_mean,
    b_features_6_7_running_var,
    b_features_7_1_running_mean,
    b_features_7_1_running_var,
    b_features_7_4_running_mean,
    b_features_7_4_running_var,
    b_features_7_7_running_mean,
    b_features_7_7_running_var,
    b_features_8_1_running_mean,
    b_features_8_1_running_var,
    b_features_8_4_running_mean,
    b_features_8_4_running_var,
    b_features_8_7_running_mean,
    b_features_8_7_running_var,
    b_features_9_1_running_mean,
    b_features_9_1_running_var,
    b_features_9_4_running_mean,
    b_features_9_4_running_var,
    b_features_9_7_running_mean,
    b_features_9_7_running_var,
    b_features_10_1_running_mean,
    b_features_10_1_running_var,
    b_features_10_4_running_mean,
    b_features_10_4_running_var,
    b_features_10_7_running_mean,
    b_features_10_7_running_var,
    b_features_11_1_running_mean,
    b_features_11_1_running_var,
    b_features_11_4_running_mean,
    b_features_11_4_running_var,
    b_features_11_7_running_mean,
    b_features_11_7_running_var,
    b_features_12_1_running_mean,
    b_features_12_1_running_var,
    b_features_12_4_running_mean,
    b_features_12_4_running_var,
    b_features_12_7_running_mean,
    b_features_12_7_running_var,
    b_features_13_1_running_mean,
    b_features_13_1_running_var,
    b_features_13_4_running_mean,
    b_features_13_4_running_var,
    b_features_13_7_running_mean,
    b_features_13_7_running_var,
    b_features_14_1_running_mean,
    b_features_14_1_running_var,
    b_features_14_4_running_mean,
    b_features_14_4_running_var,
    b_features_14_7_running_mean,
    b_features_14_7_running_var,
    b_features_15_1_running_mean,
    b_features_15_1_running_var,
    b_features_15_4_running_mean,
    b_features_15_4_running_var,
    b_features_15_7_running_mean,
    b_features_15_7_running_var,
    b_features_16_1_running_mean,
    b_features_16_1_running_var,
    b_features_16_4_running_mean,
    b_features_16_4_running_var,
    b_features_16_7_running_mean,
    b_features_16_7_running_var,
    b_features_17_1_running_mean,
    b_features_17_1_running_var,
    b_features_17_4_running_mean,
    b_features_17_4_running_var,
    b_features_17_7_running_mean,
    b_features_17_7_running_var,
    b_features_18_1_running_mean,
    b_features_18_1_running_var,
    b_features_18_4_running_mean,
    b_features_18_4_running_var,
    b_features_18_7_running_mean,
    b_features_18_7_running_var,
    b_features_19_1_running_mean,
    b_features_19_1_running_var,
    b_features_19_4_running_mean,
    b_features_19_4_running_var,
    b_features_19_7_running_mean,
    b_features_19_7_running_var,
    b_features_21_running_mean,
    b_features_21_running_var,
    x,
):
    eps = 1e-5
    
    # features[0]: Conv2d(3, 32, 3, stride=2, padding=1) + BN + ReLU6
    x = conv2d_bn_relu6(x, p_features_0_weight,
                        b_features_1_running_mean, b_features_1_running_var,
                        p_features_1_weight, p_features_1_bias,
                        stride=2, padding=1)
    # shape: [10, 32, 112, 112]
    
    # features[3]: InvertedResidual t=1, c=16, s=1 (no expand)
    # Depthwise conv
    x = conv2d_bn_relu6(x, p_features_3_0_weight,
                        b_features_3_1_running_mean, b_features_3_1_running_var,
                        p_features_3_1_weight, p_features_3_1_bias,
                        stride=1, padding=1, groups=32)
    # Pointwise linear
    x = conv2d_bn(x, p_features_3_3_weight,
                  b_features_3_4_running_mean, b_features_3_4_running_var,
                  p_features_3_4_weight, p_features_3_4_bias,
                  stride=1, padding=0)
    # shape: [10, 16, 112, 112]
    
    # features[4]: InvertedResidual t=6, c=24, s=2
    # Expand
    x = conv2d_bn_relu6(x, p_features_4_0_weight,
                        b_features_4_1_running_mean, b_features_4_1_running_var,
                        p_features_4_1_weight, p_features_4_1_bias,
                        stride=1, padding=0)
    # Depthwise
    x = conv2d_bn_relu6(x, p_features_4_3_weight,
                        b_features_4_4_running_mean, b_features_4_4_running_var,
                        p_features_4_4_weight, p_features_4_4_bias,
                        stride=2, padding=1, groups=96)
    # Pointwise linear
    x = conv2d_bn(x, p_features_4_6_weight,
                  b_features_4_7_running_mean, b_features_4_7_running_var,
                  p_features_4_7_weight, p_features_4_7_bias,
                  stride=1, padding=0)
    # shape: [10, 24, 56, 56]
    
    # features[5]: InvertedResidual t=6, c=24, s=1 (with residual)
    residual = x
    # Expand
    x = conv2d_bn_relu6(x, p_features_5_0_weight,
                        b_features_5_1_running_mean, b_features_5_1_running_var,
                        p_features_5_1_weight, p_features_5_1_bias,
                        stride=1, padding=0)
    # Depthwise
    x = conv2d_bn_relu6(x, p_features_5_3_weight,
                        b_features_5_4_running_mean, b_features_5_4_running_var,
                        p_features_5_4_weight, p_features_5_4_bias,
                        stride=1, padding=1, groups=144)
    # Pointwise linear
    x = conv2d_bn(x, p_features_5_6_weight,
                  b_features_5_7_running_mean, b_features_5_7_running_var,
                  p_features_5_7_weight, p_features_5_7_bias,
                  stride=1, padding=0)
    x = x + residual
    # shape: [10, 24, 56, 56]
    
    # features[6]: InvertedResidual t=6, c=32, s=2
    # Expand
    x = conv2d_bn_relu6(x, p_features_6_0_weight,
                        b_features_6_1_running_mean, b_features_6_1_running_var,
                        p_features_6_1_weight, p_features_6_1_bias,
                        stride=1, padding=0)
    # Depthwise
    x = conv2d_bn_relu6(x, p_features_6_3_weight,
                        b_features_6_4_running_mean, b_features_6_4_running_var,
                        p_features_6_4_weight, p_features_6_4_bias,
                        stride=2, padding=1, groups=144)
    # Pointwise linear
    x = conv2d_bn(x, p_features_6_6_weight,
                  b_features_6_7_running_mean, b_features_6_7_running_var,
                  p_features_6_7_weight, p_features_6_7_bias,
                  stride=1, padding=0)
    # shape: [10, 32, 28, 28]
    
    # features[7]: InvertedResidual t=6, c=32, s=1 (with residual)
    residual = x
    # Expand
    x = conv2d_bn_relu6(x, p_features_7_0_weight,
                        b_features_7_1_running_mean, b_features_7_1_running_var,
                        p_features_7_1_weight, p_features_7_1_bias,
                        stride=1, padding=0)
    # Depthwise
    x = conv2d_bn_relu6(x, p_features_7_3_weight,
                        b_features_7_4_running_mean, b_features_7_4_running_var,
                        p_features_7_4_weight, p_features_7_4_bias,
                        stride=1, padding=1, groups=192)
    # Pointwise linear
    x = conv2d_bn(x, p_features_7_6_weight,
                  b_features_7_7_running_mean, b_features_7_7_running_var,
                  p_features_7_7_weight, p_features_7_7_bias,
                  stride=1, padding=0)
    x = x + residual
    # shape: [10, 32, 28, 28]
    
    # features[8]: InvertedResidual t=6, c=32, s=1 (with residual)
    residual = x
    # Expand
    x = conv2d_bn_relu6(x, p_features_8_0_weight,
                        b_features_8_1_running_mean, b_features_8_1_running_var,
                        p_features_8_1_weight, p_features_8_1_bias,
                        stride=1, padding=0)
    # Depthwise
    x = conv2d_bn_relu6(x, p_features_8_3_weight,
                        b_features_8_4_running_mean, b_features_8_4_running_var,
                        p_features_8_4_weight, p_features_8_4_bias,
                        stride=1, padding=1, groups=192)
    # Pointwise linear
    x = conv2d_bn(x, p_features_8_6_weight,
                  b_features_8_7_running_mean, b_features_8_7_running_var,
                  p_features_8_7_weight, p_features_8_7_bias,
                  stride=1, padding=0)
    x = x + residual
    # shape: [10, 32, 28, 28]
    
    # features[9]: InvertedResidual t=6, c=64, s=2
    # Expand
    x = conv2d_bn_