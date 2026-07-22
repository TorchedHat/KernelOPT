import torch
import triton
import triton.language as tl

# Fused BatchNorm + HardTanh (ReLU6) kernel
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
def fused_bn_relu6_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps: tl.constexpr,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    # Layout: (N, C, H, W) -> channel = (offset // spatial) % num_channels
    chan = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0).to(tl.float32)
    
    # BatchNorm
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    # HardTanh (ReLU6: clamp to [0, 6])
    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# Fused BatchNorm only (no activation)
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
def fused_bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    chan = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def apply_bn_relu6(x, weight, bias, running_mean, running_var, eps=1e-5):
    """Apply BatchNorm + ReLU6 using fused Triton kernel."""
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu6_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial,
        eps=eps,
        apply_relu6=True,
    )
    return out


def apply_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    """Apply BatchNorm using fused Triton kernel."""
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial,
        eps=eps,
    )
    return out


def conv2d(x, weight, stride=1, padding=0, groups=1):
    """Standard conv2d using PyTorch (cuDNN)."""
    return torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


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
    # features[0]: Conv2d(3, 32, 3, stride=2, padding=1)
    x = conv2d(x, p_features_0_weight, stride=2, padding=1)
    # features[1]: BN + ReLU6
    x = apply_bn_relu6(x, p_features_1_weight, p_features_1_bias,

    # features[3]: InvertedResidual t=1, c=16, s=1 (no expand)
    # Depthwise conv
    x = conv2d(x, p_features_3_0_weight, stride=1, padding=1, groups=32)
    x = apply_bn_relu6(x, p_features_3_1_weight, p_features_3_1_bias,
    # Pointwise linear
    x = conv2d(x, p_features_3_3_weight, stride=1, padding=0)
    x = apply_bn(x, p_features_3_4_weight, p_features_3_4_bias,

    # features[4]: InvertedResidual t=6, c=24, s=2
    # Expand
    x = conv2d(x, p_features_4_0_weight, stride=1, padding=0)
    x = apply_bn_relu6(x, p_features_4_1_weight, p_features_4_1_bias,
    # Depthwise
    x = conv2d(x, p_features_4_3_weight, stride=2, padding=1, groups=96)
    x = apply_bn_relu6(x, p_features_4_4_weight, p_features_4_4_bias,
    # Project
    x = conv2d(x, p_features_4_6_weight, stride=1, padding=0)
    x = apply_bn(x, p_features_4_7_weight, p_features_4_7_bias,

    # features[5]: InvertedResidual t=6, c=24, s=1 (residual)
    residual = x
    x = conv2d(x, p_features_5_0_weight, stride=1, padding=0)
    x = apply_bn_relu6(x, p_features_5_1_weight, p_features_5_1_bias,
    x = conv2d(x, p_features_5_3_weight, stride=1, padding=1, groups=144)
    x = apply_bn_relu6(x, p_features_5_4_weight, p_features_5_4_bias,
    x = conv2d(x, p_features_5_6_weight, stride=1, padding=0)
    x = apply_bn(x, p_features_5_7_weight, p_features_5_7_bias,
    x = x + residual

    # features[6]: InvertedResidual t=6, c=32, s=2
    x = conv2d(x, p_features_6_0_weight, stride=1, padding=0)
    x = apply_bn_relu6(x, p_features_6_1_weight, p_features_6_1_bias,
    x = conv2d(x, p_features_6_3_weight, stride=2, padding=1, groups=144)
    x = apply_bn_relu6(x, p_features_6_4_weight, p_features_6_4_bias,
    x = conv2d(x, p_features_6_6_weight, stride=1, padding=0)
    x = apply_bn(x, p_features_6_7_weight, p_features_6_7_bias,

    # features[7]: InvertedResidual t=6, c=32, s=1 (residual)
    residual = x
    x = conv2d(x, p_features_7_0_weight, stride=1, padding=0)
    x = apply_bn_relu6(x, p_features_7_1_weight, p_features_7_1_bias,
    x = conv2d(x, p_features_7_3_weight, stride=1, padding=1, groups=192)
    x = apply_bn_relu6(x, p_features_7_4_weight, p_features_7_4_bias,
    x = conv2d(x, p_features_7_6_weight, stride=1, padding=0)
    x = apply_bn(x, p_features_7_7_weight, p_features_7_7_bias,
    x = x + residual

    # features[8]: InvertedResidual t=6, c=32, s=1 (residual)
    residual = x
    x = conv2d(x, p_features_8_0_weight, stride=1, padding=0)
    x = apply_bn_relu6(x, p_features_8_1_weight, p_features_8_1_bias,
    x = conv2d(x, p_features_8_3_weight, stride=1, padding=1, groups=192)
    x = apply_bn_relu6(x, p_features_8_4_weight, p_features_8_4_bias,
    x = conv2d(x, p_features_8_6_weight, stride=1, padding=0)
    x = apply_bn(x, p_features_8_7_weight, p_features_8_7_bias,
    x = x + residual

    # features[9]: InvertedResidual t=6, c=64, s=2
    x = conv2d(x, p_features_9_0_weight, stride=1, padding=0)
    x = apply_bn_relu6(x, p_features_9_1_weight, p_features_9_1_bias,
    x = conv2d(x, p_features_9_3_weight, stride=2, padding=1, groups=192)
    x = apply_bn_relu6(x, p_features_9_4_weight, p_features_9_4_bias,
    x = conv2d(x, p_features_9_6_weight, stride=1, padding=0)
    x = apply_bn(x, p_features_9_7_weight, p_features_9_7_bias,

    # features[10]: InvertedResidual t=6, c=64, s=1 (residual)
    residual = x
    x = conv2d(x, p_features_10_0_weight, stride=1, padding=0)
    x = apply_bn_relu6(x, p_features_10_1_weight, p_features_10_1_bias,
    x = conv2d(x, p_features_10_3_weight, stride=1, padding=1, groups=384)
    x = apply_bn_relu6(x, p_features_10_4_weight, p_features_10_4_bias,
    x = conv2d(x, p_features_10_6_weight, stride=1, padding=0)
    x = apply_bn(x, p_features_10_7_weight, p_features_10_7_bias,
    x = x + residual

    # features[11]: InvertedResidual t=6, c=64, s=1 (residual)
    residual = x
    x = conv2d(x, p_features_11_0_weight, stride=1, padding=0)
    x = apply_bn_relu6(x, p_features_11_1_weight, p_features_11_1_bias,
    x = conv2d(x, p_features_11_3_weight, stride=1, padding=1, groups=384)
    x = apply_bn_relu6(x, p_features_11_4_weight, p_features_11_4_bias,
    x = conv2d(x, p_features_11_6_weight, stride=1, padding=0)
    x = apply_bn(x, p_features_11_7_weight, p_features_11_7_bias,
    x = x + residual

    # features[12]: InvertedResidual t=6, c=64, s=1 (residual)
    residual = x
    x = conv2d(x, p_features_12_0_weight, stride=1, padding=0)
    x = apply_bn_relu6(x, p_features_12_1_weight, p_features_12_1_bias,
    x = conv2d(x, p_features_12_3_weight, stride=1, padding=1, groups=384)
    x = apply_bn_relu6(x, p_features_12_4_weight, p_features_12_4_bias,
    x = conv2d(x, p_features_12_6_weight, stride=1, padding=0)
    x = apply_bn(x, p_features_12_7_weight, p_features_12_7_bias,
    x = x + residual

    # features[13]: InvertedResidual t=6, c=96, s=1
    x = conv2d(x, p_features_13_0_weight, stride=1, padding=0)
    x = apply_bn_relu6(x, p_features_13_1_weight, p_features_