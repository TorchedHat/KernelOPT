import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm + HardTanh (ReLU6) kernel
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
def bn_hardtanh_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index: layout is (N, C, H, W) -> channel = (offset // HW) % C
    hw_idx = offsets % HW
    c_idx = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b
    
    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    
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
def hardtanh_inplace_kernel(
    x_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    x = tl.minimum(tl.maximum(x, 0.0), 6.0)
    tl.store(x_ptr + offsets, x, mask=mask)


def fused_bn_hardtanh(x, running_mean, running_var, weight, bias, eps=1e-5, apply_relu6=True):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    if apply_relu6:
        bn_hardtanh_kernel[grid](
            x, out, running_mean, running_var, weight, bias,
        )
    else:
        bn_hardtanh_kernel[grid](
            x, out, running_mean, running_var, weight, bias,
        )
    return out


def conv2d_op(x, weight, bias=None, stride=1, padding=0, groups=1):
    """Use aten convolution - implementing via low-level call"""
    return torch.ops.aten.convolution(x, weight, bias, 
                                       [stride, stride] if isinstance(stride, int) else stride,
                                       [padding, padding] if isinstance(padding, int) else padding,


def adaptive_avg_pool2d_op(x, output_size):
    return torch.ops.aten.adaptive_avg_pool2d(x, output_size)


def linear_op(x, weight, bias):
    return torch.ops.aten.linear(x, weight, bias)


def kernel_function(
    p_features_0_weight,
    p_features_1_weight, p_features_1_bias,
    p_features_3_0_weight,
    p_features_3_1_weight, p_features_3_1_bias,
    p_features_3_3_weight,
    p_features_3_4_weight, p_features_3_4_bias,
    p_features_4_0_weight,
    p_features_4_1_weight, p_features_4_1_bias,
    p_features_4_3_weight,
    p_features_4_4_weight, p_features_4_4_bias,
    p_features_4_6_weight,
    p_features_4_7_weight, p_features_4_7_bias,
    p_features_5_0_weight,
    p_features_5_1_weight, p_features_5_1_bias,
    p_features_5_3_weight,
    p_features_5_4_weight, p_features_5_4_bias,
    p_features_5_6_weight,
    p_features_5_7_weight, p_features_5_7_bias,
    p_features_6_0_weight,
    p_features_6_1_weight, p_features_6_1_bias,
    p_features_6_3_weight,
    p_features_6_4_weight, p_features_6_4_bias,
    p_features_6_6_weight,
    p_features_6_7_weight, p_features_6_7_bias,
    p_features_7_0_weight,
    p_features_7_1_weight, p_features_7_1_bias,
    p_features_7_3_weight,
    p_features_7_4_weight, p_features_7_4_bias,
    p_features_7_6_weight,
    p_features_7_7_weight, p_features_7_7_bias,
    p_features_8_0_weight,
    p_features_8_1_weight, p_features_8_1_bias,
    p_features_8_3_weight,
    p_features_8_4_weight, p_features_8_4_bias,
    p_features_8_6_weight,
    p_features_8_7_weight, p_features_8_7_bias,
    p_features_9_0_weight,
    p_features_9_1_weight, p_features_9_1_bias,
    p_features_9_3_weight,
    p_features_9_4_weight, p_features_9_4_bias,
    p_features_9_6_weight,
    p_features_9_7_weight, p_features_9_7_bias,
    p_features_10_0_weight,
    p_features_10_1_weight, p_features_10_1_bias,
    p_features_10_3_weight,
    p_features_10_4_weight, p_features_10_4_bias,
    p_features_10_6_weight,
    p_features_10_7_weight, p_features_10_7_bias,
    p_features_11_0_weight,
    p_features_11_1_weight, p_features_11_1_bias,
    p_features_11_3_weight,
    p_features_11_4_weight, p_features_11_4_bias,
    p_features_11_6_weight,
    p_features_11_7_weight, p_features_11_7_bias,
    p_features_12_0_weight,
    p_features_12_1_weight, p_features_12_1_bias,
    p_features_12_3_weight,
    p_features_12_4_weight, p_features_12_4_bias,
    p_features_12_6_weight,
    p_features_12_7_weight, p_features_12_7_bias,
    p_features_13_0_weight,
    p_features_13_1_weight, p_features_13_1_bias,
    p_features_13_3_weight,
    p_features_13_4_weight, p_features_13_4_bias,
    p_features_13_6_weight,
    p_features_13_7_weight, p_features_13_7_bias,
    p_features_14_0_weight,
    p_features_14_1_weight, p_features_14_1_bias,
    p_features_14_3_weight,
    p_features_14_4_weight, p_features_14_4_bias,
    p_features_14_6_weight,
    p_features_14_7_weight, p_features_14_7_bias,
    p_features_15_0_weight,
    p_features_15_1_weight, p_features_15_1_bias,
    p_features_15_3_weight,
    p_features_15_4_weight, p_features_15_4_bias,
    p_features_15_6_weight,
    p_features_15_7_weight, p_features_15_7_bias,
    p_features_16_0_weight,
    p_features_16_1_weight, p_features_16_1_bias,
    p_features_16_3_weight,
    p_features_16_4_weight, p_features_16_4_bias,
    p_features_16_6_weight,
    p_features_16_7_weight, p_features_16_7_bias,
    p_features_17_0_weight,
    p_features_17_1_weight, p_features_17_1_bias,
    p_features_17_3_weight,
    p_features_17_4_weight, p_features_17_4_bias,
    p_features_17_6_weight,
    p_features_17_7_weight, p_features_17_7_bias,
    p_features_18_0_weight,
    p_features_18_1_weight, p_features_18_1_bias,
    p_features_18_3_weight,
    p_features_18_4_weight, p_features_18_4_bias,
    p_features_18_6_weight,
    p_features_18_7_weight, p_features_18_7_bias,
    p_features_19_0_weight,
    p_features_19_1_weight, p_features_19_1_bias,
    p_features_19_3_weight,
    p_features_19_4_weight, p_features_19_4_bias,
    p_features_19_6_weight,
    p_features_19_7_weight, p_features_19_7_bias,
    p_features_20_weight,
    p_features_21_weight, p_features_21_bias,
    p_classifier_1_weight, p_classifier_1_bias,
    b_features_1_running_mean, b_features_1_running_var,
    b_features_3_1_running_mean, b_features_3_1_running_var,
    b_features_3_4_running_mean, b_features_3_4_running_var,
    b_features_4_1_running_mean, b_features_4_1_running_var,
    b_features_4_4_running_mean, b_features_4_4_running_var,
    b_features_4_7_running_mean, b_features_4_7_running_var,
    b_features_5_1_running_mean, b_features_5_1_running_var,
    b_features_5_4_running_mean, b_features_5_4_running_var,
    b_features_5_7_running_mean, b_features_5_7_running_var,
    b_features_6_1_running_mean, b_features_6_1_running_var,
    b_features_6_4_running_mean, b_features_6_4_running_var,
    b_features_6_7_running_mean, b_features_6_7_running_var,
    b_features_7_1_running_mean, b_features_7_1_running_var,
    b_features_7_4_running_mean, b_features_7_4_running_var,
    b_features_7_7_running_mean, b_features_7_7_running_var,
    b_features_8_1_running_mean, b_features_8_1_running_var,
    b_features_8_4_running_mean, b_features_8_4_running_var,
    b_features_8_7_running_mean, b_features_8_7_running_var,
    b_features_9_1_running_mean, b_features_9_1_running_var,
    b_features_9_4_running_mean, b_features_9_4_running_var,
    b_features_9_7_running_mean, b_features_9_7_running_var,
    b_features_10_1_running_mean, b_features_10_1_running_var,
    b_features_10_4_running_mean, b_features_10_4_running_var,
    b_features_10_7_running_mean, b_features_10_7_running_var,
    b_features_11_1_running_mean, b_features_11_1_running_var,
    b_features_11_4_running_mean, b_features_11_4_running_var,
    b_features_11_7_running_mean, b_features_11_7_running_var,
    b_features_12_1_running_mean, b_features_12_1_running_var,
    b_features_12_4_running_mean, b_features_12_4_running_var,
    b_features_12_7_running_mean, b_features_12_7_running_var,
    b_features_13_1_running_mean, b_features_13_1_running_var,
    b_features_13_4_running_mean, b_features_13_4_running_var,
    b_features_13_7_running_mean, b_features_13_7_running_var,
    b_features_14_1_running_mean, b_features_14_1_running_var,
    b_features_14_4_running_mean, b_features_14_4_running_var,
    b_features_14_7_running_mean, b_features_14_7_running_var,
    b_features_15_1_running_mean, b_features_15_1_running_var,
    b_features_15_4_running_mean, b_features_15_4_running_var,
    b_features_15_7_running_mean, b_features_15_7_running_var,
    b_features_16_1_running_mean, b_features_16_1_running_var,
    b_features_16_4_running_mean, b_features_16_4_running_var,
    b_features_16_7_running_mean, b_features_16_7_running_var,
    b_features_17_1_running_mean, b_features_17_1_running_var,
    b_features_17_4_running_mean, b_features_17_4_running_var,
    b_features_17_7_running_mean, b_features_17_7_running_var,
    b_features_18_1_running_mean, b_features_18_1_running_var,
    b_features_18_4_running_mean, b_features_18_4_running_var,
    b_features_18_7_running_mean, b_features_18_7_running_var,
    b_features_19_1_running_mean, b_features_19_1_running_var,
    b_features_19_4_running_mean, b_features_19_4_running_var,
    b_features_19_7_running_mean, b_features_19_7_running_var,
    b_features_21_running_mean, b_features_21_running_var,
    x,
):
    eps = 1e-5
    
    def conv(inp, w, stride=1, padding=0, groups=1):
        return torch.ops.aten.convolution(inp, w, None,
                                          [stride, stride] if isinstance(stride, int) else list(stride),
                                          [padding, padding] if isinstance(padding, int) else list(padding),
    
    def bn_relu6(inp, rm, rv, w, b):
        return fused_bn_hardtanh(inp, rm, rv, w, b, eps=eps, apply_relu6=True)
    
    def bn(inp, rm, rv, w, b):
        return fused_bn_hardtanh(inp, rm, rv, w, b, eps=eps, apply_relu6=False)
    
    # features[0]: Conv2d(3, 32, 3, stride=2, padding=1)
    # features[1]: BN(32) + ReLU6
    x = conv(x, p_features_0_weight, stride=2, padding=1)
    x = bn_relu6(x, b_features_1_running_mean, b_features_1_running_var, p_features_1_weight, p_features_1_bias)
    
    # features[3]: InvertedResidual t=1, c=16, s=1 (no expand since t=1)
    # [0]: depthwise conv 32->32, stride=1, groups=32
    # [1]: BN + ReLU6
    # [3]: pointwise 32->16
    # [4]: BN (no relu)
    x = conv(x, p_features_3_0_weight, stride=1, padding=1, groups=32)
    x = bn_relu6(x, b_features_3_1_running_mean, b_features_3_1_running_var, p_features_3_1_weight, p_features_3_1_bias)
    x = conv(x, p_features_3_3_weight, stride=1, padding=0)
    x = bn(x, b_features_3_4_running_mean, b_features_3_4_running_var, p_features_3_4_weight, p_features_3_4_bias)
    
    # features[4]: InvertedResidual t=6, inp=16, oup=24, stride=2
    # expand: 16->96, depthwise 96->96 stride=2, project 96->24
    x = conv(x, p_features_4_0_weight, stride=1, padding=0)
    x = bn_relu6(x, b_features_4_1_running_mean, b_features_4_1_running_var, p_features_4_1_weight, p_features_4_1_bias)
    x = conv(x, p_features_4_3_weight, stride=2, padding=1, groups=96)
    x = bn_relu6(x, b_features_4_4_running_mean, b_features_4_4_running_var, p_features_4_4_weight, p_features_4_4_bias)
    x = conv(x, p_features_4_6_weight, stride=1, padding=0)
    x = bn(x, b_features_4_7_running_mean, b_features_4_7_running_var, p_features_4_7_weight, p_features_4_7_bias)
    
    # features[5]: InvertedResidual t=6, inp=24, oup=24, stride=1 (residual)
    residual = x
    x = conv(x, p_features_5_0_weight, stride=1, padding=0)
    x = bn_relu6(x, b_features_5_1_running_mean, b_features_5_1_running_var, p_features_5_1_weight, p_features_5_1_bias)
    x = conv(x, p_features_5_3_weight, stride=1, padding=1, groups=144)
    x = bn_relu6(x, b_features_5_4_running_mean, b_features_5_4_running_var, p_features_5_4_weight, p_features_5_4_bias)
    x = conv(x, p_features_5_6_weight, stride=1, padding=0)
    x = bn(x, b_features_5_7_running_mean, b_features_5_7_running_var, p_features_5_7_weight, p_features_5_7_bias)
    x = x + residual
    
    # features[6]: InvertedResidual t=6, inp=24, oup=32, stride=2
    x = conv(x, p_features_6_0_weight, stride=1, padding=0)
    x = bn_relu6(x, b_features_6_1_running_mean, b_features_6_1_running_var, p_features_6_1_weight, p_features_6_1_bias)
    x = conv(x, p_features_6_3_weight, stride=2, padding=1, groups=144)
    x = bn_relu6(x, b_features_6_4_running_mean, b_features_6_4_running_var, p_features_6_4_weight, p_features_6_4_bias)
    x = conv(x, p_features_6_6_weight, stride=1, padding=0)
    x = bn(x, b_features_6_7_running_mean, b_features_6_7_running_var, p_features_6_7_weight, p_features_6_7_bias)
    
    # features[7]: InvertedResidual t=6, inp=32, oup=32, stride=1 (residual)
    residual = x
    x = conv(x, p_features_7_0_weight, stride=1, padding=0)
    x = bn_relu6(x, b_features_7_1_running_mean, b_features_7_1_running_var, p_features_7_1_weight, p_features_7_1_bias)
    x = conv(x, p_features_7_3_weight, stride=1, padding=1, groups=192)
    x = bn_relu6(x, b_features_7_4_running_mean, b_features_7_4_running_var, p_features_7_4_weight, p_features_7_4_bias)
    x = conv(x, p_features_7_6_weight, stride=1, padding=0)
    x = bn(x, b_features_7_7_running_mean, b_features_7_7_running_var, p_features_7_7_weight, p_features_7_7_bias)
    x = x + residual
    
    # features[8]: InvertedResidual t=6, inp=32, oup=32, stride=1 (residual)
    residual = x
    x = conv(x, p_features_8_0_weight, stride=1, padding=0)
    x = bn_relu6(x, b_features_8_1_running_mean, b_features_8_1_running_var, p_features_8_1_weight, p_features_8_1_bias)
    x = conv(x, p_features_8_3_weight, stride=1, padding=1, groups=192)
    x = bn_relu6(x, b_features_8_4_running_mean, b_features_8_4_running_var, p_features_8_4_weight, p_features_8_4_bias)
    x = conv(x, p_features_8_6_weight, stride=1, padding=0)
    x = bn(x, b_features_8_7_running_mean, b_features_8_7_running_var, p_features_8_7_weight, p_features_8_7_bias)
    x = x + residual
    
    # features[9]: InvertedResidual t=6, inp=32, oup=64, stride=2
    x = conv(x, p_features_9_0_weight, stride=1, padding=0)
    x = bn_relu6(x, b_features_9_1_running_mean, b_features_9_1_running_var, p_features_9_1_weight, p_features_9_1_bias)
    x = conv(x, p_features_9_3_weight, stride=2, padding=1, groups=192)
    x = bn_relu6(x, b_features_9_4_running_mean, b_features_9_4_running_var, p_features_9_4_weight, p_features_9_4_bias)
    x = conv(x, p_features_9_6_weight, stride=1, padding=0)
    x = bn(x, b_features_9_7_running_mean, b_features_9_7_running_var, p_features_9_7_weight, p_features_9_7_bias)
    
    # features[10]: InvertedResidual t=6, inp=64, oup=64, stride=1 (residual)
    residual = x
    x = conv(x, p_features_10_0_weight, stride=1, padding=0)
    x = bn_relu6(x, b_features_10_1_running_mean, b_features_10_1_running_var, p_features_10_1_weight, p_features_10_1_bias)
    x = conv(x, p_features_10_3_weight, stride=1, padding=1, groups=384)
    x = bn_relu6(x, b_features_10_4_running_mean, b_features_10_4_running_var, p_features_10_4_weight, p_features_10_4_bias)
    x = conv(x, p_features_10_6_weight, stride=1, padding=0)
    x = bn(x, b_features_10_7_running_mean, b_features_10_7_running_var, p_features_10_7_weight, p_features_10_7_bias)
    x = x + residual
    
    # features[11]: InvertedResidual t=6, inp=64, oup=64, stride=1 (residual)
    residual = x
    x = conv(x, p_features_11_0_weight, stride=1, padding=0)
    x = bn_relu6(x, b_features_11_1_running_mean, b_features_11_1_running_var, p_features_11_1_weight, p_features_11_1_bias)
    x = conv(x, p_features_11_3_weight, stride=1, padding=1, groups=384)
    x = bn_relu6(x, b_features_11_4_running_mean, b_features_11_4_running_var, p_features_11_4_weight, p_features_11_4_bias)
    x = conv(x, p_features_11_6_weight, stride=1, padding=0)
    x = bn(x, b_features_11_7_running_mean, b_features_11_7_running_var, p_features_11_7_weight, p_features_11_7_bias)
    x = x + residual
    
    # features[12]: InvertedResidual t=6, inp=64, oup=64, stride=1 (residual)
    residual = x
    x = conv(x, p_features_12_0_weight, stride=1, padding=0)
    x = bn_