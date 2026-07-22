import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm + ReLU6 (HardTanh[0,6]) pointwise kernel
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
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    numel,
    num_channels,
    spatial,
    eps,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index: layout is (N, C, H, W) -> channel = (offset / spatial) % C
    # But we store as flat: offset -> channel index
    # For NCHW: index = n*C*H*W + c*H*W + h*W + w
    # channel = (offset // spatial) % num_channels
    chan = (offsets // spatial) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)

    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b

    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def apply_bn_relu6(x, mean, var, weight, bias, eps=1e-5, apply_relu6=True):
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu6_kernel[grid](
        x, out, mean, var, weight, bias,
        numel, C, spatial, eps, apply_relu6,
    )
    return out


def mobilenetv2_forward(
    # weights
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
    # running stats
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
    # input
    x,
):
    def conv2d(inp, w, stride=1, padding=0, groups=1):
        return torch.ops.aten.convolution(inp, w, None, [stride, stride], [padding, padding], [1, 1], False, [0, 0], groups)

    def bn_act(t, rm, rv, w, b, relu6=True):
        return apply_bn_relu6(t, rm, rv, w, b, eps=1e-5, apply_relu6=relu6)

    # features[0]: Conv2d(3, 32, 3, stride=2, padding=1)
    t = conv2d(x, p_features_0_weight, stride=2, padding=1)
    t = bn_act(t, b_features_1_running_mean, b_features_1_running_var,
               p_features_1_weight, p_features_1_bias, relu6=True)

    # features[3]: InvertedResidual t=1, inp=32, oup=16, stride=1
    # (no expand since t=1, just dw+pw)
    t = conv2d(t, p_features_3_0_weight, stride=1, padding=1, groups=32)
    t = bn_act(t, b_features_3_1_running_mean, b_features_3_1_running_var,
               p_features_3_1_weight, p_features_3_1_bias, relu6=True)
    t = conv2d(t, p_features_3_3_weight, stride=1, padding=0)
    t = bn_act(t, b_features_3_4_running_mean, b_features_3_4_running_var,
               p_features_3_4_weight, p_features_3_4_bias, relu6=False)

    # features[4]: InvertedResidual t=6, inp=16, oup=24, stride=2
    t = conv2d(t, p_features_4_0_weight, stride=1, padding=0)
    t = bn_act(t, b_features_4_1_running_mean, b_features_4_1_running_var,
               p_features_4_1_weight, p_features_4_1_bias, relu6=True)
    t = conv2d(t, p_features_4_3_weight, stride=2, padding=1, groups=96)
    t = bn_act(t, b_features_4_4_running_mean, b_features_4_4_running_var,
               p_features_4_4_weight, p_features_4_4_bias, relu6=True)
    t = conv2d(t, p_features_4_6_weight, stride=1, padding=0)
    t = bn_act(t, b_features_4_7_running_mean, b_features_4_7_running_var,
               p_features_4_7_weight, p_features_4_7_bias, relu6=False)

    # features[5]: InvertedResidual t=6, inp=24, oup=24, stride=1 (residual)
    residual = t
    t = conv2d(t, p_features_5_0_weight, stride=1, padding=0)
    t = bn_act(t, b_features_5_1_running_mean, b_features_5_1_running_var,
               p_features_5_1_weight, p_features_5_1_bias, relu6=True)
    t = conv2d(t, p_features_5_3_weight, stride=1, padding=1, groups=144)
    t = bn_act(t, b_features_5_4_running_mean, b_features_5_4_running_var,
               p_features_5_4_weight, p_features_5_4_bias, relu6=True)
    t = conv2d(t, p_features_5_6_weight, stride=1, padding=0)
    t = bn_act(t, b_features_5_7_running_mean, b_features_5_7_running_var,
               p_features_5_7_weight, p_features_5_7_bias, relu6=False)
    t = t + residual

    # features[6]: InvertedResidual t=6, inp=24, oup=32, stride=2
    t = conv2d(t, p_features_6_0_weight, stride=1, padding=0)
    t = bn_act(t, b_features_6_1_running_mean, b_features_6_1_running_var,
               p_features_6_1_weight, p_features_6_1_bias, relu6=True)
    t = conv2d(t, p_features_6_3_weight, stride=2, padding=1, groups=144)
    t = bn_act(t, b_features_6_4_running_mean, b_features_6_4_running_var,
               p_features_6_4_weight, p_features_6_4_bias, relu6=True)
    t = conv2d(t, p_features_6_6_weight, stride=1, padding=0)
    t = bn_act(t, b_features_6_7_running_mean, b_features_6_7_running_var,
               p_features_6_7_weight, p_features_6_7_bias, relu6=False)

    # features[7]: InvertedResidual t=6, inp=32, oup=32, stride=1 (residual)
    residual = t
    t = conv2d(t, p_features_7_0_weight, stride=1, padding=0)
    t = bn_act(t, b_features_7_1_running_mean, b_features_7_1_running_var,
               p_features_7_1_weight, p_features_7_1_bias, relu6=True)
    t = conv2d(t, p_features_7_3_weight, stride=1, padding=1, groups=192)
    t = bn_act(t, b_features_7_4_running_mean, b_features_7_4_running_var,
               p_features_7_4_weight, p_features_7_4_bias, relu6=True)
    t = conv2d(t, p_features_7_6_weight, stride=1, padding=0)
    t = bn_act(t, b_features_7_7_running_mean, b_features_7_7_running_var,
               p_features_7_7_weight, p_features_7_7_bias, relu6=False)
    t = t + residual

    # features[8]: InvertedResidual t=6, inp=32, oup=32, stride=1 (residual)
    residual = t
    t = conv2d(t, p_features_8_0_weight, stride=1, padding=0)
    t = bn_act(t, b_features_8_1_running_mean, b_features_8_1_running_var,
               p_features_8_1_weight, p_features_8_1_bias, relu6=True)
    t = conv2d(t, p_features_8_3_weight, stride=1, padding=1, groups=192)
    t = bn_act(t, b_features_8_4_running_mean, b_features_8_4_running_var,
               p_features_8_4_weight, p_features_8_4_bias, relu6=True)
    t = conv2d(t, p_features_8_6_weight, stride=1, padding=0)
    t = bn_act(t, b_features_8_7_running_mean, b_features_8_7_running_var,
               p_features_8_7_weight, p_features_8_7_bias, relu6=False)
    t = t + residual

    # features[9]: InvertedResidual t=6, inp=32, oup=64, stride=2
    t = conv2d(t, p_features_9_0_weight, stride=1, padding=0)
    t = bn_act(t, b_features_9_1_running_mean, b_features_9_1_running_var,
               p_features_9_1_weight, p_features_9_1_bias, relu6=True)
    t = conv2d(t, p_features_9_3_weight, stride=2, padding=1, groups=192)
    t = bn_act(t, b_features_9_4_running_mean, b_features_9_4_running_var,
               p_features_9_4_weight, p_features_9_4_bias, relu6=True)
    t = conv2d(t, p_features_9_6_weight, stride=1, padding=0)
    t = bn_act(t, b_features_9_7_running_mean, b_features_9_7_running_var,
               p_features_9_7_weight, p_features_9_7_bias, relu6=False)

    # features[10]: InvertedResidual t=6, inp=64, oup=64, stride=1 (residual)
    residual = t
    t = conv2d(t, p_features_10_0_weight, stride=1, padding=0)
    t = bn_act(t, b_features_10_1_running_mean, b_features_10_1_running_var,
               p_features_10_1_weight, p_features_10_1_bias, relu6=True)
    t = conv2d(t, p_features_10_3_weight, stride=1, padding=1, groups=384)
    t = bn_act(t, b_features_10_4_running_mean, b_features_10_4_running_var,
               p_features_10_4_weight, p_features_10_4_bias, relu6=True)
    t = conv2d(t, p_features_10_6_weight, stride=1, padding=0)
    t = bn_act(t, b_features_10_7_running_mean, b_features_10_7_running_var,
               p_features_10_7_weight, p_features_10_7_bias, relu6=False)
    t = t + residual

    # features[11]: InvertedResidual t=6, inp=64, oup=64, stride=1 (residual)
    residual = t
    t = conv2d(t, p_features_11_0_weight, stride=1, padding=0)
    t = bn_act(t, b_features_11_1_running_mean, b_features_11_1_running_var,
               p_features_11_1_weight, p_features_11_1_bias, relu6=True)
    t = conv2d(t, p_features_11_3_weight, stride=1, padding=1, groups=384)
    t = bn_act(t, b_features_11_4_running_mean, b_features_11_4_running_var,
               p_features_11_4_weight, p_features_11_4_bias, relu6=True)
    t = conv2d(t, p_features_11_6_weight, stride=1, padding=0)
    t = bn_act(t, b_features_11_7_running_mean, b_features_11_7_running_var,
               p_features_11_7_weight, p_features_11_7_bias, relu6=False)
    t = t + residual

    # features[12]: InvertedResidual t=6, inp=64, oup=64, stride=1 (residual)
    residual = t
    t = conv2d(t, p_features_12_0_weight, stride=1, padding=0)
    t = bn_act(t, b_features_12_1_running_mean, b_features_12_1_running_var,
               p_features_12_1_weight, p_features_12_1_bias, relu6=True)
    t = conv2d(t, p_features_12_3_weight, stride=1, padding=1, groups=384)
    t = bn_act(t, b_features_12_4_running_mean, b_features_12_4_running_var,
               p_features_12_4_weight, p_features_12_4_bias, relu6=True)
    t = conv2d(t, p_features_12_6_weight, stride=1, padding=0)
    t = bn_act(t, b_features_12_7_running_mean, b_features_12_7_running_var,
               p_features_12_7_weight, p_features_12_7_bias, relu6=False)
    t = t + residual

    # features[13]: InvertedResidual t=6, inp=64, oup=96, stride=1
    t = conv2d(t, p_features_13_0_weight, stride=1, padding=0)
    t = bn_act(t, b_features_13_1_running_mean, b_features_13_1_running_var,
               p_features_13_1_weight, p_features_13_1_bias, relu6=True)
    t = conv2d(t, p_features_13_3_weight, stride=1, padding=1, groups=384)
    t = bn_act(t, b_features_13_4_running_mean, b_features_13_4_running_var,
               p_features_13_4_weight, p_features_13_4_bias, relu6=True)
    t = conv2d(t, p_features_13_6_weight, stride=1, padding=0)
    t = bn_act(t, b_features_13_7_running_mean, b_features_13_7_running_var,
               p_features_13_7_weight, p_features_13_7_bias, relu6=False)

    # features[14]: InvertedResidual t=6, inp=96, oup=96, stride=1 (residual)
    residual = t
    t = conv2d(t, p_features_14_0_weight, stride=1, padding=0)
    t = bn_act(t, b_features_14_1_running_mean, b_features_14_1_running_var,
               p_features_14_1_weight, p_features_14_1_bias, relu6=True)
    t = conv2d(t, p_features_14_3_weight, stride=1, padding=1, groups=576)
    t = bn_act(t, b_features_14_4_running_