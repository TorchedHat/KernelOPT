import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
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

    # Compute channel index for each element
    # Layout: (N, C, H, W) -> channel = (offset // spatial) % num_channels
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


def bn_relu6(x, mean, var, weight, bias, eps=1e-5, apply_relu6=True):
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


def conv2d_bn_relu6(x, conv_w, bn_mean, bn_var, bn_w, bn_b,
                    stride=1, padding=0, groups=1, eps=1e-5, apply_relu6=True):
    # Use PyTorch's optimized conv2d (cuDNN)
    import torch.nn.functional as F
    y = F.conv2d(x, conv_w, bias=None, stride=stride, padding=padding, groups=groups)
    return bn_relu6(y, bn_mean, bn_var, bn_w, bn_b, eps=eps, apply_relu6=apply_relu6)


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
    import torch.nn.functional as F

    eps = 1e-5

    # features[0]: Conv2d(3, 32, 3, stride=2, padding=1) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_0_weight,
        b_features_1_running_mean, b_features_1_running_var,
        p_features_1_weight, p_features_1_bias,
        stride=2, padding=1, groups=1, eps=eps, apply_relu6=True
    )

    # features[3]: InvertedResidual t=1, inp=32, oup=16, stride=1
    # (no expand since t=1, just depthwise + pointwise)
    # depthwise: Conv2d(32, 32, 3, stride=1, padding=1, groups=32) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_3_0_weight,
        b_features_3_1_running_mean, b_features_3_1_running_var,
        p_features_3_1_weight, p_features_3_1_bias,
        stride=1, padding=1, groups=32, eps=eps, apply_relu6=True
    )
    # pointwise linear: Conv2d(32, 16, 1) + BN (no relu)
    x = conv2d_bn_relu6(
        x, p_features_3_3_weight,
        b_features_3_4_running_mean, b_features_3_4_running_var,
        p_features_3_4_weight, p_features_3_4_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=False
    )

    # features[4]: InvertedResidual t=6, inp=16, oup=24, stride=2
    # expand: Conv2d(16, 96, 1) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_4_0_weight,
        b_features_4_1_running_mean, b_features_4_1_running_var,
        p_features_4_1_weight, p_features_4_1_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=True
    )
    # depthwise: Conv2d(96, 96, 3, stride=2, padding=1, groups=96) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_4_3_weight,
        b_features_4_4_running_mean, b_features_4_4_running_var,
        p_features_4_4_weight, p_features_4_4_bias,
        stride=2, padding=1, groups=96, eps=eps, apply_relu6=True
    )
    # pointwise linear: Conv2d(96, 24, 1) + BN
    x = conv2d_bn_relu6(
        x, p_features_4_6_weight,
        b_features_4_7_running_mean, b_features_4_7_running_var,
        p_features_4_7_weight, p_features_4_7_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=False
    )

    # features[5]: InvertedResidual t=6, inp=24, oup=24, stride=1 (with residual)
    residual = x
    # expand: Conv2d(24, 144, 1) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_5_0_weight,
        b_features_5_1_running_mean, b_features_5_1_running_var,
        p_features_5_1_weight, p_features_5_1_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=True
    )
    # depthwise: Conv2d(144, 144, 3, stride=1, padding=1, groups=144) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_5_3_weight,
        b_features_5_4_running_mean, b_features_5_4_running_var,
        p_features_5_4_weight, p_features_5_4_bias,
        stride=1, padding=1, groups=144, eps=eps, apply_relu6=True
    )
    # pointwise linear: Conv2d(144, 24, 1) + BN
    x = conv2d_bn_relu6(
        x, p_features_5_6_weight,
        b_features_5_7_running_mean, b_features_5_7_running_var,
        p_features_5_7_weight, p_features_5_7_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=False
    )
    x = x + residual

    # features[6]: InvertedResidual t=6, inp=24, oup=32, stride=2
    # expand: Conv2d(24, 144, 1) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_6_0_weight,
        b_features_6_1_running_mean, b_features_6_1_running_var,
        p_features_6_1_weight, p_features_6_1_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=True
    )
    # depthwise: Conv2d(144, 144, 3, stride=2, padding=1, groups=144) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_6_3_weight,
        b_features_6_4_running_mean, b_features_6_4_running_var,
        p_features_6_4_weight, p_features_6_4_bias,
        stride=2, padding=1, groups=144, eps=eps, apply_relu6=True
    )
    # pointwise linear: Conv2d(144, 32, 1) + BN
    x = conv2d_bn_relu6(
        x, p_features_6_6_weight,
        b_features_6_7_running_mean, b_features_6_7_running_var,
        p_features_6_7_weight, p_features_6_7_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=False
    )

    # features[7]: InvertedResidual t=6, inp=32, oup=32, stride=1 (with residual)
    residual = x
    # expand: Conv2d(32, 192, 1) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_7_0_weight,
        b_features_7_1_running_mean, b_features_7_1_running_var,
        p_features_7_1_weight, p_features_7_1_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=True
    )
    # depthwise: Conv2d(192, 192, 3, stride=1, padding=1, groups=192) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_7_3_weight,
        b_features_7_4_running_mean, b_features_7_4_running_var,
        p_features_7_4_weight, p_features_7_4_bias,
        stride=1, padding=1, groups=192, eps=eps, apply_relu6=True
    )
    # pointwise linear: Conv2d(192, 32, 1) + BN
    x = conv2d_bn_relu6(
        x, p_features_7_6_weight,
        b_features_7_7_running_mean, b_features_7_7_running_var,
        p_features_7_7_weight, p_features_7_7_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=False
    )
    x = x + residual

    # features[8]: InvertedResidual t=6, inp=32, oup=32, stride=1 (with residual)
    residual = x
    # expand: Conv2d(32, 192, 1) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_8_0_weight,
        b_features_8_1_running_mean, b_features_8_1_running_var,
        p_features_8_1_weight, p_features_8_1_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=True
    )
    # depthwise: Conv2d(192, 192, 3, stride=1, padding=1, groups=192) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_8_3_weight,
        b_features_8_4_running_mean, b_features_8_4_running_var,
        p_features_8_4_weight, p_features_8_4_bias,
        stride=1, padding=1, groups=192, eps=eps, apply_relu6=True
    )
    # pointwise linear: Conv2d(192, 32, 1) + BN
    x = conv2d_bn_relu6(
        x, p_features_8_6_weight,
        b_features_8_7_running_mean, b_features_8_7_running_var,
        p_features_8_7_weight, p_features_8_7_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=False
    )
    x = x + residual

    # features[9]: InvertedResidual t=6, inp=32, oup=64, stride=2
    # expand: Conv2d(32, 192, 1) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_9_0_weight,
        b_features_9_1_running_mean, b_features_9_1_running_var,
        p_features_9_1_weight, p_features_9_1_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=True
    )
    # depthwise: Conv2d(192, 192, 3, stride=2, padding=1, groups=192) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_9_3_weight,
        b_features_9_4_running_mean, b_features_9_4_running_var,
        p_features_9_4_weight, p_features_9_4_bias,
        stride=2, padding=1, groups=192, eps=eps, apply_relu6=True
    )
    # pointwise linear: Conv2d(192, 64, 1) + BN
    x = conv2d_bn_relu6(
        x, p_features_9_6_weight,
        b_features_9_7_running_mean, b_features_9_7_running_var,
        p_features_9_7_weight, p_features_9_7_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=False
    )

    # features[10]: InvertedResidual t=6, inp=64, oup=64, stride=1 (with residual)
    residual = x
    # expand: Conv2d(64, 384, 1) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_10_0_weight,
        b_features_10_1_running_mean, b_features_10_1_running_var,
        p_features_10_1_weight, p_features_10_1_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=True
    )
    # depthwise: Conv2d(384, 384, 3, stride=1, padding=1, groups=384) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_10_3_weight,
        b_features_10_4_running_mean, b_features_10_4_running_var,
        p_features_10_4_weight, p_features_10_4_bias,
        stride=1, padding=1, groups=384, eps=eps, apply_relu6=True
    )
    # pointwise linear: Conv2d(384, 64, 1) + BN
    x = conv2d_bn_relu6(
        x, p_features_10_6_weight,
        b_features_10_7_running_mean, b_features_10_7_running_var,
        p_features_10_7_weight, p_features_10_7_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=False
    )
    x = x + residual

    # features[11]: InvertedResidual t=6, inp=64, oup=64, stride=1 (with residual)
    residual = x
    # expand: Conv2d(64, 384, 1) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_11_0_weight,
        b_features_11_1_running_mean, b_features_11_1_running_var,
        p_features_11_1_weight, p_features_11_1_bias,
        stride=1, padding=0, groups=1, eps=eps, apply_relu6=True
    )
    # depthwise: Conv2d(384, 384, 3, stride=1, padding=1, groups=384) + BN + ReLU6
    x = conv2d_bn_relu6(
        x, p_features_11_3_weight,
        b_features_11_4_running_mean, b_features_11_4_running_var,