import torch
import triton
import triton.language as tl
import torch.nn.functional as F


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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index: layout is (N, C, H, W) -> channel = (offset / spatial) % C
    c_idx = (offsets // spatial) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std
    x_scaled = x_norm * w + b

    # ReLU6 (hardtanh with min=0, max=6)
    x_relu6 = tl.minimum(tl.maximum(x_scaled, 0.0), 6.0)

    tl.store(out_ptr + offsets, x_relu6, mask=mask)


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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c_idx = (offsets // spatial) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std
    x_scaled = x_norm * w + b

    tl.store(out_ptr + offsets, x_scaled, mask=mask)


def apply_bn_relu6(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu6_kernel[grid](
        x, out, running_mean, running_var, weight, bias,
    )
    return out


def apply_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out, running_mean, running_var, weight, bias,
    )
    return out


def conv2d_bn_relu6(x, conv_w, bn_w, bn_b, bn_mean, bn_var,
                    stride=1, padding=0, groups=1, eps=1e-5):
    x = F.conv2d(x, conv_w, bias=None, stride=stride, padding=padding, groups=groups)
    x = apply_bn_relu6(x, bn_w, bn_b, bn_mean, bn_var, eps)
    return x


def conv2d_bn(x, conv_w, bn_w, bn_b, bn_mean, bn_var,
              stride=1, padding=0, groups=1, eps=1e-5):
    x = F.conv2d(x, conv_w, bias=None, stride=stride, padding=padding, groups=groups)
    x = apply_bn(x, bn_w, bn_b, bn_mean, bn_var, eps)
    return x


def inverted_residual(x, expand_ratio,
                      pw_w, pw_bn_w, pw_bn_b, pw_bn_mean, pw_bn_var,
                      dw_w, dw_bn_w, dw_bn_b, dw_bn_mean, dw_bn_var,
                      proj_w, proj_bn_w, proj_bn_b, proj_bn_mean, proj_bn_var,
    identity = x
    if expand_ratio != 1:
        x = conv2d_bn_relu6(x, pw_w, pw_bn_w, pw_bn_b, pw_bn_mean, pw_bn_var,
                            stride=1, padding=0)
    channels = dw_w.shape[0]
    x = conv2d_bn_relu6(x, dw_w, dw_bn_w, dw_bn_b, dw_bn_mean, dw_bn_var,
                        stride=stride, padding=1, groups=channels)
    x = conv2d_bn(x, proj_w, proj_bn_w, proj_bn_b, proj_bn_mean, proj_bn_var,
                  stride=1, padding=0)
    if use_res_connect:
        x = x + identity
    return x


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
    # First conv: 3->32, stride=2, pad=1, kernel=3
    x = conv2d_bn_relu6(
        x, p_features_0_weight,
        p_features_1_weight, p_features_1_bias,
        b_features_1_running_mean, b_features_1_running_var,
        stride=2, padding=1
    )

    # Block 3: t=1, c=16, n=1, s=1 (expand_ratio=1, no pw conv)
    # dw conv: 32->32, stride=1, pad=1, groups=32
    x = conv2d_bn_relu6(
        x, p_features_3_0_weight,
        p_features_3_1_weight, p_features_3_1_bias,
        b_features_3_1_running_mean, b_features_3_1_running_var,
        stride=1, padding=1, groups=32
    )
    # proj conv: 32->16, stride=1
    x = conv2d_bn(
        x, p_features_3_3_weight,
        p_features_3_4_weight, p_features_3_4_bias,
        b_features_3_4_running_mean, b_features_3_4_running_var,
        stride=1, padding=0
    )

    # Block 4: t=6, c=24, n=2, s=2
    # Block 4 iter 0: inp=16, oup=24, stride=2, expand=6 -> hidden=96
    x = conv2d_bn_relu6(
        x, p_features_4_0_weight,
        p_features_4_1_weight, p_features_4_1_bias,
        b_features_4_1_running_mean, b_features_4_1_running_var,
        stride=1, padding=0
    )
    x = conv2d_bn_relu6(
        x, p_features_4_3_weight,
        p_features_4_4_weight, p_features_4_4_bias,
        b_features_4_4_running_mean, b_features_4_4_running_var,
        stride=2, padding=1, groups=96
    )
    x = conv2d_bn(
        x, p_features_4_6_weight,
        p_features_4_7_weight, p_features_4_7_bias,
        b_features_4_7_running_mean, b_features_4_7_running_var,
        stride=1, padding=0
    )

    # Block 5: t=6, c=24, n=2, s=2 -> iter 1: inp=24, oup=24, stride=1, use_res=True
    x_res = x
    x = conv2d_bn_relu6(
        x, p_features_5_0_weight,
        p_features_5_1_weight, p_features_5_1_bias,
        b_features_5_1_running_mean, b_features_5_1_running_var,
        stride=1, padding=0
    )
    x = conv2d_bn_relu6(
        x, p_features_5_3_weight,
        p_features_5_4_weight, p_features_5_4_bias,
        b_features_5_4_running_mean, b_features_5_4_running_var,
        stride=1, padding=1, groups=144
    )
    x = conv2d_bn(
        x, p_features_5_6_weight,
        p_features_5_7_weight, p_features_5_7_bias,
        b_features_5_7_running_mean, b_features_5_7_running_var,
        stride=1, padding=0
    )
    x = x + x_res

    # Block 6: t=6, c=32, n=3, s=2 -> iter 0: inp=24, oup=32, stride=2
    x = conv2d_bn_relu6(
        x, p_features_6_0_weight,
        p_features_6_1_weight, p_features_6_1_bias,
        b_features_6_1_running_mean, b_features_6_1_running_var,
        stride=1, padding=0
    )
    x = conv2d_bn_relu6(
        x, p_features_6_3_weight,
        p_features_6_4_weight, p_features_6_4_bias,
        b_features_6_4_running_mean, b_features_6_4_running_var,
        stride=2, padding=1, groups=144
    )
    x = conv2d_bn(
        x, p_features_6_6_weight,
        p_features_6_7_weight, p_features_6_7_bias,
        b_features_6_7_running_mean, b_features_6_7_running_var,
        stride=1, padding=0
    )

    # Block 7: t=6, c=32, n=3, s=2 -> iter 1: inp=32, oup=32, stride=1, use_res=True
    x_res = x
    x = conv2d_bn_relu6(
        x, p_features_7_0_weight,
        p_features_7_1_weight, p_features_7_1_bias,
        b_features_7_1_running_mean, b_features_7_1_running_var,
        stride=1, padding=0
    )
    x = conv2d_bn_relu6(
        x, p_features_7_3_weight,
        p_features_7_4_weight, p_features_7_4_bias,
        b_features_7_4_running_mean, b_features_7_4_running_var,
        stride=1, padding=1, groups=192
    )
    x = conv2d_bn(
        x, p_features_7_6_weight,
        p_features_7_7_weight, p_features_7_7_bias,
        b_features_7_7_running_mean, b_features_7_7_running_var,
        stride=1, padding=0
    )
    x = x + x_res

    # Block 8: t=6, c=32, n=3, s=2 -> iter 2: inp=32, oup=32, stride=1, use_res=True
    x_res = x
    x = conv2d_bn_relu6(
        x, p_features_8_0_weight,
        p_features_8_1_weight, p_features_8_1_bias,
        b_features_8_1_running_mean, b_features_8_1_running_var,
        stride=1, padding=0
    )
    x = conv2d_bn_relu6(
        x, p_features_8_3_weight,
        p_features_8_4_weight, p_features_8_4_bias,
        b_features_8_4_running_mean, b_features_8_4_running_var,
        stride=1, padding=1, groups=192
    )
    x = conv2d_bn(
        x, p_features_8_6_weight,
        p_features_8_7_weight, p_features_8_7_bias,
        b_features_8_7_running_mean, b_features_8_7_running_var,
        stride=1, padding=0
    )
    x = x + x_res

    # Block 9: t=6, c=64, n=4, s=2 -> iter 0: inp=32, oup=64, stride=2
    x = conv2d_bn_relu6(
        x, p_features_9_0_weight,
        p_features_9_1_weight, p_features_9_1_bias,
        b_features_9_1_running_mean, b_features_9_1_running_var,
        stride=1, padding=0
    )
    x = conv2d_bn_relu6(
        x, p_features_9_3_weight,
        p_features_9_4_weight, p_features_9_4_bias,
        b_features_9_4_running_mean, b_features_9_4_running_var,
        stride=2, padding=1, groups=192
    )
    x = conv2d_bn(
        x, p_features_9_6_weight,
        p_features_9_7_weight, p_features_9_7_bias,
        b_features_9_7_running_mean, b_features_9_7_running_var,
        stride=1, padding=0
    )

    # Block 10: t=6, c=64, n=4, s=2 -> iter 1: inp=64, oup=64, stride=1, use_res=True
    x_res = x
    x = conv2d_bn_relu6(
        x, p_features_10_0_weight,
        p_features_10_1_weight, p_features_10_1_bias,
        b_features_10_1_running_mean, b_features_10_1_running_var,
        stride=1, padding=0
    )
    x = conv2d_bn