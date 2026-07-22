import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import argparse

# Fused BatchNorm (inference) + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    eps: tl.constexpr,
    C: tl.constexpr,
    HW,
    numel,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # offsets layout: (N, C, H, W) -> flat index
    # channel index = (offset // HW) % C
    c_idx = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, tl.zeros_like(out))
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_add_relu_kernel(
    x_ptr, residual_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    eps: tl.constexpr,
    C: tl.constexpr,
    HW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c_idx = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b + res
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Channel shuffle kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    groups,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # output index -> (n, c_out, h, w)
    w_idx = offsets % W
    tmp = offsets // W
    h_idx = tmp % H
    tmp = tmp // H
    c_out = tmp % C
    n_idx = tmp // C
    
    # channel shuffle: c_out = g * (C//groups) + c_in_group
    # inverse: c_in = (c_out % groups) * (C//groups) + (c_out // groups)
    channels_per_group = C // groups
    c_in = (c_out % groups) * channels_per_group + (c_out // groups)
    
    in_offsets = n_idx * C * H * W + c_in * H * W + h_idx * W + w_idx
    
    val = tl.load(x_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def apply_bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
        eps, C, HW, numel, do_relu,
    )
    return out


def apply_bn_add_relu(x, residual, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_relu_kernel[grid](
        x, residual, out, mean, var, weight, bias,
        eps, C, HW, numel,
    )
    return out


def channel_shuffle(x, groups):
    N, C, H, W = x.shape
    numel = N * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
        x, out, N, C, H, W, groups, numel,
    )
    return out


def conv2d_op(x, weight, stride=1, padding=0, groups=1):
    return torch.ops.aten.convolution(x, weight, None, 
                                       [stride, stride] if isinstance(stride, int) else stride,
                                       [padding, padding] if isinstance(padding, int) else padding,


def shufflenet_unit(x, 
                    conv1_w, bn1_mean, bn1_var, bn1_w, bn1_b,
                    conv2_w, bn2_mean, bn2_var, bn2_w, bn2_b,
                    conv3_w, bn3_mean, bn3_var, bn3_w, bn3_b,
                    shortcut_conv_w=None, shortcut_bn_mean=None, shortcut_bn_var=None,
                    shortcut_bn_w=None, shortcut_bn_b=None,
                    groups=3):
    # shortcut
    if shortcut_conv_w is not None:
        sc = conv2d_op(x, shortcut_conv_w, stride=1, padding=0, groups=1)
        sc = apply_bn_relu(sc, shortcut_bn_mean, shortcut_bn_var, shortcut_bn_w, shortcut_bn_b, do_relu=False)
    else:
        sc = x
    
    # conv1: 1x1 group conv
    out = conv2d_op(x, conv1_w, stride=1, padding=0, groups=groups)
    out = apply_bn_relu(out, bn1_mean, bn1_var, bn1_w, bn1_b, do_relu=True)
    
    # conv2: depthwise 3x3
    mid_c = out.shape[1]
    out = conv2d_op(out, conv2_w, stride=1, padding=1, groups=mid_c)
    out = apply_bn_relu(out, bn2_mean, bn2_var, bn2_w, bn2_b, do_relu=False)
    
    # channel shuffle
    out = channel_shuffle(out, groups)
    
    # conv3: 1x1 group conv
    out = conv2d_op(out, conv3_w, stride=1, padding=0, groups=groups)
    out = apply_bn_relu(out, bn3_mean, bn3_var, bn3_w, bn3_b, do_relu=True)
    
    # add shortcut
    out = out + sc
    return out


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_stage2_0_conv1_weight, p_stage2_0_bn1_weight, p_stage2_0_bn1_bias,
    p_stage2_0_conv2_weight, p_stage2_0_bn2_weight, p_stage2_0_bn2_bias,
    p_stage2_0_conv3_weight, p_stage2_0_bn3_weight, p_stage2_0_bn3_bias,
    p_stage2_0_shortcut_0_weight, p_stage2_0_shortcut_1_weight, p_stage2_0_shortcut_1_bias,
    p_stage2_1_conv1_weight, p_stage2_1_bn1_weight, p_stage2_1_bn1_bias,
    p_stage2_1_conv2_weight, p_stage2_1_bn2_weight, p_stage2_1_bn2_bias,
    p_stage2_1_conv3_weight, p_stage2_1_bn3_weight, p_stage2_1_bn3_bias,
    p_stage2_2_conv1_weight, p_stage2_2_bn1_weight, p_stage2_2_bn1_bias,
    p_stage2_2_conv2_weight, p_stage2_2_bn2_weight, p_stage2_2_bn2_bias,
    p_stage2_2_conv3_weight, p_stage2_2_bn3_weight, p_stage2_2_bn3_bias,
    p_stage3_0_conv1_weight, p_stage3_0_bn1_weight, p_stage3_0_bn1_bias,
    p_stage3_0_conv2_weight, p_stage3_0_bn2_weight, p_stage3_0_bn2_bias,
    p_stage3_0_conv3_weight, p_stage3_0_bn3_weight, p_stage3_0_bn3_bias,
    p_stage3_0_shortcut_0_weight, p_stage3_0_shortcut_1_weight, p_stage3_0_shortcut_1_bias,
    p_stage3_1_conv1_weight, p_stage3_1_bn1_weight, p_stage3_1_bn1_bias,
    p_stage3_1_conv2_weight, p_stage3_1_bn2_weight, p_stage3_1_bn2_bias,
    p_stage3_1_conv3_weight, p_stage3_1_bn3_weight, p_stage3_1_bn3_bias,
    p_stage3_2_conv1_weight, p_stage3_2_bn1_weight, p_stage3_2_bn1_bias,
    p_stage3_2_conv2_weight, p_stage3_2_bn2_weight, p_stage3_2_bn2_bias,
    p_stage3_2_conv3_weight, p_stage3_2_bn3_weight, p_stage3_2_bn3_bias,
    p_stage3_3_conv1_weight, p_stage3_3_bn1_weight, p_stage3_3_bn1_bias,
    p_stage3_3_conv2_weight, p_stage3_3_bn2_weight, p_stage3_3_bn2_bias,
    p_stage3_3_conv3_weight, p_stage3_3_bn3_weight, p_stage3_3_bn3_bias,
    p_stage3_4_conv1_weight, p_stage3_4_bn1_weight, p_stage3_4_bn1_bias,
    p_stage3_4_conv2_weight, p_stage3_4_bn2_weight, p_stage3_4_bn2_bias,
    p_stage3_4_conv3_weight, p_stage3_4_bn3_weight, p_stage3_4_bn3_bias,
    p_stage3_5_conv1_weight, p_stage3_5_bn1_weight, p_stage3_5_bn1_bias,
    p_stage3_5_conv2_weight, p_stage3_5_bn2_weight, p_stage3_5_bn2_bias,
    p_stage3_5_conv3_weight, p_stage3_5_bn3_weight, p_stage3_5_bn3_bias,
    p_stage3_6_conv1_weight, p_stage3_6_bn1_weight, p_stage3_6_bn1_bias,
    p_stage3_6_conv2_weight, p_stage3_6_bn2_weight, p_stage3_6_bn2_bias,
    p_stage3_6_conv3_weight, p_stage3_6_bn3_weight, p_stage3_6_bn3_bias,
    p_stage4_0_conv1_weight, p_stage4_0_bn1_weight, p_stage4_0_bn1_bias,
    p_stage4_0_conv2_weight, p_stage4_0_bn2_weight, p_stage4_0_bn2_bias,
    p_stage4_0_conv3_weight, p_stage4_0_bn3_weight, p_stage4_0_bn3_bias,
    p_stage4_0_shortcut_0_weight, p_stage4_0_shortcut_1_weight, p_stage4_0_shortcut_1_bias,
    p_stage4_1_conv1_weight, p_stage4_1_bn1_weight, p_stage4_1_bn1_bias,
    p_stage4_1_conv2_weight, p_stage4_1_bn2_weight, p_stage4_1_bn2_bias,
    p_stage4_1_conv3_weight, p_stage4_1_bn3_weight, p_stage4_1_bn3_bias,
    p_stage4_2_conv1_weight, p_stage4_2_bn1_weight, p_stage4_2_bn1_bias,
    p_stage4_2_conv2_weight, p_stage4_2_bn2_weight, p_stage4_2_bn2_bias,
    p_stage4_2_conv3_weight, p_stage4_2_bn3_weight, p_stage4_2_bn3_bias,
    p_conv5_weight, p_bn5_weight, p_bn5_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_stage2_0_bn1_running_mean, b_stage2_0_bn1_running_var,
    b_stage2_0_bn2_running_mean, b_stage2_0_bn2_running_var,
    b_stage2_0_bn3_running_mean, b_stage2_0_bn3_running_var,
    b_stage2_0_shortcut_1_running_mean, b_stage2_0_shortcut_1_running_var,
    b_stage2_1_bn1_running_mean, b_stage2_1_bn1_running_var,
    b_stage2_1_bn2_running_mean, b_stage2_1_bn2_running_var,
    b_stage2_1_bn3_running_mean, b_stage2_1_bn3_running_var,
    b_stage2_2_bn1_running_mean, b_stage2_2_bn1_running_var,
    b_stage2_2_bn2_running_mean, b_stage2_2_bn2_running_var,
    b_stage2_2_bn3_running_mean, b_stage2_2_bn3_running_var,
    b_stage3_0_bn1_running_mean, b_stage3_0_bn1_running_var,
    b_stage3_0_bn2_running_mean, b_stage3_0_bn2_running_var,
    b_stage3_0_bn3_running_mean, b_stage3_0_bn3_running_var,
    b_stage3_0_shortcut_1_running_mean, b_stage3_0_shortcut_1_running_var,
    b_stage3_1_bn1_running_mean, b_stage3_1_bn1_running_var,
    b_stage3_1_bn2_running_mean, b_stage3_1_bn2_running_var,
    b_stage3_1_bn3_running_mean, b_stage3_1_bn3_running_var,
    b_stage3_2_bn1_running_mean, b_stage3_2_bn1_running_var,
    b_stage3_2_bn2_running_mean, b_stage3_2_bn2_running_var,
    b_stage3_2_bn3_running_mean, b_stage3_2_bn3_running_var,
    b_stage3_3_bn1_running_mean, b_stage3_3_bn1_running_var,
    b_stage3_3_bn2_running_mean, b_stage3_3_bn2_running_var,
    b_stage3_3_bn3_running_mean, b_stage3_3_bn3_running_var,
    b_stage3_4_bn1_running_mean, b_stage3_4_bn1_running_var,
    b_stage3_4_bn2_running_mean, b_stage3_4_bn2_running_var,
    b_stage3_4_bn3_running_mean, b_stage3_4_bn3_running_var,
    b_stage3_5_bn1_running_mean, b_stage3_5_bn1_running_var,
    b_stage3_5_bn2_running_mean, b_stage3_5_bn2_running_var,
    b_stage3_5_bn3_running_mean, b_stage3_5_bn3_running_var,
    b_stage3_6_bn1_running_mean, b_stage3_6_bn1_running_var,
    b_stage3_6_bn2_running_mean, b_stage3_6_bn2_running_var,
    b_stage3_6_bn3_running_mean, b_stage3_6_bn3_running_var,
    b_stage4_0_bn1_running_mean, b_stage4_0_bn1_running_var,
    b_stage4_0_bn2_running_mean, b_stage4_0_bn2_running_var,
    b_stage4_0_bn3_running_mean, b_stage4_0_bn3_running_var,
    b_stage4_0_shortcut_1_running_mean, b_stage4_0_shortcut_1_running_var,
    b_stage4_1_bn1_running_mean, b_stage4_1_bn1_running_var,
    b_stage4_1_bn2_running_mean, b_stage4_1_bn2_running_var,
    b_stage4_1_bn3_running_mean, b_stage4_1_bn3_running_var,
    b_stage4_2_bn1_running_mean, b_stage4_2_bn1_running_var,
    b_stage4_2_bn2_running_mean, b_stage4_2_bn2_running_var,
    b_stage4_2_bn3_running_mean, b_stage4_2_bn3_running_var,
    b_bn5_running_mean, b_bn5_running_var,
    x,
):
    # Initial conv + bn + relu + maxpool
    out = torch.ops.aten.convolution(x, p_conv1_weight, None, [2,2], [1,1], [1,1], False, [0,0], 1)
    out = apply_bn_relu(out, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias, do_relu=True)
    out = torch.nn.functional.max_pool2d(out, kernel_size=3, stride=2, padding=1)
    
    # Stage 2: 3 units, groups=3
    # Unit 0: in=24, out=240, has shortcut conv
    out = shufflenet_unit(
        out,
        p_stage2_0_conv1_weight, b_stage2_0_bn1_running_mean, b_stage2_0_bn1_running_var, p_stage2_0_bn1_weight, p_stage2_0_bn1_bias,
        p_stage2_0_conv2_weight, b_stage2_0_bn2_running_mean, b_stage2_0_bn2_running_var, p_stage2_0_bn2_weight, p_stage2_0_bn2_bias,
        p_stage2_0_conv3_weight, b_stage2_0_bn3_running_mean, b_stage2_0_bn3_running_var, p_stage2_0_bn3_weight, p_stage2_0_bn3_bias,
        p_stage2_0_shortcut_0_weight, b_stage2_0_shortcut_1_running_mean, b_stage2_0_shortcut_1_running_var,
        p_stage2_0_shortcut_1_weight, p_stage2_0_shortcut_1_bias,
        groups=3
    )
    
    # Unit 1: in=240, out=240, no shortcut conv
    out = shufflenet_unit(
        out,
        p_stage2_1_conv1_weight, b_stage2_1_bn1_running_mean, b_stage2_1_bn1_running_var, p_stage2_1_bn1_weight, p_stage2_1_bn1_bias,
        p_stage2_1_conv2_weight, b_stage2_1_bn2_running_mean, b_stage2_1_bn2_running_var, p_stage2_1_bn2_weight, p_stage2_1_bn2_bias,
        p_stage2_1_conv3_weight, b_stage2_1_bn3_running_mean, b_stage2_1_bn3_running_var, p_stage2_1_bn3_weight, p_stage2_1_bn3_bias,
        groups=3
    )
    
    # Unit 2: in=240, out=240, no shortcut conv
    out = shufflenet_unit(
        out,
        p_stage2_2_conv1_weight, b_stage2_2_bn1_running_mean, b_stage2_2_bn1_running_var, p_stage2_2_bn1_weight, p_stage2_2_bn1_bias,
        p_stage2_2_conv2_weight, b_stage2_2_bn2_running_mean, b_stage2_2_bn2_running_var, p_stage2_2_bn2_weight, p_stage2_2_bn2_bias,
        p_stage2_2_conv3_weight, b_stage2_2_bn3_running_mean, b_stage2_2_bn3_running_var, p_stage2_2_bn3_weight, p_stage2_2_bn3_bias,
        groups=3
    )
    
    # Stage 3: 7 units, groups=3
    # Unit 0: in=240, out=480, has shortcut conv
    out = shufflenet_unit(
        out,
        p_stage3_0_conv1_weight, b_stage3_0_bn1_running_mean, b_stage3_0_bn1_running_var, p_stage3_0_bn1_weight, p_stage3_0_bn1_bias,
        p_stage3_0_conv2_weight, b_stage3_0_bn2_running_mean, b_stage3_0_bn2_running_var, p_stage3_0_bn2_weight, p_stage3_0_bn2_bias,
        p_stage3_0_conv3_weight, b_stage3_0_bn3_running_mean, b_stage3_0_bn3_running_var, p_stage3_0_bn3_weight, p_stage3_0_bn3_bias,
        p_stage3_0_shortcut_0_weight, b_stage3_0_shortcut_1_running_mean, b_stage3_0_shortcut_1_running_var,
        p_stage3_0_shortcut_1_weight, p_stage3_0_shortcut_1_bias,
        groups=3
    )
    
    # Units 1-6: no shortcut conv
    for conv1_w, bn1_m, bn1_v, bn1_w_, bn1_b_, \
        conv2_w, bn2_m, bn2_v, bn2_w_, bn2_b_, \
        (p_stage3_1_conv1_weight, b_stage3_1_bn1_running_mean, b_stage3_1_bn1_running_var, p_stage3_1_bn1_weight, p_stage3_1_bn1_bias,
         p_stage3_1_conv2_weight, b_stage3_1_bn2_running_mean, b_stage3_1_bn2_running_var, p_stage3_1_bn2_weight, p_stage3_1_bn2_bias,
         p_stage3_1_conv3_weight, b_stage3_1_bn3_running_mean, b_stage3_1_bn3_running_var, p_stage3_1_bn3_weight, p_stage3_1_bn3_bias),
        (p_stage3_2_conv1_weight, b_stage3_2_bn1_running_mean, b_stage3_2_bn1_running_var, p_stage3_2_bn1_weight, p_stage3_2_bn1_bias,
         p_stage3_2_conv2_weight, b_stage3_2_bn2_running_mean, b_stage3_2_bn2_running_var, p_stage3_2_bn2_weight, p_stage3_2_bn2_bias,