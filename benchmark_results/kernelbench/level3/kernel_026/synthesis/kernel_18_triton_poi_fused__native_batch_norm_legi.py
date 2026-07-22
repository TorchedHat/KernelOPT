import torch
import triton
import triton.language as tl
import torch.nn.functional as F

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
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, numel,
    eps,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute channel index - layout is NCHW, so channel = (offset // HW) % C
    # But we need to handle this carefully
    c_idx = (offsets // 1) % C  # This won't work for NCHW
    
    # For NCHW layout: offset = n*C*H*W + c*H*W + h*W + w
    # c = (offset // HW) % C - but HW is not known here
    # We'll pass HW as a parameter


import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel', 'HW'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    HW, C, numel,
    eps: tl.constexpr,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if apply_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel', 'HW'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    HW, C, numel,
    eps,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // HW) % C

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    if apply_relu:
        out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_add_relu_kernel(
    x_ptr, residual_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    HW, C, numel,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """BN + add residual + relu fused kernel"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // HW) % C

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b + res
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel', 'HW'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    groups,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Channel shuffle: reshape to (N, groups, C//groups, H, W), transpose dims 1&2, reshape back"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Output index: n, c_out, h, w
    HW = H * W
    CHW = C * HW
    n = offsets // CHW
    rem = offsets % CHW
    c_out = rem // HW
    rem2 = rem % HW
    h = rem2 // W
    w = rem2 % W

    # c_out = g * (C//groups) + c_in_group
    # After shuffle: c_in = c_in_group * groups + g
    channels_per_group = C // groups
    g = c_out // channels_per_group
    c_in_group = c_out % channels_per_group
    c_in = c_in_group * groups + g

    in_offset = n * CHW + c_in * HW + h * W + w
    val = tl.load(x_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def bn_relu(x, mean, var, weight, bias, eps=1e-5, relu=True):
    N, C, H, W = x.shape
    numel = x.numel()
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        HW, C, numel,
        eps,
        apply_relu=relu,
    )
    return out


def bn_add_relu(x, residual, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = x.numel()
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_add_relu_kernel[grid](
        x, residual, mean, var, weight, bias,
        out,
        HW, C, numel,
        eps,
    )
    return out


def channel_shuffle(x, groups):
    N, C, H, W = x.shape
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
        x, out,
        N, C, H, W,
        groups,
        numel,
    )
    return out


def conv2d(x, weight, stride=1, padding=0, groups=1):
    return torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


def shufflenet_unit(x, conv1_w, bn1_mean, bn1_var, bn1_w, bn1_b,
                    conv2_w, bn2_mean, bn2_var, bn2_w, bn2_b,
                    conv3_w, bn3_mean, bn3_var, bn3_w, bn3_b,
                    shortcut_conv_w, shortcut_bn_mean, shortcut_bn_var, shortcut_bn_w, shortcut_bn_b,
    # conv1: 1x1 group conv
    out = conv2d(x, conv1_w, stride=1, padding=0, groups=groups)
    out = bn_relu(out, bn1_mean, bn1_var, bn1_w, bn1_b, relu=True)

    # conv2: depthwise 3x3
    mid_channels = out.shape[1]
    out = conv2d(out, conv2_w, stride=1, padding=1, groups=mid_channels)
    out = bn_relu(out, bn2_mean, bn2_var, bn2_w, bn2_b, relu=False)

    # channel shuffle
    out = channel_shuffle(out, groups)

    # conv3: 1x1 group conv
    out = conv2d(out, conv3_w, stride=1, padding=0, groups=groups)

    # shortcut
    if has_shortcut_conv:
        shortcut = conv2d(x, shortcut_conv_w, stride=1, padding=0, groups=1)
        shortcut = bn_relu(shortcut, shortcut_bn_mean, shortcut_bn_var, shortcut_bn_w, shortcut_bn_b, relu=False)
    else:
        shortcut = x

    out = bn_add_relu(out, shortcut, bn3_mean, bn3_var, bn3_w, bn3_b)
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
    import torch.nn.functional as F

    # Initial conv + bn + relu + maxpool
    x = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    x = bn_relu(x, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias, relu=True)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    # Stage 2, block 0 (has shortcut conv, groups=3, in=24, out=240, mid=60)
    x = shufflenet_unit(
        x,
        p_stage2_0_conv1_weight, b_stage2_0_bn1_running_mean, b_stage2_0_bn1_running_var,
        p_stage2_0_bn1_weight, p_stage2_0_bn1_bias,
        p_stage2_0_conv2_weight, b_stage2_0_bn2_running_mean, b_stage2_0_bn2_running_var,
        p_stage2_0_bn2_weight, p_stage2_0_bn2_bias,
        p_stage2_0_conv3_weight, b_stage2_0_bn3_running_mean, b_stage2_0_bn3_running_var,
        p_stage2_0_bn3_weight, p_stage2_0_bn3_bias,
        p_stage2_0_shortcut_0_weight,
        b_stage2_0_shortcut_1_running_mean, b_stage2_0_shortcut_1_running_var,
        p_stage2_0_shortcut_1_weight, p_stage2_0_shortcut_1_bias,
        groups=3, has_shortcut_conv=True,
    )

    # Stage 2, block 1 (no shortcut conv, groups=3)
    x = shufflenet_unit(
        x,
        p_stage2_1_conv1_weight, b_stage2_1_bn1_running_mean, b_stage2_1_bn1_running_var,
        p_stage2_1_bn1_weight, p_stage2_1_bn1_bias,
        p_stage2_1_conv2_weight, b_stage2_1_bn2_running_mean, b_stage2_1_bn2_running_var,
        p_stage2_1_bn2_weight, p_stage2_1_bn2_bias,
        p_stage2_1_conv3_weight, b_stage2_1_bn3_running_mean, b_stage2_1_bn3_running_var,
        p_stage2_1_bn3_weight, p_stage2_1_bn3_bias,
        None, None, None, None, None,
        groups=3, has_shortcut_conv=False,
    )

    # Stage 2, block 2 (no shortcut conv, groups=3)
    x = shufflenet_unit(
        x,
        p_stage2_2_conv1_weight, b_stage2_2_bn1_running_mean, b_stage2_2_bn1_running_var,
        p_stage2_2_bn1_weight, p_stage2_2_bn1_bias,
        p_stage2_2_conv2_weight, b_stage2_2_bn2_running_mean, b_stage2_2_bn2_running_var,
        p_stage2_2_bn2_weight, p_stage2_2_bn2_bias,
        p_stage2_2_conv3_weight, b_stage2_2_bn3_running_mean, b_stage2_2_bn3_running_var,
        p_stage2_2_bn3_weight, p_stage2_2_bn3_bias,
        None, None, None, None, None,
        groups=3, has_shortcut_conv=False,
    )

    # Stage 3, block 0 (has shortcut conv, groups=3, in=240, out=480, mid=120)
    x = shufflenet_unit(
        x,
        p_stage3_0_conv1_weight, b_stage3_0_bn1_running_mean, b_stage3_0_bn1_running_var,
        p_stage3_0_bn1_weight, p_stage3_0_bn1_bias,
        p_stage3_0_conv2_weight, b_stage3_0_bn2_running_mean, b_stage3_0_bn2_running_var,
        p_stage3_0_bn2_weight, p_stage3_0_bn2_bias,
        p_stage3_0_conv3_weight, b_stage3_0_bn3_running_mean, b_stage3_0_bn3_running_var,
        p_stage3_0_bn3_weight, p_stage3_0_bn3_bias,
        p_stage3_0_shortcut_0_weight,
        b_stage3_0_shortcut_1_running_mean, b_stage3_0_shortcut_1_running_var,
        p_stage3_0_shortcut_1_weight, p_stage3_0_shortcut_1_bias,
        groups=3, has_shortcut_conv=True,
    )

    # Stage 3, blocks 1-6 (no shortcut conv, groups=3)
    for conv1_w, bn1_mean, bn1_var, bn1_w, bn1_b, \
        conv2_w, bn2_mean, bn2_var, bn2_w, bn2_b, \
        (p_stage3_1_conv1_weight, b_stage3_1_bn1_running_mean, b_stage3_1_bn1_running_var,
         p_stage3_1_bn1_weight, p_stage3_1_bn1_bias,
         p_stage3_1_conv2_weight, b_stage3_1_bn2_running_mean, b_stage3_1_bn2_running_var,