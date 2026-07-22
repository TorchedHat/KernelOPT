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

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # channel index: layout is (N, C, H, W) -> channel = (offset // spatial) % num_channels
    chan = (offsets // spatial) % num_channels

    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


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

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    chan = (offsets // spatial) % num_channels

    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    tl.store(out_ptr + offsets, out, mask=mask)


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
def bn_add_relu_kernel(
    x_ptr, residual_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)

    chan = (offsets // spatial) % num_channels

    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b + res
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


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
def relu_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


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
def add_kernel(
    x_ptr, y_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x + y, mask=mask)


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
def linear_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, x, mask=mask)


def fused_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def fused_bn(x, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def fused_bn_add_relu(x, residual, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_relu_kernel[grid](
        x, residual, out, mean, var, weight, bias,
    )
    return out


def channel_shuffle(x, groups):
    batch_size, channels, height, width = x.shape
    channels_per_group = channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch_size, channels, height, width)
    return x


def conv2d_op(x, weight, stride=1, padding=0, groups=1):
    return F.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


def shufflenet_unit_no_shortcut(x, conv1_w, bn1_mean, bn1_var, bn1_w, bn1_b,
                                 conv2_w, bn2_mean, bn2_var, bn2_w, bn2_b,
                                 conv3_w, bn3_mean, bn3_var, bn3_w, bn3_b,
    out = conv2d_op(x, conv1_w, groups=groups)
    out = fused_bn_relu(out, bn1_mean, bn1_var, bn1_w, bn1_b)
    out = conv2d_op(out, conv2_w, padding=1, groups=mid_channels)
    out = fused_bn(out, bn2_mean, bn2_var, bn2_w, bn2_b)
    out = channel_shuffle(out, groups)
    out = conv2d_op(out, conv3_w, groups=groups)
    out = fused_bn_relu(out, bn3_mean, bn3_var, bn3_w, bn3_b)
    return out


def shufflenet_unit_with_shortcut(x, conv1_w, bn1_mean, bn1_var, bn1_w, bn1_b,
                                   conv2_w, bn2_mean, bn2_var, bn2_w, bn2_b,
                                   conv3_w, bn3_mean, bn3_var, bn3_w, bn3_b,
                                   sc_conv_w, sc_bn_mean, sc_bn_var, sc_bn_w, sc_bn_b,
    shortcut = conv2d_op(x, sc_conv_w)
    shortcut = fused_bn(shortcut, sc_bn_mean, sc_bn_var, sc_bn_w, sc_bn_b)

    out = conv2d_op(x, conv1_w, groups=groups)
    out = fused_bn_relu(out, bn1_mean, bn1_var, bn1_w, bn1_b)
    out = conv2d_op(out, conv2_w, padding=1, groups=mid_channels)
    out = fused_bn(out, bn2_mean, bn2_var, bn2_w, bn2_b)
    out = channel_shuffle(out, groups)
    out = conv2d_op(out, conv3_w, groups=groups)
    out = fused_bn_add_relu(out, shortcut, bn3_mean, bn3_var, bn3_w, bn3_b)
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
    # Initial conv + BN + ReLU + MaxPool
    out = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    out = fused_bn_relu(out, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias)
    out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

    # Stage 2, block 0 (with shortcut conv)
    out = shufflenet_unit_with_shortcut(
        out,
        p_stage2_0_conv1_weight, b_stage2_0_bn1_running_mean, b_stage2_0_bn1_running_var, p_stage2_0_bn1_weight, p_stage2_0_bn1_bias,
        p_stage2_0_conv2_weight, b_stage2_0_bn2_running_mean, b_stage2_0_bn2_running_var, p_stage2_0_bn2_weight, p_stage2_0_bn2_bias,
        p_stage2_0_conv3_weight, b_stage2_0_bn3_running_mean, b_stage2_0_bn3_running_var, p_stage2_0_bn3_weight, p_stage2_0_bn3_bias,
        p_stage2_0_shortcut_0_weight, b_stage2_0_shortcut_1_running_mean, b_stage2_0_shortcut_1_running_var, p_stage2_0_shortcut_1_weight, p_stage2_0_shortcut_1_bias,
        groups=3, mid_channels=60
    )

    # Stage 2, block 1 (identity shortcut)
    out = shufflenet_unit_no_shortcut(
        out,
        p_stage2_1_conv1_weight, b_stage2_1_bn1_running_mean, b_stage2_1_bn1_running_var, p_stage2_1_bn1_weight, p_stage2_1_bn1_bias,
        p_stage2_1_conv2_weight, b_stage2_1_bn2_running_mean, b_stage2_1_bn2_running_var, p_stage2_1_bn2_weight, p_stage2_1_bn2_bias,
        p_stage2_1_conv3_weight, b_stage2_1_bn3_running_mean, b_stage2_1_bn3_running_var, p_stage2_1_bn3_weight, p_stage2_1_bn3_bias,
        groups=3, mid_channels=60
    )

    # Stage 2, block 2 (identity shortcut)
    out = shufflenet_unit_no_shortcut(
        out,
        p_stage2_2_conv1_weight, b_stage2_2_bn1_running_mean, b_stage2_2_bn1_running_var, p_stage2_2_bn1_weight, p_stage2_2_bn1_bias,
        p_stage2_2_conv2_weight, b_stage2_2_bn2_running_mean, b_stage2_2_bn2_running_var, p_stage2_2_bn2_weight, p_stage2_2_bn2_bias,
        p_stage2_2_conv3_weight, b_stage2_2_bn3_running_mean, b_stage2_2_bn3_running_var, p_stage2_2_bn3_weight, p_stage2_2_bn3_bias,
        groups=3, mid_channels=60
    )

    # Stage 3, block 0 (with shortcut conv)
    out = shufflenet_unit_with_shortcut(
        out,
        p_stage3_0_conv1_weight, b_stage3_0_bn1_running_mean, b_stage3_0_bn1_running_var, p_stage3_0_bn1_weight, p_stage3_0_bn1_bias,
        p_stage3_0_conv2_weight, b_stage3_0_bn2_running_mean, b_stage3_0_bn2_running_var, p_stage3_0_bn2_weight, p_stage3_0_bn2_bias,
        p_stage3_0_conv3_weight, b_stage3_0_bn3_running_mean, b_stage3_0_bn3_running_var, p_stage3_0_bn3_weight, p_stage3_0_bn3_bias,
        p_stage3_0_shortcut_0_weight, b_stage3_0_shortcut_1_running_mean, b_stage3_0_shortcut_1_running_var, p_stage3_0_shortcut_1_weight, p_stage3_0_shortcut_1_bias,
        groups=3, mid_channels=120
    )

    # Stage 3, blocks 1-6 (identity shortcut)
    for conv1_w, bn1_m, bn1_v, bn1_w, bn1_b, conv2_w, bn2_m, bn2_v, bn2_w, bn2_b, conv3_w, bn3_m, bn3_v, bn3_w, bn3_b in [
        (p_stage3_1_conv1_weight, b_stage3_1_bn1_running_mean, b_stage3_1_bn1_running_var, p_stage3_1_bn1_weight, p_stage3_1_bn1_bias,
         p_stage3_1_conv2_weight, b_stage3_1_bn2_running_mean, b_stage3_1_bn2_running_var, p_stage3_1_bn2_weight, p_stage3_1_bn2_bias,