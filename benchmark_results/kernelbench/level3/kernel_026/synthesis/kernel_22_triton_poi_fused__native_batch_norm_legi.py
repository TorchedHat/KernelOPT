import torch
import triton
import triton.language as tl

# Fused BatchNorm (inference) + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Determine channel index for each element
    # Layout: (N, C, H, W) -> channel = (offset / spatial_size) % num_channels
    spatial_idx = offsets % spatial_size
    channel_idx = (offsets // spatial_size) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std
    x_scaled = x_norm * weight + bias
    
    # ReLU
    result = tl.maximum(x_scaled, 0.0)
    
    tl.store(out_ptr + offsets, result, mask=mask)


# Fused BatchNorm (inference) kernel (no ReLU)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    channel_idx = (offsets // spatial_size) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std
    result = x_norm * weight + bias
    
    tl.store(out_ptr + offsets, result, mask=mask)


# Channel shuffle kernel
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
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    N, C, H, W, groups,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Output index: (n, c_out, h, w)
    # c_out = (c_in % groups) * channels_per_group + c_in // groups
    # We need to compute input index from output index
    
    w_idx = offsets % W
    h_idx = (offsets // W) % H
    c_out = (offsets // (W * H)) % C
    n_idx = offsets // (W * H * C)
    
    channels_per_group = C // groups
    # c_out = (c_in % groups) * channels_per_group + c_in // groups
    # Inverse: given c_out, find c_in
    # group_id = c_out // channels_per_group
    # within_group = c_out % channels_per_group
    # c_in = within_group * groups + group_id
    group_id = c_out // channels_per_group
    within_group = c_out % channels_per_group
    c_in = within_group * groups + group_id
    
    in_offset = w_idx + W * h_idx + W * H * c_in + W * H * C * n_idx
    
    val = tl.load(x_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


def apply_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    """Apply batch norm + relu using Triton kernel."""
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial_size = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, num_channels, spatial_size,
        eps,
    )
    return out


def apply_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    """Apply batch norm using Triton kernel."""
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial_size = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, num_channels, spatial_size,
        eps,
    )
    return out


def channel_shuffle(x, groups):
    """Apply channel shuffle using Triton kernel."""
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
        x, out,
        N, C, H, W, groups,
        numel,
    )
    return out


def conv2d_op(x, weight, stride=1, padding=0, groups=1):
    """Standard conv2d using PyTorch (cuDNN)."""
    import torch.nn.functional as F
    return F.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


def shufflenet_unit(x, 
                    conv1_w, bn1_w, bn1_b, bn1_rm, bn1_rv,
                    conv2_w, bn2_w, bn2_b, bn2_rm, bn2_rv,
                    conv3_w, bn3_w, bn3_b, bn3_rm, bn3_rv,
                    shortcut_conv_w=None, shortcut_bn_w=None, shortcut_bn_b=None,
                    shortcut_bn_rm=None, shortcut_bn_rv=None,
                    groups=3):
    """ShuffleNet unit forward pass."""
    import torch.nn.functional as F
    
    # Shortcut
    if shortcut_conv_w is not None:
        shortcut = conv2d_op(x, shortcut_conv_w, stride=1, padding=0, groups=1)
        shortcut = apply_bn(shortcut, shortcut_bn_w, shortcut_bn_b, shortcut_bn_rm, shortcut_bn_rv)
    else:
        shortcut = x
    
    # Conv1: 1x1 group conv
    out = conv2d_op(x, conv1_w, stride=1, padding=0, groups=groups)
    out = apply_bn_relu(out, bn1_w, bn1_b, bn1_rm, bn1_rv)
    
    # Conv2: depthwise 3x3
    mid_channels = out.shape[1]
    out = conv2d_op(out, conv2_w, stride=1, padding=1, groups=mid_channels)
    out = apply_bn(out, bn2_w, bn2_b, bn2_rm, bn2_rv)
    
    # Channel shuffle
    out = channel_shuffle(out, groups)
    
    # Conv3: 1x1 group conv
    out = conv2d_op(out, conv3_w, stride=1, padding=0, groups=groups)
    out = apply_bn_relu(out, bn3_w, bn3_b, bn3_rm, bn3_rv)
    
    # Add shortcut
    out = out + shortcut
    
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
    out = conv2d_op(x, p_conv1_weight, stride=2, padding=1, groups=1)
    out = apply_bn_relu(out, p_bn1_weight, p_bn1_bias, b_bn1_running_mean, b_bn1_running_var)
    out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)
    
    # Stage 2
    # Block 0 (with shortcut conv)
    out = shufflenet_unit(
        out,
        p_stage2_0_conv1_weight, p_stage2_0_bn1_weight, p_stage2_0_bn1_bias,
        b_stage2_0_bn1_running_mean, b_stage2_0_bn1_running_var,
        p_stage2_0_conv2_weight, p_stage2_0_bn2_weight, p_stage2_0_bn2_bias,
        b_stage2_0_bn2_running_mean, b_stage2_0_bn2_running_var,
        p_stage2_0_conv3_weight, p_stage2_0_bn3_weight, p_stage2_0_bn3_bias,
        b_stage2_0_bn3_running_mean, b_stage2_0_bn3_running_var,
        p_stage2_0_shortcut_0_weight, p_stage2_0_shortcut_1_weight, p_stage2_0_shortcut_1_bias,
        b_stage2_0_shortcut_1_running_mean, b_stage2_0_shortcut_1_running_var,
        groups=3,
    )
    
    # Block 1 (no shortcut conv)
    out = shufflenet_unit(
        out,
        p_stage2_1_conv1_weight, p_stage2_1_bn1_weight, p_stage2_1_bn1_bias,
        b_stage2_1_bn1_running_mean, b_stage2_1_bn1_running_var,
        p_stage2_1_conv2_weight, p_stage2_1_bn2_weight, p_stage2_1_bn2_bias,
        b_stage2_1_bn2_running_mean, b_stage2_1_bn2_running_var,
        p_stage2_1_conv3_weight, p_stage2_1_bn3_weight, p_stage2_1_bn3_bias,
        b_stage2_1_bn3_running_mean, b_stage2_1_bn3_running_var,
        groups=3,
    )
    
    # Block 2 (no shortcut conv)
    out = shufflenet_unit(
        out,
        p_stage2_2_conv1_weight, p_stage2_2_bn1_weight, p_stage2_2_bn1_bias,
        b_stage2_2_bn1_running_mean, b_stage2_2_bn1_running_var,
        p_stage2_2_conv2_weight, p_stage2_2_bn2_weight, p_stage2_2_bn2_bias,
        b_stage2_2_bn2_running_mean, b_stage2_2_bn2_running_var,
        p_stage2_2_conv3_weight, p_stage2_2_bn3_weight, p_stage2_2_bn3_bias,
        b_stage2_2_bn3_running_mean, b_stage2_2_bn3_running_var,
        groups=3,
    )
    
    # Stage 3
    # Block 0 (with shortcut conv)
    out = shufflenet_unit(
        out,
        p_stage3_0_conv1_weight, p_stage3_0_bn1_weight, p_stage3_0_bn1_bias,
        b_stage3_0_bn1_running_mean, b_stage3_0_bn1_running_var,
        p_stage3_0_conv2_weight, p_stage3_0_bn2_weight, p_stage3_0_bn2_bias,
        b_stage3_0_bn2_running_mean, b_stage3_0_bn2_running_var,
        p_stage3_0_conv3_weight, p_stage3_0_bn3_weight, p_stage3_0_bn3_bias,
        b_stage3_0_bn3_running_mean, b_stage3_0_bn3_running_var,
        p_stage3_0_shortcut_0_weight, p_stage3_0_shortcut_1_weight, p_stage3_0_shortcut_1_bias,
        b_stage3_0_shortcut_1_running_mean, b_stage3_0_shortcut_1_running_var,
        groups=3,
    )
    
    # Blocks 1-6 (no shortcut conv)
    out = shufflenet_unit(
        out,
        p_stage3_1_conv1_weight, p_stage3_1_bn1_weight, p_stage3_1_bn1_bias,
        b_stage3_1_bn1_running_mean, b_stage3_1_bn1_running_var,
        p_stage3_1_conv2_weight, p_stage3_1_bn2_weight, p_stage3_1_bn2_bias,
        b_stage3_1_bn2_running_mean, b_stage3_1_bn2_running_var,
        p_stage3_1_conv3_weight, p_stage3_1_bn3_weight, p_stage3_1_bn3_bias,
        b_stage3_1_bn3_running_mean, b_stage3_1_bn3_running_var,
        groups=3,
    )
    
    out = shufflenet_unit(
        out,
        p_stage3_2_conv1_weight, p_stage3_2_bn1_weight, p_stage3_2_bn1_bias,
        b_stage3_2_bn1_running_mean, b_stage3_2_bn1_running_var,
        p_stage3_2_conv2_weight, p_stage3_2_bn2_weight, p_stage3_2_bn2_bias,
        b_stage3_2_bn2_running_mean, b_stage3_2_bn2_running_var,
        p_stage3_2_conv3_weight, p_stage3_2_bn3_weight, p_stage3_2_bn3_bias,
        b_stage3_2_bn3_running_mean, b_stage3_2_bn3_running_var,
        groups=3,
    )
    
    out = shufflenet_unit(
        out,
        p_stage3_3_conv1_weight, p_stage3_3_bn1_weight, p_stage3_3_bn1_bias,
        b_stage3_3_bn1_running_mean, b_stage3_3_bn1_running_var,
        p_stage3_3_conv2_weight, p_stage3_3_bn2_weight, p_stage3_3_bn2_bias,
        b_stage3_3_bn2_running_mean, b_stage3_3_bn2_running_var,
        p_stage3_3_conv3_weight, p_stage3_3_bn3_weight, p_stage3_3_bn3_bias,
        b_stage3_3_bn3_running_mean, b_stage3_3_bn3_running_var,
        groups=3,
    )
    
    out = shufflenet_unit(
        out,
        p_stage3_4_conv1_weight, p_stage3_4_bn1_weight, p_stage3_4_bn1_bias,
        b_stage3_4_bn1_running_mean, b_stage3_4_bn1_running_var,
        p_stage3_4_conv2_weight, p_stage3_4_bn2_weight, p_stage3_4_bn2_bias,
        b_stage3_4_bn2_running_mean, b_stage3_4_bn2_running_var,
        p_stage3_4_conv3_weight, p_stage3_4_bn3_weight, p_stage3_4_bn3_bias,
        b_stage3_4_bn3_running_mean, b_stage3_4_bn3_running_var,
        groups=3,
    )
    
    out = shufflenet_unit(
        out,
        p_stage3_5_conv1_weight, p_stage3_5_bn1_weight, p_stage3_5_bn1_bias,