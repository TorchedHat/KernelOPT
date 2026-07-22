import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm + ReLU kernel
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
    ACTIVATION: tl.constexpr,  # 0=none, 1=relu, 2=hardtanh6
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
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b
    
    # Activation
    if ACTIVATION == 1:  # ReLU
        out = tl.maximum(out, 0.0)
    elif ACTIVATION == 2:  # HardTanh (clamp to [-6, 6] for ReLU6)
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Fused BatchNorm + Add + (no activation) kernel
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
def bn_add_kernel(
    x_ptr, residual_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b + res
    
    tl.store(out_ptr + offsets, out, mask=mask)


def apply_bn_act(x, weight, bias, running_mean, running_var, activation=0, eps=1e-5):
    """Apply BatchNorm + optional activation using Triton kernel."""
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial,
        eps,
        ACTIVATION=activation,
    )
    return out


def apply_bn_add(x, residual, weight, bias, running_mean, running_var, eps=1e-5):
    """Apply BatchNorm + Add residual using Triton kernel."""
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_kernel[grid](
        x, residual, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial,
        eps,
    )
    return out


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_blocks_0_depthwise_conv_0_weight, p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
    p_blocks_0_project_conv_0_weight, p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
    p_blocks_1_expand_conv_0_weight, p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
    p_blocks_1_depthwise_conv_0_weight, p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
    p_blocks_1_project_conv_0_weight, p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
    p_blocks_2_expand_conv_0_weight, p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
    p_blocks_2_depthwise_conv_0_weight, p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
    p_blocks_2_project_conv_0_weight, p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
    p_blocks_3_expand_conv_0_weight, p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
    p_blocks_3_depthwise_conv_0_weight, p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
    p_blocks_3_project_conv_0_weight, p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
    p_blocks_4_expand_conv_0_weight, p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
    p_blocks_4_depthwise_conv_0_weight, p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
    p_blocks_4_project_conv_0_weight, p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,
    p_blocks_5_expand_conv_0_weight, p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias,
    p_blocks_5_depthwise_conv_0_weight, p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias,
    p_blocks_5_project_conv_0_weight, p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_1_bias,
    p_blocks_6_expand_conv_0_weight, p_blocks_6_expand_conv_1_weight, p_blocks_6_expand_conv_1_bias,
    p_blocks_6_depthwise_conv_0_weight, p_blocks_6_depthwise_conv_1_weight, p_blocks_6_depthwise_conv_1_bias,
    p_blocks_6_project_conv_0_weight, p_blocks_6_project_conv_1_weight, p_blocks_6_project_conv_1_bias,
    p_blocks_7_expand_conv_0_weight, p_blocks_7_expand_conv_1_weight, p_blocks_7_expand_conv_1_bias,
    p_blocks_7_depthwise_conv_0_weight, p_blocks_7_depthwise_conv_1_weight, p_blocks_7_depthwise_conv_1_bias,
    p_blocks_7_project_conv_0_weight, p_blocks_7_project_conv_1_weight, p_blocks_7_project_conv_1_bias,
    p_blocks_8_expand_conv_0_weight, p_blocks_8_expand_conv_1_weight, p_blocks_8_expand_conv_1_bias,
    p_blocks_8_depthwise_conv_0_weight, p_blocks_8_depthwise_conv_1_weight, p_blocks_8_depthwise_conv_1_bias,
    p_blocks_8_project_conv_0_weight, p_blocks_8_project_conv_1_weight, p_blocks_8_project_conv_1_bias,
    p_blocks_9_expand_conv_0_weight, p_blocks_9_expand_conv_1_weight, p_blocks_9_expand_conv_1_bias,
    p_blocks_9_depthwise_conv_0_weight, p_blocks_9_depthwise_conv_1_weight, p_blocks_9_depthwise_conv_1_bias,
    p_blocks_9_project_conv_0_weight, p_blocks_9_project_conv_1_weight, p_blocks_9_project_conv_1_bias,
    p_blocks_10_expand_conv_0_weight, p_blocks_10_expand_conv_1_weight, p_blocks_10_expand_conv_1_bias,
    p_blocks_10_depthwise_conv_0_weight, p_blocks_10_depthwise_conv_1_weight, p_blocks_10_depthwise_conv_1_bias,
    p_blocks_10_project_conv_0_weight, p_blocks_10_project_conv_1_weight, p_blocks_10_project_conv_1_bias,
    p_blocks_11_expand_conv_0_weight, p_blocks_11_expand_conv_1_weight, p_blocks_11_expand_conv_1_bias,
    p_blocks_11_depthwise_conv_0_weight, p_blocks_11_depthwise_conv_1_weight, p_blocks_11_depthwise_conv_1_bias,
    p_blocks_11_project_conv_0_weight, p_blocks_11_project_conv_1_weight, p_blocks_11_project_conv_1_bias,
    p_blocks_12_expand_conv_0_weight, p_blocks_12_expand_conv_1_weight, p_blocks_12_expand_conv_1_bias,
    p_blocks_12_depthwise_conv_0_weight, p_blocks_12_depthwise_conv_1_weight, p_blocks_12_depthwise_conv_1_bias,
    p_blocks_12_project_conv_0_weight, p_blocks_12_project_conv_1_weight, p_blocks_12_project_conv_1_bias,
    p_conv2_weight, p_bn2_weight, p_bn2_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
    b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
    b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
    b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
    b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
    b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
    b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
    b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
    b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
    b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
    b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
    b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
    b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var,
    b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var,
    b_blocks_5_expand_conv_1_running_mean, b_blocks_5_expand_conv_1_running_var,
    b_blocks_5_depthwise_conv_1_running_mean, b_blocks_5_depthwise_conv_1_running_var,
    b_blocks_5_project_conv_1_running_mean, b_blocks_5_project_conv_1_running_var,
    b_blocks_6_expand_conv_1_running_mean, b_blocks_6_expand_conv_1_running_var,
    b_blocks_6_depthwise_conv_1_running_mean, b_blocks_6_depthwise_conv_1_running_var,
    b_blocks_6_project_conv_1_running_mean, b_blocks_6_project_conv_1_running_var,
    b_blocks_7_expand_conv_1_running_mean, b_blocks_7_expand_conv_1_running_var,
    b_blocks_7_depthwise_conv_1_running_mean, b_blocks_7_depthwise_conv_1_running_var,
    b_blocks_7_project_conv_1_running_mean, b_blocks_7_project_conv_1_running_var,
    b_blocks_8_expand_conv_1_running_mean, b_blocks_8_expand_conv_1_running_var,
    b_blocks_8_depthwise_conv_1_running_mean, b_blocks_8_depthwise_conv_1_running_var,
    b_blocks_8_project_conv_1_running_mean, b_blocks_8_project_conv_1_running_var,
    b_blocks_9_expand_conv_1_running_mean, b_blocks_9_expand_conv_1_running_var,
    b_blocks_9_depthwise_conv_1_running_mean, b_blocks_9_depthwise_conv_1_running_var,
    b_blocks_9_project_conv_1_running_mean, b_blocks_9_project_conv_1_running_var,
    b_blocks_10_expand_conv_1_running_mean, b_blocks_10_expand_conv_1_running_var,
    b_blocks_10_depthwise_conv_1_running_mean, b_blocks_10_depthwise_conv_1_running_var,
    b_blocks_10_project_conv_1_running_mean, b_blocks_10_project_conv_1_running_var,
    b_blocks_11_expand_conv_1_running_mean, b_blocks_11_expand_conv_1_running_var,
    b_blocks_11_depthwise_conv_1_running_mean, b_blocks_11_depthwise_conv_1_running_var,
    b_blocks_11_project_conv_1_running_mean, b_blocks_11_project_conv_1_running_var,
    b_blocks_12_expand_conv_1_running_mean, b_blocks_12_expand_conv_1_running_var,
    b_blocks_12_depthwise_conv_1_running_mean, b_blocks_12_depthwise_conv_1_running_var,
    b_blocks_12_project_conv_1_running_mean, b_blocks_12_project_conv_1_running_var,
    b_bn2_running_mean, b_bn2_running_var,
    x,
):
    # Initial conv + BN + ReLU
    x = F.conv2d(x, p_conv1_weight, None, stride=2, padding=1)
    x = apply_bn_act(x, p_bn1_weight, p_bn1_bias, b_bn1_running_mean, b_bn1_running_var, activation=1)

    # Block 0: MBConv1 (32->16, stride=1, expand=1)
    # depthwise conv (groups=32)
    t = F.conv2d(x, p_blocks_0_depthwise_conv_0_weight, None, stride=1, padding=1, groups=32)
    t = apply_bn_act(t, p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
                     b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_0_project_conv_0_weight, None, stride=1, padding=0)
    x = apply_bn_act(t, p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
                     b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var, activation=0)

    # Block 1: MBConv6 (16->24, stride=2, expand=6)
    t = F.conv2d(x, p_blocks_1_expand_conv_0_weight, None, stride=1, padding=0)
    t = apply_bn_act(t, p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
                     b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_1_depthwise_conv_0_weight, None, stride=2, padding=1, groups=96)
    t = apply_bn_act(t, p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
                     b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_1_project_conv_0_weight, None, stride=1, padding=0)
    x = apply_bn_act(t, p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
                     b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var, activation=0)

    # Block 2: MBConv6 (24->24, stride=1, expand=6) - has residual
    residual = x
    t = F.conv2d(x, p_blocks_2_expand_conv_0_weight, None, stride=1, padding=0)
    t = apply_bn_act(t, p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
                     b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_2_depthwise_conv_0_weight, None, stride=1, padding=1, groups=144)
    t = apply_bn_act(t, p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
                     b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_2_project_conv_0_weight, None, stride=1, padding=0)
    x = apply_bn_add(t, residual, p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,

    # Block 3: MBConv6 (24->40, stride=2, expand=6, kernel=5)
    t = F.conv2d(x, p_blocks_3_expand_conv_0_weight, None, stride=1, padding=0)
    t = apply_bn_act(t, p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
                     b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_3_depthwise_conv_0_weight, None, stride=2, padding=2, groups=144)
    t = apply_bn_act(t, p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
                     b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_3_project_conv_0_weight, None, stride=1, padding=0)
    x = apply_bn_act(t, p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
                     b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var, activation=0)

    # Block 4: MBConv6 (40->40, stride=1, expand=6, kernel=5) - has residual
    residual = x
    t = F.conv2d(x, p_blocks_4_expand_conv_0_weight, None, stride=1, padding=0)
    t = apply_bn_act(t, p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
                     b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_4_depthwise_conv_0_weight, None, stride=1, padding=2, groups=240)
    t = apply_bn_act(t, p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
                     b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_4_project_conv_0_weight, None, stride=1, padding=0)
    x = apply_bn_add(t, residual, p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,

    # Block 5: MBConv6 (40->80, stride=2, expand=6, kernel=3)
    t = F.conv2d(x, p_blocks_5_expand_conv_0_weight, None, stride=1, padding=0)
    t = apply_bn_act(t, p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias,
                     b_blocks_5_expand_conv_1_running_mean, b_blocks_5_expand_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_5_depthwise_conv_0_weight, None, stride=2, padding=1, groups=240)
    t = apply_bn_act(t, p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias,
                     b_blocks_5_depthwise_conv_1_running_mean, b_blocks_5_depthwise_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_5_project_conv_0_weight, None, stride=1, padding=0)
    x = apply_bn_act(t, p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_1_bias,
                     b_blocks_5_project_conv_1_running_mean, b_blocks_5_project_conv_1_running_var, activation=0)

    # Block 6: MBConv6 (80->80, stride=1, expand=6, kernel=3) - has residual
    residual = x
    t = F.conv2d(x, p_blocks_6_expand_conv_0_weight, None, stride=1, padding=0)
    t = apply_bn_act(t, p_blocks_6_expand_conv_1_weight, p_blocks_6_expand_conv_1_bias,
                     b_blocks_6_expand_conv_1_running_mean, b_blocks_6_expand_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_6_depthwise_conv_0_weight, None, stride=1, padding=1, groups=480)
    t = apply_bn_act(t, p_blocks_6_depthwise_conv_1_weight, p_blocks_6_depthwise_conv_1_bias,
                     b_blocks_6_depthwise_conv_1_running_mean, b_blocks_6_depthwise_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_6_project_conv_0_weight, None, stride=1, padding=0)
    x = apply_bn_add(t, residual, p_blocks_6_project_conv_1_weight, p_blocks_6_project_conv_1_bias,

    # Block 7: MBConv6 (80->112, stride=1, expand=6, kernel=5)
    t = F.conv2d(x, p_blocks_7_expand_conv_0_weight, None, stride=1, padding=0)
    t = apply_bn_act(t, p_blocks_7_expand_conv_1_weight, p_blocks_7_expand_conv_1_bias,
                     b_blocks_7_expand_conv_1_running_mean, b_blocks_7_expand_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_7_depthwise_conv_0_weight, None, stride=1, padding=2, groups=480)
    t = apply_bn_act(t, p_blocks_7_depthwise_conv_1_weight, p_blocks_7_depthwise_conv_1_bias,
                     b_blocks_7_depthwise_conv_1_running_mean, b_blocks_7_depthwise_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_7_project_conv_0_weight, None, stride=1, padding=0)
    x = apply_bn_act(t, p_blocks_7_project_conv_1_weight, p_blocks_7_project_conv_1_bias,
                     b_blocks_7_project_conv_1_running_mean, b_blocks_7_project_conv_1_running_var, activation=0)

    # Block 8: MBConv6 (112->112, stride=1, expand=6, kernel=5) - has residual
    residual = x
    t = F.conv2d(x, p_blocks_8_expand_conv_0_weight, None, stride=1, padding=0)
    t = apply_bn_act(t, p_blocks_8_expand_conv_1_weight, p_blocks_8_expand_conv_1_bias,
                     b_blocks_8_expand_conv_1_running_mean, b_blocks_8_expand_conv_1_running_var, activation=2)
    t = F.conv2d(t, p_blocks_8_depthwise_conv_0_weight, None, stride=1, padding=2, groups=672)
    t = apply_bn_act(t, p_