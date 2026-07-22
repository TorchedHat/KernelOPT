import torch
import triton
import triton.language as tl


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
    ACTIVATION: tl.constexpr,  # 0=none, 1=relu, 2=hardtanh
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
    
    if ACTIVATION == 1:  # ReLU
        out = tl.maximum(out, 0.0)
    elif ACTIVATION == 2:  # HardTanh [-6, 6]
        out = tl.minimum(tl.maximum(out, -6.0), 6.0)
    
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
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b + res
    
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


def fused_bn_act(x, mean, var, weight, bias, activation=0, eps=1e-5):
    """Fused BatchNorm + optional activation"""
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, C, spatial,
        eps,
        ACTIVATION=activation,
    )
    return out


def fused_bn_add(x, residual, mean, var, weight, bias, eps=1e-5):
    """Fused BatchNorm + residual add"""
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_kernel[grid](
        x, residual, out,
        mean, var, weight, bias,
        numel, C, spatial,
        eps,
    )
    return out


def conv2d_op(x, weight, stride=1, padding=0, groups=1):
    return F.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


def mbconv(x, expand_w, expand_bn_w, expand_bn_b, expand_mean, expand_var,
           dw_w, dw_bn_w, dw_bn_b, dw_mean, dw_var,
           proj_w, proj_bn_w, proj_bn_b, proj_mean, proj_var,
           stride, expand_ratio, use_residual, kernel_size=3):
    """MBConv block"""
    residual = x
    in_channels = x.shape[1]
    
    if expand_ratio != 1:
        # Expand conv
        x = conv2d_op(x, expand_w, stride=1, padding=0)
        x = fused_bn_act(x, expand_mean, expand_var, expand_bn_w, expand_bn_b, activation=2)
    
    # Depthwise conv
    pad = kernel_size // 2
    groups = x.shape[1]
    x = conv2d_op(x, dw_w, stride=stride, padding=pad, groups=groups)
    x = fused_bn_act(x, dw_mean, dw_var, dw_bn_w, dw_bn_b, activation=2)
    
    # Project conv
    x = conv2d_op(x, proj_w, stride=1, padding=0)
    
    if use_residual:
        x = fused_bn_add(x, residual, proj_mean, proj_var, proj_bn_w, proj_bn_b)
    else:
        x = fused_bn_act(x, proj_mean, proj_var, proj_bn_w, proj_bn_b, activation=0)
    
    return x


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
    x = conv2d_op(x, p_conv1_weight, stride=2, padding=1)
    x = fused_bn_act(x, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias, activation=1)
    
    # Block 0: MBConv1 (32->16, stride=1, expand=1, kernel=3)
    # No expand conv since expand_ratio=1
    x_dw = conv2d_op(x, p_blocks_0_depthwise_conv_0_weight, stride=1, padding=1, groups=32)
    x_dw = fused_bn_act(x_dw, b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
                        p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias, activation=2)
    x_proj = conv2d_op(x_dw, p_blocks_0_project_conv_0_weight, stride=1, padding=0)
    x = fused_bn_act(x_proj, b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
                     p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias, activation=0)
    
    # Block 1: MBConv6 (16->24, stride=2, expand=6, kernel=3)
    x_exp = conv2d_op(x, p_blocks_1_expand_conv_0_weight, stride=1, padding=0)
    x_exp = fused_bn_act(x_exp, b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
                         p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias, activation=2)
    x_dw = conv2d_op(x_exp, p_blocks_1_depthwise_conv_0_weight, stride=2, padding=1, groups=96)
    x_dw = fused_bn_act(x_dw, b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
                        p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias, activation=2)
    x_proj = conv2d_op(x_dw, p_blocks_1_project_conv_0_weight, stride=1, padding=0)
    x = fused_bn_act(x_proj, b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
                     p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias, activation=0)
    
    # Block 2: MBConv6 (24->24, stride=1, expand=6, kernel=3) - has residual
    residual = x
    x_exp = conv2d_op(x, p_blocks_2_expand_conv_0_weight, stride=1, padding=0)
    x_exp = fused_bn_act(x_exp, b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
                         p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias, activation=2)
    x_dw = conv2d_op(x_exp, p_blocks_2_depthwise_conv_0_weight, stride=1, padding=1, groups=144)
    x_dw = fused_bn_act(x_dw, b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
                        p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias, activation=2)
    x_proj = conv2d_op(x_dw, p_blocks_2_project_conv_0_weight, stride=1, padding=0)
    x = fused_bn_add(x_proj, residual, b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
    
    # Block 3: MBConv6 (24->40, stride=2, expand=6, kernel=5)
    x_exp = conv2d_op(x, p_blocks_3_expand_conv_0_weight, stride=1, padding=0)
    x_exp = fused_bn_act(x_exp, b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
                         p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias, activation=2)
    x_dw = conv2d_op(x_exp, p_blocks_3_depthwise_conv_0_weight, stride=2, padding=2, groups=144)
    x_dw = fused_bn_act(x_dw, b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
                        p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias, activation=2)
    x_proj = conv2d_op(x_dw, p_blocks_3_project_conv_0_weight, stride=1, padding=0)
    x = fused_bn_act(x_proj, b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
                     p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias, activation=0)
    
    # Block 4: MBConv6 (40->40, stride=1, expand=6, kernel=5) - has residual
    residual = x
    x_exp = conv2d_op(x, p_blocks_4_expand_conv_0_weight, stride=1, padding=0)
    x_exp = fused_bn_act(x_exp, b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
                         p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias, activation=2)
    x_dw = conv2d_op(x_exp, p_blocks_4_depthwise_conv_0_weight, stride=1, padding=2, groups=240)
    x_dw = fused_bn_act(x_dw, b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var,
                        p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias, activation=2)
    x_proj = conv2d_op(x_dw, p_blocks_4_project_conv_0_weight, stride=1, padding=0)
    x = fused_bn_add(x_proj, residual, b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var,
    
    # Block 5: MBConv6 (40->80, stride=2, expand=6, kernel=3)
    x_exp = conv2d_op(x, p_blocks_5_expand_conv_0_weight, stride=1, padding=0)
    x_exp = fused_bn_act(x_exp, b_blocks_5_expand_conv_1_running_mean, b_blocks_5_expand_conv_1_running_var,
                         p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias, activation=2)
    x_dw = conv2d_op(x_exp, p_blocks_5_depthwise_conv_0_weight, stride=2, padding=1, groups=240)
    x_dw = fused_bn_act(x_dw, b_blocks_5_depthwise_conv_1_running_mean, b_blocks_5_depthwise_conv_1_running_var,