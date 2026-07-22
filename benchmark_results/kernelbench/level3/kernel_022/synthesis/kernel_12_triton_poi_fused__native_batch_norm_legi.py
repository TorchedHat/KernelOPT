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
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_channels', 'numel'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, numel,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Channel index: layout is (N, C, H, W), channel = (offset / (H*W)) % C
    # But we pass numel = N*C*H*W, and channel stride = H*W
    # We need to figure out channel from offset
    # offset = n * C * HW + c * HW + hw
    # c = (offset // HW) % C ... but we don't know HW here
    # Instead, pass spatial_size = H*W
    # Actually let's use a different approach: pass channel stride
    # offset // channel_stride % n_channels
    # We'll pass hw_size (H*W) as a parameter
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute channel index
    # This kernel assumes contiguous NCHW layout
    # channel = (offset // hw_size) % n_channels
    # We'll embed hw_size computation differently
    # For simplicity, pass channel index via offset % n_channels won't work for NCHW
    # Let's restructure: pass hw_size
    pass

# Better approach: separate kernel with hw_size parameter
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
def fused_bn_act_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    hw_size, n_channels, numel,
    eps,
    act_type,  # 0=relu, 1=hardtanh(-6,6), 2=none
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # channel index for NCHW: c = (offset // hw_size) % n_channels
    c_idx = (offsets // hw_size) % n_channels
    
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_out = x_norm * w + b
    
    # Activation
    if act_type == 0:  # ReLU
        x_out = tl.maximum(x_out, 0.0)
    elif act_type == 1:  # HardTanh(-6, 6) = ReLU6
        x_out = tl.minimum(tl.maximum(x_out, -6.0), 6.0)
    
    tl.store(out_ptr + offsets, x_out, mask=mask)


def apply_bn_act(x, weight, bias, running_mean, running_var, eps=1e-5, act_type=0):
    """Apply BatchNorm + activation using Triton kernel."""
    N, C, H, W = x.shape
    numel = N * C * H * W
    hw_size = H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_act_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        hw_size, C, numel,
        eps,
        act_type,
    )
    return out


def conv2d_bn_act(x, conv_weight, bn_weight, bn_bias, bn_mean, bn_var,
                  stride=1, padding=0, groups=1, dilation=1, act_type=0, eps=1e-5):
    """Conv2d + BN + activation."""
    # Use PyTorch for conv (cuDNN is faster)
    out = F.conv2d(x, conv_weight, bias=None, stride=stride, padding=padding, 
                   groups=groups, dilation=dilation)
    out = apply_bn_act(out, bn_weight, bn_bias, bn_mean, bn_var, eps=eps, act_type=act_type)
    return out


def conv2d_bn(x, conv_weight, bn_weight, bn_bias, bn_mean, bn_var,
              stride=1, padding=0, groups=1, dilation=1, eps=1e-5):
    """Conv2d + BN (no activation)."""
    out = F.conv2d(x, conv_weight, bias=None, stride=stride, padding=padding,
                   groups=groups, dilation=dilation)
    out = apply_bn_act(out, bn_weight, bn_bias, bn_mean, bn_var, eps=eps, act_type=2)
    return out


def mbconv_block(x, params, block_idx):
    """MBConv block forward pass."""
    p = params[block_idx]
    in_c = x.shape[1]
    
    if p['expand_ratio'] == 1:
        # No expansion conv
        # Depthwise conv
        dw = conv2d_bn_act(
            x,
            p['dw_weight'], p['dw_bn_w'], p['dw_bn_b'], p['dw_bn_mean'], p['dw_bn_var'],
            stride=p['stride'], padding=p['dw_padding'], groups=in_c, act_type=1
        )
        # Project conv (no activation)
        out = conv2d_bn(
            dw,
            p['proj_weight'], p['proj_bn_w'], p['proj_bn_b'], p['proj_bn_mean'], p['proj_bn_var'],
        )
        # No residual for expand_ratio=1 with stride=1 (but block 0 has stride=1, in=32, out=16, no skip)
        return out
    else:
        # Expand conv
        expanded = conv2d_bn_act(
            x,
            p['exp_weight'], p['exp_bn_w'], p['exp_bn_b'], p['exp_bn_mean'], p['exp_bn_var'],
            act_type=1
        )
        # Depthwise conv
        dw = conv2d_bn_act(
            expanded,
            p['dw_weight'], p['dw_bn_w'], p['dw_bn_b'], p['dw_bn_mean'], p['dw_bn_var'],
            stride=p['stride'], padding=p['dw_padding'], groups=expanded.shape[1], act_type=1
        )
        # Project conv (no activation)
        out = conv2d_bn(
            dw,
            p['proj_weight'], p['proj_bn_w'], p['proj_bn_b'], p['proj_bn_mean'], p['proj_bn_var'],
        )
        # Residual if stride=1 and in_c == out_c
        if p['stride'] == 1 and in_c == out.shape[1]:
            out = out + x
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
    x = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    x = apply_bn_act(x, p_bn1_weight, p_bn1_bias, b_bn1_running_mean, b_bn1_running_var, act_type=0)
    
    # Block 0: MBConv1 (32->16, stride=1, expand=1, kernel=3)
    # No expand conv (expand_ratio=1)
    # Depthwise conv (groups=32, stride=1, padding=1)
    dw = F.conv2d(x, p_blocks_0_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=32)
    dw = apply_bn_act(dw, p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
                      b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var, act_type=1)
    # Project conv (no activation)
    x = F.conv2d(dw, p_blocks_0_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = apply_bn_act(x, p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
                     b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var, act_type=2)
    # No residual (in_c=32 != out_c=16)
    
    # Block 1: MBConv6 (16->24, stride=2, expand=6, kernel=3)
    # Expand conv
    exp = F.conv2d(x, p_blocks_1_expand_conv_0_weight, bias=None, stride=1, padding=0)
    exp = apply_bn_act(exp, p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
                       b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var, act_type=1)
    # Depthwise conv (groups=96, stride=2, padding=1)
    dw = F.conv2d(exp, p_blocks_1_depthwise_conv_0_weight, bias=None, stride=2, padding=1, groups=96)
    dw = apply_bn_act(dw, p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
                      b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var, act_type=1)
    # Project conv
    x = F.conv2d(dw, p_blocks_1_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = apply_bn_act(x, p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
                     b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var, act_type=2)
    # No residual (stride=2)
    
    # Block 2: MBConv6 (24->24, stride=1, expand=6, kernel=3)
    residual = x
    exp = F.conv2d(x, p_blocks_2_expand_conv_0_weight, bias=None, stride=1, padding=0)
    exp = apply_bn_act(exp, p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
                       b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var, act_type=1)
    dw = F.conv2d(exp, p_blocks_2_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=144)
    dw = apply_bn_act(dw, p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
                      b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var, act_type=1)
    x = F.conv2d(dw, p_blocks_2_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = apply_bn_act(x, p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
                     b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var, act_type=2)
    x = x + residual  # residual (24==24, stride=1)
    
    # Block 3: MBConv6 (24->40, stride=2, expand=6, kernel=5)
    exp = F.conv2d(x, p_blocks_3_expand_conv_0_weight, bias=None, stride=1, padding=0)
    exp = apply_bn_act(exp, p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
                       b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var, act_type=1)
    dw = F.conv2d(exp, p_blocks_3_depthwise_conv_0_weight, bias=None, stride=2, padding=2, groups=144)
    dw = apply_bn_act(dw, p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
                      b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var, act_type=1)
    x = F.conv2d(dw, p_blocks_3_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = apply_bn_act(x, p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
                     b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var, act_type=2)
    # No residual (stride=2)
    
    # Block 4: MBConv6 (40->40, stride=1, expand=6, kernel=5)
    residual = x
    exp = F.conv2d(x, p_blocks_4_expand_conv_0_weight, bias=None, stride=1, padding=0)
    exp = apply_bn_act(exp, p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
                       b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var, act_type=1)
    dw = F.conv2d(exp, p_blocks_4_depthwise_conv_0_weight, bias=None, stride=1, padding=2, groups=240)
    dw = apply_bn_act(dw, p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
                      b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var, act_type=1)
    x = F.conv2d(dw, p_blocks_4_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = apply_bn_act(x, p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,
                     b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var, act_type=2)
    x = x + residual  # residual (40==40, stride=1)
    
    # Block 5: MBConv6 (40->80, stride=2, expand=6, kernel=3)
    exp = F.conv2d(x, p_blocks_5_expand_conv_0_weight, bias=None, stride=1, padding=0)
    exp = apply_bn_act(exp, p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias,
                       b_blocks_5_expand_conv_1_running_mean, b_blocks_5_expand_conv_1_running_var, act_type=1)
    dw = F.conv2d(exp, p_blocks_5_depthwise_conv_0_weight, bias=None, stride=2, padding=1, groups=240)
    dw = apply_bn_act(dw, p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias,
                      b_blocks_5_depthwise_conv_1_running_mean, b_blocks_5_depthwise_conv_1_running_var, act_type=1)
    x = F.conv2d(dw, p_blocks_5_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = apply_bn_act(x, p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_