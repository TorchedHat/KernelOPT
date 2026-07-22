import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm + Activation kernel
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
def bn_act_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, spatial,
    eps,
    act_type,  # 0=none, 1=relu, 2=hardtanh
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // spatial) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # BatchNorm
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_out = x_norm * w + b
    
    # Activation
    if act_type == 1:  # relu
        x_out = tl.maximum(x_out, 0.0)
    elif act_type == 2:  # hardtanh (clamp -6, 6 for relu6)
        x_out = tl.minimum(tl.maximum(x_out, -6.0), 6.0)
    
    tl.store(out_ptr + offsets, x_out, mask=mask)


def bn_act(x, mean, var, weight, bias, eps=1e-5, act_type=0):
    """Fused BatchNorm + Activation"""
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_act_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, C, spatial,
        eps,
        act_type,
    )
    return out


def conv2d_bn_act(x, conv_weight, mean, var, bn_weight, bn_bias,
                  stride=1, padding=0, groups=1, act_type=0, eps=1e-5):
    """Conv2d followed by fused BN+Act"""
    # Use cuDNN for conv
    x = F.conv2d(x, conv_weight, bias=None, stride=stride, padding=padding, groups=groups)
    x = bn_act(x, mean, var, bn_weight, bn_bias, eps=eps, act_type=act_type)
    return x


def kernel_function(
    p_conv1_weight,
    p_bn1_weight, p_bn1_bias,
    p_blocks_0_depthwise_conv_0_weight,
    p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
    p_blocks_0_project_conv_0_weight,
    p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
    p_blocks_1_expand_conv_0_weight,
    p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
    p_blocks_1_depthwise_conv_0_weight,
    p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
    p_blocks_1_project_conv_0_weight,
    p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
    p_blocks_2_expand_conv_0_weight,
    p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
    p_blocks_2_depthwise_conv_0_weight,
    p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
    p_blocks_2_project_conv_0_weight,
    p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
    p_blocks_3_expand_conv_0_weight,
    p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
    p_blocks_3_depthwise_conv_0_weight,
    p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
    p_blocks_3_project_conv_0_weight,
    p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
    p_blocks_4_expand_conv_0_weight,
    p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
    p_blocks_4_depthwise_conv_0_weight,
    p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
    p_blocks_4_project_conv_0_weight,
    p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,
    p_blocks_5_expand_conv_0_weight,
    p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias,
    p_blocks_5_depthwise_conv_0_weight,
    p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias,
    p_blocks_5_project_conv_0_weight,
    p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_1_bias,
    p_blocks_6_expand_conv_0_weight,
    p_blocks_6_expand_conv_1_weight, p_blocks_6_expand_conv_1_bias,
    p_blocks_6_depthwise_conv_0_weight,
    p_blocks_6_depthwise_conv_1_weight, p_blocks_6_depthwise_conv_1_bias,
    p_blocks_6_project_conv_0_weight,
    p_blocks_6_project_conv_1_weight, p_blocks_6_project_conv_1_bias,
    p_blocks_7_expand_conv_0_weight,
    p_blocks_7_expand_conv_1_weight, p_blocks_7_expand_conv_1_bias,
    p_blocks_7_depthwise_conv_0_weight,
    p_blocks_7_depthwise_conv_1_weight, p_blocks_7_depthwise_conv_1_bias,
    p_blocks_7_project_conv_0_weight,
    p_blocks_7_project_conv_1_weight, p_blocks_7_project_conv_1_bias,
    p_blocks_8_expand_conv_0_weight,
    p_blocks_8_expand_conv_1_weight, p_blocks_8_expand_conv_1_bias,
    p_blocks_8_depthwise_conv_0_weight,
    p_blocks_8_depthwise_conv_1_weight, p_blocks_8_depthwise_conv_1_bias,
    p_blocks_8_project_conv_0_weight,
    p_blocks_8_project_conv_1_weight, p_blocks_8_project_conv_1_bias,
    p_blocks_9_expand_conv_0_weight,
    p_blocks_9_expand_conv_1_weight, p_blocks_9_expand_conv_1_bias,
    p_blocks_9_depthwise_conv_0_weight,
    p_blocks_9_depthwise_conv_1_weight, p_blocks_9_depthwise_conv_1_bias,
    p_blocks_9_project_conv_0_weight,
    p_blocks_9_project_conv_1_weight, p_blocks_9_project_conv_1_bias,
    p_blocks_10_expand_conv_0_weight,
    p_blocks_10_expand_conv_1_weight, p_blocks_10_expand_conv_1_bias,
    p_blocks_10_depthwise_conv_0_weight,
    p_blocks_10_depthwise_conv_1_weight, p_blocks_10_depthwise_conv_1_bias,
    p_blocks_10_project_conv_0_weight,
    p_blocks_10_project_conv_1_weight, p_blocks_10_project_conv_1_bias,
    p_blocks_11_expand_conv_0_weight,
    p_blocks_11_expand_conv_1_weight, p_blocks_11_expand_conv_1_bias,
    p_blocks_11_depthwise_conv_0_weight,
    p_blocks_11_depthwise_conv_1_weight, p_blocks_11_depthwise_conv_1_bias,
    p_blocks_11_project_conv_0_weight,
    p_blocks_11_project_conv_1_weight, p_blocks_11_project_conv_1_bias,
    p_blocks_12_expand_conv_0_weight,
    p_blocks_12_expand_conv_1_weight, p_blocks_12_expand_conv_1_bias,
    p_blocks_12_depthwise_conv_0_weight,
    p_blocks_12_depthwise_conv_1_weight, p_blocks_12_depthwise_conv_1_bias,
    p_blocks_12_project_conv_0_weight,
    p_blocks_12_project_conv_1_weight, p_blocks_12_project_conv_1_bias,
    p_conv2_weight,
    p_bn2_weight, p_bn2_bias,
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
    # act_type: 0=none, 1=relu, 2=hardtanh(-6,6)
    
    # Initial conv + BN + ReLU
    x0 = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    x0 = bn_act(x0, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias, act_type=1)
    
    # Block 0: MBConv1 (32->16, k=3, s=1, expand=1)
    # depthwise conv (groups=32)
    x0 = F.conv2d(x0, p_blocks_0_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=32)
    x0 = bn_act(x0, b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
                p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias, act_type=2)
    # project conv
    x0 = F.conv2d(x0, p_blocks_0_project_conv_0_weight, bias=None, stride=1, padding=0)
    x0 = bn_act(x0, b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
                p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias, act_type=0)
    
    # Block 1: MBConv6 (16->24, k=3, s=2, expand=6)
    identity = x0
    # expand conv
    x1 = F.conv2d(x0, p_blocks_1_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x1 = bn_act(x1, b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
                p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias, act_type=2)
    # depthwise conv (groups=96)
    x1 = F.conv2d(x1, p_blocks_1_depthwise_conv_0_weight, bias=None, stride=2, padding=1, groups=96)
    x1 = bn_act(x1, b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
                p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias, act_type=2)
    # project conv
    x1 = F.conv2d(x1, p_blocks_1_project_conv_0_weight, bias=None, stride=1, padding=0)
    x1 = bn_act(x1, b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
                p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias, act_type=0)
    # no residual (stride=2, channels change)
    
    # Block 2: MBConv6 (24->24, k=3, s=1, expand=6)
    identity2 = x1
    x2 = F.conv2d(x1, p_blocks_2_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x2 = bn_act(x2, b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
                p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias, act_type=2)
    x2 = F.conv2d(x2, p_blocks_2_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=144)
    x2 = bn_act(x2, b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
                p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias, act_type=2)
    x2 = F.conv2d(x2, p_blocks_2_project_conv_0_weight, bias=None, stride=1, padding=0)
    x2 = bn_act(x2, b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
                p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias, act_type=0)
    x2 = x2 + identity2  # residual
    
    # Block 3: MBConv6 (24->40, k=5, s=2, expand=6)
    x3 = F.conv2d(x2, p_blocks_3_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x3 = bn_act(x3, b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
                p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias, act_type=2)
    x3 = F.conv2d(x3, p_blocks_3_depthwise_conv_0_weight, bias=None, stride=2, padding=2, groups=144)
    x3 = bn_act(x3, b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
                p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias, act_type=2)
    x3 = F.conv2d(x3, p_blocks_3_project_conv_0_weight, bias=None, stride=1, padding=0)
    x3 = bn_act(x3, b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
                p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias, act_type=0)
    # no residual (stride=2, channels change)
    
    # Block 4: MBConv6 (40->40, k=5, s=1, expand=6)
    identity4 = x3
    x4 = F.conv2d(x3, p_blocks_4_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x4 = bn_act(x4, b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
                p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias, act_type=2)
    x4 = F.conv2d(x4, p_blocks_4_depthwise_conv_0_weight, bias=None, stride=1, padding=2, groups=240)
    x4 = bn_act(x4, b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var,
                p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias, act_type=2)
    x4 = F.conv2d(x4, p_blocks_4_project_conv_0_weight, bias=None, stride=1, padding=0)
    x4 = bn_act(x4, b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var,
                p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias, act_type=0)
    x4 = x4 + identity4  # residual
    
    # Block 5: MBConv6 (40->80, k=3, s=2, expand=6)
    x5 = F.conv2d(x4, p_blocks_5_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x5 = bn_act(x5, b_blocks_5_expand_conv_1_running_mean, b_blocks_5_expand_conv_1_running_var,
                p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias, act_type=2)
    x5 = F.conv2d(x5, p_blocks_5_depthwise_conv_0_weight, bias=None, stride=2, padding=1, groups=240)
    x5 = bn_act(x5, b_blocks_5_depthwise_conv_1_running_mean, b_blocks_5_depthwise_conv_1_running_var,
                p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias, act_type=2)
    x5 = F.conv2d(x5, p_blocks_5_project_conv_0_weight, bias=None, stride=1, padding=0)
    x5 = bn_act(x5, b_blocks_5_project_conv_1_running_mean, b_blocks_5_project_conv_1_running_var,
                p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_1_bias, act_type=0)
    # no residual (stride=2, channels change)
    
    # Block 6: MBConv6 (80->80, k=3, s=1, expand=6)
    identity6 = x5
    x6 = F.conv2d(x5, p_blocks_6_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x6 = bn_act(x6, b_blocks_6_expand_conv_1_running_mean, b_blocks_6_expand_conv_1_running_var,
                p_blocks_6_expand_conv_1_weight, p_blocks_6_expand_conv_1_bias, act_type=2)
    x6 = F.conv2d(x6, p_blocks_6_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=480)
    x6 = bn_act(x6, b_blocks_6_depthwise_conv_1_running_mean, b_blocks_6_depthwise_conv_1_running_var,
                p_blocks_6_depthwise_conv_1_weight, p_blocks_6_depthwise_conv_1_bias, act_type=2)
    x6 = F.conv2d(x6, p_blocks_6_project_conv_0_weight, bias=None, stride=1, padding=0)
    x6 = bn_act(x6, b_blocks_6_project_conv_1_running_mean, b_blocks_6_project_conv_1_running_var,
                p_blocks_6_project_conv_1_weight, p_blocks_6_project_conv_1_bias, act_type=0)
    x6 = x6 + identity6  # residual
    
    # Block 7: MBConv6 (80->112, k=5, s=1, expand=6)
    x7 = F.conv2d(x6, p_blocks_7_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x7 = bn_act(x7, b_blocks_7_expand_conv_1_running_mean, b_blocks_7_expand_conv_1_running_var,
                p_blocks_7_expand_conv_1_weight, p_blocks_7_expand_conv_1_bias, act_type=2)
    x7 = F.conv2d(x7, p_blocks_7_depthwise_conv_0_weight, bias=None, stride=1, padding=2, groups=480)
    x7 = bn_act(x7, b_blocks_7_depthwise_conv_1_running_mean, b_blocks_7_depthwise_conv_1_running_var,
                p_blocks_7_depthwise_conv_1_weight, p_blocks_7_depthwise_conv_1_bias, act_type=2)
    x7 = F.conv2d(x7, p_blocks_7_project_conv_0_weight, bias=None, stride=1, padding=0)
    x7 = bn_act(x7, b_blocks_7_project_conv_1_running_mean, b_blocks_7_project_conv_1_running_var,
                p_blocks_7_project_conv_1_weight, p_blocks_7_project_conv_1_bias, act_type=0)
    # no residual (channels change)
    
    # Block 8: MBConv6 (112->112, k=5, s=1, expand=6)
    identity8 = x7
    x8 = F.conv2d(x7, p_blocks_8_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x8 = bn_act(x8, b_blocks_8_expand_conv_1_running_mean, b_blocks_8_expand_conv_1_running_var,
                p_blocks_8_expand_conv_1_weight, p_blocks_8_expand_conv_1_bias, act_type=2)
    x8 = F.conv2d(x8, p_blocks_8_depthwise_conv_0_weight, bias=None, stride=1, padding=2, groups=672)
    x8 = bn_act(x8, b_blocks_8_depthwise_conv_1_running_mean, b_blocks_8_depthwise_conv_1_running_var,
                p_blocks_8_depthwise_conv_1_weight, p_blocks_8_depthwise_conv_1_bias, act_type=2)
    x8 = F.conv2d(x8, p_blocks_8_project_conv_0_weight, bias=None, stride=1, padding=0)
    x8 = bn_act(x8, b_blocks