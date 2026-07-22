import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_act_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial_size,
    eps,
    activation: tl.constexpr,  # 0=none, 1=relu, 2=hardtanh
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Determine channel index
    c = (offsets // spatial_size) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_out = x_norm * w + b
    
    # Activation
    if activation == 1:  # ReLU
        x_out = tl.maximum(x_out, 0.0)
    elif activation == 2:  # HardTanh [-6, 6]
        x_out = tl.minimum(tl.maximum(x_out, -6.0), 6.0)
    
    tl.store(out_ptr + offsets, x_out, mask=mask)


def fused_bn_act(x, mean, var, weight, bias, eps=1e-5, activation=0):
    """
    activation: 0=none, 1=relu, 2=hardtanh
    """
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial_size = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_act_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, num_channels, spatial_size,
        eps,
        activation,
    )
    return out


def mbconv(x, expand_w, expand_bn_w, expand_bn_b, expand_bn_mean, expand_bn_var,
           dw_w, dw_bn_w, dw_bn_b, dw_bn_mean, dw_bn_var,
           proj_w, proj_bn_w, proj_bn_b, proj_bn_mean, proj_bn_var,
    """MBConv block"""
    residual = x
    
    # Expand phase (if expand_ratio > 1)
    if expand_ratio > 1:
        x = F.conv2d(x, expand_w, bias=None, stride=1, padding=0)
        x = fused_bn_act(x, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b, activation=2)
    
    # Depthwise conv
    pad = (kernel_size - 1) // 2
    x = F.conv2d(x, dw_w, bias=None, stride=stride, padding=pad, groups=x.shape[1])
    x = fused_bn_act(x, dw_bn_mean, dw_bn_var, dw_bn_w, dw_bn_b, activation=2)
    
    # Project
    x = F.conv2d(x, proj_w, bias=None, stride=1, padding=0)
    x = fused_bn_act(x, proj_bn_mean, proj_bn_var, proj_bn_w, proj_bn_b, activation=0)
    
    # Residual connection
    if stride == 1 and residual.shape == x.shape:
        x = x + residual
    
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
    x = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    x = fused_bn_act(x, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias, activation=1)
    
    # Block 0: MBConv1 (32->16, k=3, s=1, expand=1) - no expand conv
    residual = x
    x = F.conv2d(x, p_blocks_0_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=32)
    x = fused_bn_act(x, b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
                     p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias, activation=2)
    x = F.conv2d(x, p_blocks_0_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = fused_bn_act(x, b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
                     p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias, activation=0)
    # No residual for block 0 (in_channels != out_channels: 32 != 16)
    
    # Block 1: MBConv6 (16->24, k=3, s=2, expand=6)
    x = mbconv(x,
               p_blocks_1_expand_conv_0_weight, p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
               b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
               p_blocks_1_depthwise_conv_0_weight, p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
               b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
               p_blocks_1_project_conv_0_weight, p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
               b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
               stride=2, expand_ratio=6, in_channels=16, kernel_size=3)
    
    # Block 2: MBConv6 (24->24, k=3, s=1, expand=6)
    x = mbconv(x,
               p_blocks_2_expand_conv_0_weight, p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
               b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
               p_blocks_2_depthwise_conv_0_weight, p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
               b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
               p_blocks_2_project_conv_0_weight, p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
               b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
               stride=1, expand_ratio=6, in_channels=24, kernel_size=3)
    
    # Block 3: MBConv6 (24->40, k=5, s=2, expand=6)
    x = mbconv(x,
               p_blocks_3_expand_conv_0_weight, p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
               b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
               p_blocks_3_depthwise_conv_0_weight, p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
               b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
               p_blocks_3_project_conv_0_weight, p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
               b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
               stride=2, expand_ratio=6, in_channels=24, kernel_size=5)
    
    # Block 4: MBConv6 (40->40, k=5, s=1, expand=6)
    x = mbconv(x,
               p_blocks_4_expand_conv_0_weight, p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
               b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
               p_blocks_4_depthwise_conv_0_weight, p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
               b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var,
               p_blocks_4_project_conv_0_weight, p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,
               b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var,
               stride=1, expand_ratio=6, in_channels=40, kernel_size=5)
    
    # Block 5: MBConv6 (40->80, k=3, s=2, expand=6)
    x = mbconv(x,
               p_blocks_5_expand_conv_0_weight, p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias,
               b_blocks_5_expand_conv_1_running_mean, b_blocks_5_expand_conv_1_running_var,
               p_blocks_5_depthwise_conv_0_weight, p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias,
               b_blocks_5_depthwise_conv_1_running_mean, b_blocks_5_depthwise_conv_1_running_var,
               p_blocks_5_project_conv_0_weight, p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_1_bias,
               b_blocks_5_project_conv_1_running_mean, b_blocks_5_project_conv_1_running_var,
               stride=2, expand_ratio=6, in_channels=40, kernel_size=3)
    
    # Block 6: MBConv6 (80->80, k=3, s=1, expand=6)
    x = mbconv(x,
               p_blocks_6_expand_conv_0_weight, p_blocks_6_expand_conv_1_weight, p_blocks_6_expand_conv_1_bias,
               b_blocks_6_expand_conv_1_running_mean, b_blocks_6_expand_conv_1_running_var,
               p_blocks_6_depthwise_conv_0_weight, p_blocks_6_depthwise_conv_1_weight, p_blocks_6_depthwise_conv_1_bias,
               b_blocks_6_depthwise_conv_1_running_mean, b_blocks_6_depthwise_conv_1_running_var,
               p_blocks_6_project_conv_0_weight, p_blocks_6_project_conv_1_weight, p_blocks_6_project_conv_1_bias,
               b_blocks_6_project_conv_1_running_mean, b_blocks_6_project_conv_1_running_var,
               stride=1, expand_ratio=6, in_channels=80, kernel_size=3)
    
    # Block 7: MBConv6 (80->112, k=5, s=1, expand=6)
    x = mbconv(x,
               p_blocks_7_expand_conv_0_weight, p_blocks_7_expand_conv_1_weight, p_blocks_7_expand_conv_1_bias,
               b_blocks_7_expand_conv_1_running_mean, b_blocks_7_expand_conv_1_running_var,
               p_blocks_7_depthwise_conv_0_weight, p_blocks_7_depthwise_conv_1_weight, p_blocks_7_depthwise_conv_1_bias,
               b_blocks_7_depthwise_conv_1_running_mean, b_blocks_7_depthwise_conv_1_running_var,
               p_blocks_7_project_conv_0_weight, p_blocks_7_project_conv_1_weight, p_blocks_7_project_conv_1_bias,
               b_blocks_7_project_conv_1_running_mean, b_blocks_7_project_conv_1_running_var,
               stride=1, expand_ratio=6, in_channels=80, kernel_size=5)
    
    # Block 8: MBConv6 (112->112, k=5, s=1, expand=6)
    x = mbconv(x,
               p_blocks_8_expand_conv_0_weight, p_blocks_8_expand_conv_1_weight, p_blocks_8_expand_conv_1_bias,
               b_blocks_8_expand_conv_1_running_mean, b_blocks_8_expand_conv_1_running_var,
               p_blocks_8_depthwise_conv_0_weight, p_blocks_8_depthwise_conv_1_weight, p_blocks_8_depthwise_conv_1_bias,
               b_blocks_8_depthwise_conv_1_running_mean, b_blocks_8_depthwise_conv_1_running_var,
               p_blocks_8_project_conv_0_weight, p_blocks_8_project_conv_1_weight, p_blocks_8_project_conv_1_bias,
               b_blocks_8_project_conv_1_running_mean, b_blocks_8_project_conv_1_running_var,
               stride=1, expand_ratio=6, in_channels=112, kernel_size=5)
    
    # Block 9: MBConv6 (112->192, k=5, s=2, expand=6)
    x = mbconv(x,
               p_blocks_9_expand_conv_0_weight, p_blocks_9_expand_conv_1_weight, p_blocks_9_expand_conv_1_bias,
               b_blocks_9_expand_conv_1_running_mean, b_blocks_9_expand_conv_1_running_var,
               p_blocks_9_depthwise_conv_0_weight, p_blocks_9_depthwise_conv_1_weight, p_blocks_9_depthwise_conv_1_bias,
               b_blocks_9_depthwise_conv_1_running_mean, b_blocks_9_depthwise_conv_1_running_var,
               p_blocks_9_project_conv_0_weight, p_blocks_9_project_conv_1_weight, p_blocks_9_project_conv_1_bias,
               b_blocks_9_project_conv_1_running_mean, b_blocks_9_project_conv_1_running_var,
               stride=2, expand_ratio=6, in_channels=112, kernel_size=5)
    
    # Block 10: MBConv6 (192->192, k=5, s=1, expand=6)
    x = mbconv(x,
               p_blocks_10_expand_conv_0_weight, p_blocks_10_expand_conv_1_weight, p_blocks_10_expand_conv_1_bias,