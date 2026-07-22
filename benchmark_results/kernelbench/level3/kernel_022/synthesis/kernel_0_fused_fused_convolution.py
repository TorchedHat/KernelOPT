import torch
import triton
import triton.language as tl

# Fused BatchNorm + ReLU kernel
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
    numel, num_channels, spatial,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # channel index: offsets // spatial % num_channels
    # For NCHW layout: index = n * C * H * W + c * H * W + h * W + w
    # channel = (offsets // spatial) % num_channels
    chan = (offsets // spatial) % num_channels
    
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    
    # ReLU
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Fused BatchNorm + HardTanh kernel
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
def bn_hardtanh_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    chan = (offsets // spatial) % num_channels
    
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    
    # HardTanh: clamp to [-6, 6] (ReLU6)
    out = tl.minimum(tl.maximum(out, -6.0), 6.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Fused BatchNorm only (no activation) kernel
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
    numel, num_channels, spatial,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    chan = (offsets // spatial) % num_channels
    
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)


def apply_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3]
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def apply_bn_hardtanh(x, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3]
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_hardtanh_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def apply_bn(x, mean, var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3]
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def conv2d(x, weight, stride=1, padding=0, groups=1):
    return torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


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
    # Initial conv + bn + relu
    x = conv2d(x, p_conv1_weight, stride=2, padding=1)
    x = apply_bn_relu(x, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias)

    # Block 0: MBConv1 (32->16, stride=1, expand=1)
    # depthwise conv (groups=32)
    identity = x
    t = conv2d(x, p_blocks_0_depthwise_conv_0_weight, stride=1, padding=1, groups=32)
    t = apply_bn_hardtanh(t, b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
    t = conv2d(t, p_blocks_0_project_conv_0_weight, stride=1, padding=0)
    x = apply_bn(t, b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
    # No residual for block 0 (in_channels != out_channels)

    # Block 1: MBConv6 (16->24, stride=2, expand=6)
    t = conv2d(x, p_blocks_1_expand_conv_0_weight, stride=1, padding=0)
    t = apply_bn_hardtanh(t, b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
    t = conv2d(t, p_blocks_1_depthwise_conv_0_weight, stride=2, padding=1, groups=96)
    t = apply_bn_hardtanh(t, b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
    t = conv2d(t, p_blocks_1_project_conv_0_weight, stride=1, padding=0)
    x = apply_bn(t, b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
    # No residual (stride=2)

    # Block 2: MBConv6 (24->24, stride=1, expand=6)
    identity = x
    t = conv2d(x, p_blocks_2_expand_conv_0_weight, stride=1, padding=0)
    t = apply_bn_hardtanh(t, b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
    t = conv2d(t, p_blocks_2_depthwise_conv_0_weight, stride=1, padding=1, groups=144)
    t = apply_bn_hardtanh(t, b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
    t = conv2d(t, p_blocks_2_project_conv_0_weight, stride=1, padding=0)
    x = apply_bn(t, b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
    x = x + identity

    # Block 3: MBConv6 (24->40, stride=2, expand=6, kernel=5)
    t = conv2d(x, p_blocks_3_expand_conv_0_weight, stride=1, padding=0)
    t = apply_bn_hardtanh(t, b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
    t = conv2d(t, p_blocks_3_depthwise_conv_0_weight, stride=2, padding=2, groups=144)
    t = apply_bn_hardtanh(t, b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
    t = conv2d(t, p_blocks_3_project_conv_0_weight, stride=1, padding=0)
    x = apply_bn(t, b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
    # No residual (stride=2)

    # Block 4: MBConv6 (40->40, stride=1, expand=6, kernel=5)
    identity = x
    t = conv2d(x, p_blocks_4_expand_conv_0_weight, stride=1, padding=0)
    t = apply_bn_hardtanh(t, b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
    t = conv2d(t, p_blocks_4_depthwise_conv_0_weight, stride=1, padding=2, groups=240)
    t = apply_bn_hardtanh(t, b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var,
    t = conv2d(t, p_blocks_4_project_conv_0_weight, stride=1, padding=0)
    x = apply_bn(t, b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var,
    x = x + identity

    # Block 5: MBConv6 (40->80, stride=2, expand=6, kernel=3)
    t = conv2d(x, p_blocks_5_expand_conv_0_weight, stride=1, padding=0)
    t = apply_bn_hardtanh(t, b_blocks_5_expand_conv_1_running_mean, b_blocks_5_expand_conv_1_running_var,
    t = conv2d(t, p_blocks_5_depthwise_conv_0_weight, stride=2, padding=1, groups=240)
    t = apply_bn_hardtanh(t, b_blocks_5_depthwise_conv_1_running_mean, b_blocks_5_depthwise_conv_1_running_var,
    t = conv2d(t, p_blocks_5_project_conv_0_weight, stride=1, padding=0)
    x = apply_bn(t, b_blocks_5_project_conv_1_running_mean, b_blocks_5_project_conv_1_running_var,
    # No residual (stride=2)

    # Block 6: MBConv6 (80->80, stride=1, expand=6, kernel=3)
    identity = x
    t = conv2d(x, p_blocks_6_expand_conv_0_weight, stride=1, padding=0)
    t = apply_bn_hardtanh(t, b_blocks_6_expand_conv_1_running_mean, b_blocks_6_expand_conv_1_running_var,
    t = conv2d(t, p_blocks_6_depthwise_conv_0_weight, stride=1, padding=1, groups=480)
    t = apply_bn_hardtanh(t, b_blocks_6_depthwise_conv_1_running_mean, b_blocks_6_depthwise_conv_1_running_var,
    t = conv2d(t, p_blocks_6_project_conv_0_weight, stride=1, padding=0)
    x = apply_bn(t, b_blocks_6_project_conv_1_running_mean, b_blocks_6_project_conv_1_running_var,
    x = x + identity

    # Block 7: MBConv6 (80->112, stride=1, expand=6, kernel=5)
    t = conv2d(x, p_blocks_7_expand_conv_0_weight, stride=1, padding=0)
    t = apply_bn_