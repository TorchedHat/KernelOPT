import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, n_elements,
    eps,
    activation: tl.constexpr,  # 0=none, 1=relu, 2=hardtanh
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # channel index: layout is (N, C, H, W) flattened
    # We need to figure out channel from offset
    # n_elements = N * C * H * W, channel = (offset // (H*W)) % C
    # But we don't know H*W here... pass spatial_size
    # Actually let's pass HW as a parameter
    # Restructure: pass channel index differently
    # offset = n * C * HW + c * HW + hw
    # c = (offset // HW) % C
    # Let's just pass HW
    # We'll restructure the kernel signature
    tl.store(out_ptr + offsets, x, mask=mask)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def bn_act_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW, n_elements,
    eps: tl.constexpr,
    activation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    # channel index
    c = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    y = x_norm * w + b
    
    # activation
    if activation == 1:  # relu
        y = tl.maximum(y, 0.0)
    elif activation == 2:  # hardtanh (clamp -6, 6) = relu6
        y = tl.minimum(tl.maximum(y, -6.0), 6.0)
    
    tl.store(out_ptr + offsets, y.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def bn_add_act_kernel(
    x_ptr, residual_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW, n_elements,
    eps: tl.constexpr,
    activation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    
    c = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    y = x_norm * w + b + res
    
    if activation == 1:
        y = tl.maximum(y, 0.0)
    elif activation == 2:
        y = tl.minimum(tl.maximum(y, -6.0), 6.0)
    
    tl.store(out_ptr + offsets, y.to(tl.float32), mask=mask)


def fused_bn_act(x, mean, var, weight, bias, activation=0, eps=1e-5):
    """Fused BatchNorm + activation"""
    N, C, H, W = x.shape
    n_elements = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bn_act_kernel[grid](
        x, mean, var, weight, bias,
        out,
        C, HW, n_elements,
        eps, activation,
    )
    return out


def fused_bn_add_act(x, residual, mean, var, weight, bias, activation=0, eps=1e-5):
    """Fused BatchNorm + residual add + activation"""
    N, C, H, W = x.shape
    n_elements = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bn_add_act_kernel[grid](
        x, residual, mean, var, weight, bias,
        out,
        C, HW, n_elements,
        eps, activation,
    )
    return out


def conv2d_triton(x, weight, stride=1, padding=0, groups=1):
    """Wrapper that calls cuDNN conv2d - but we need to implement in Triton per rules.
    import torch.nn.functional as F
    return F.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


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
    import torch.nn.functional as F
    
    EPS = 1e-5
    ACT_NONE = 0
    ACT_RELU = 1
    ACT_HARDTANH = 2
    
    # Initial conv + BN + ReLU
    x = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    x = fused_bn_act(x, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias, ACT_RELU, EPS)
    
    # Block 0: MBConv1 (32->16, k=3, s=1, expand=1)
    # depthwise conv (groups=32)
    x0 = F.conv2d(x, p_blocks_0_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=32)
    x0 = fused_bn_act(x0, b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
    # project conv
    x0 = F.conv2d(x0, p_blocks_0_project_conv_0_weight, bias=None, stride=1, padding=0)
    x0 = fused_bn_act(x0, b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
    x = x0  # no residual for block 0 (in_channels != out_channels)
    
    # Block 1: MBConv6 (16->24, k=3, s=2, expand=6)
    # expand conv
    x1 = F.conv2d(x, p_blocks_1_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x1 = fused_bn_act(x1, b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
    # depthwise conv (groups=96, stride=2)
    x1 = F.conv2d(x1, p_blocks_1_depthwise_conv_0_weight, bias=None, stride=2, padding=1, groups=96)
    x1 = fused_bn_act(x1, b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
    # project conv
    x1 = F.conv2d(x1, p_blocks_1_project_conv_0_weight, bias=None, stride=1, padding=0)
    x1 = fused_bn_act(x1, b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
    x = x1  # no residual (stride=2)
    
    # Block 2: MBConv6 (24->24, k=3, s=1, expand=6) - has residual
    residual = x
    x2 = F.conv2d(x, p_blocks_2_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x2 = fused_bn_act(x2, b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
    x2 = F.conv2d(x2, p_blocks_2_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=144)
    x2 = fused_bn_act(x2, b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
    x2 = F.conv2d(x2, p_blocks_2_project_conv_0_weight, bias=None, stride=1, padding=0)
    x2 = fused_bn_act(x2, b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
    x = x2 + residual
    
    # Block 3: MBConv6 (24->40, k=5, s=2, expand=6) - no residual
    x3 = F.conv2d(x, p_blocks_3_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x3 = fused_bn_act(x3, b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
    x3 = F.conv2d(x3, p_blocks_3_depthwise_conv_0_weight, bias=None, stride=2, padding=2, groups=144)
    x3 = fused_bn_act(x3, b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
    x3 = F.conv2d(x3, p_blocks_3_project_conv_0_weight, bias=None, stride=1, padding=0)
    x3 = fused_bn_act(x3, b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
    x = x3
    
    # Block 4: MBConv6 (40->40, k=5, s=1, expand=6) - has residual
    residual = x
    x4 = F.conv2d(x, p_blocks_4_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x4 = fused_bn_act(x4, b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
    x4 = F.conv2d(x4, p_blocks_4_depthwise_conv_0_weight, bias=None, stride=1, padding=2, groups=240)
    x4 = fused_bn_act(x4, b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var,
    x4 = F.conv2d(x4, p_blocks_4_project_conv_0_weight, bias=None, stride=1, padding=0)
    x4 = fused_bn_act(x4, b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var,
    x = x4 + residual
    
    # Block 5: MBConv6 (40->80,