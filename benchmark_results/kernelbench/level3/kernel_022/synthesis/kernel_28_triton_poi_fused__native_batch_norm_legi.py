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
    key=['n_channels', 'numel'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, numel,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Channel index: layout is (N, C, H, W) so channel = (offset / (H*W)) % C
    # But we pass channel stride info via n_channels and spatial size
    # For NCHW: channel = (offset // spatial_size) % n_channels
    # We compute channel from offset
    c = (offsets // (numel // n_channels)) % n_channels
    # Actually numel = N*C*H*W, spatial = H*W, so channel = (offset // spatial) % C
    # Let's use: channel index from flat index
    # flat index in NCHW: n*C*H*W + c*H*W + h*W + w
    # c = (offset // (H*W)) % C, but we don't have H*W directly
    # We pass spatial_size = numel // (N * C) ... but we don't have N
    # Better: pass hw_size = H*W
    # For now use: c = (offset // hw_size) % n_channels
    # We'll restructure to pass hw_size
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


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
def bn_relu_kernel_hw(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, numel, hw_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    c = (offsets // hw_size) % n_channels
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


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
def bn_hardtanh_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, numel, hw_size,
    eps: tl.constexpr,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    c = (offsets // hw_size) % n_channels
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    out = tl.minimum(tl.maximum(out, min_val), max_val)
    
    tl.store(out_ptr + offsets, out, mask=mask)


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
def bn_only_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_channels, numel, hw_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    c = (offsets // hw_size) % n_channels
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)


def apply_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    """Apply BatchNorm + ReLU using Triton kernel."""
    N, C, H, W = x.shape
    numel = x.numel()
    hw_size = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel_hw[grid](
        x, out,
        running_mean, running_var, weight, bias,
        C, numel, hw_size,
        eps=eps,
    )
    return out


def apply_bn_hardtanh(x, weight, bias, running_mean, running_var, eps=1e-5, min_val=-6.0, max_val=6.0):
    """Apply BatchNorm + HardTanh using Triton kernel."""
    N, C, H, W = x.shape
    numel = x.numel()
    hw_size = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_hardtanh_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        C, numel, hw_size,
        eps=eps,
        min_val=min_val,
        max_val=max_val,
    )
    return out


def apply_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    """Apply BatchNorm using Triton kernel."""
    N, C, H, W = x.shape
    numel = x.numel()
    hw_size = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_only_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        C, numel, hw_size,
        eps=eps,
    )
    return out


def conv2d(x, weight, bias=None, stride=1, padding=0, groups=1):
    """Wrapper for conv2d using PyTorch (cuDNN)."""
    import torch.nn.functional as F
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, groups=groups)


def mbconv_block(x, expand_w, expand_bn_w, expand_bn_b, expand_rm, expand_rv,
                 dw_w, dw_bn_w, dw_bn_b, dw_rm, dw_rv,
                 proj_w, proj_bn_w, proj_bn_b, proj_rm, proj_rv,
    """MBConv block with fused BN+activation."""
    import torch.nn.functional as F
    in_channels = x.shape[1]
    
    if expand_ratio != 1:
        # Expand conv
        x_exp = F.conv2d(x, expand_w, bias=None, stride=1, padding=0)
        x_exp = apply_bn_hardtanh(x_exp, expand_bn_w, expand_bn_b, expand_rm, expand_rv)
    else:
        x_exp = x
    
    # Depthwise conv
    pad = (kernel_size - 1) // 2
    x_dw = F.conv2d(x_exp, dw_w, bias=None, stride=stride, padding=pad, groups=x_exp.shape[1])
    x_dw = apply_bn_hardtanh(x_dw, dw_bn_w, dw_bn_b, dw_rm, dw_rv)
    
    # Project conv
    x_proj = F.conv2d(x_dw, proj_w, bias=None, stride=1, padding=0)
    x_proj = apply_bn(x_proj, proj_bn_w, proj_bn_b, proj_rm, proj_rv)
    
    # Skip connection
    if stride == 1 and x.shape == x_proj.shape:
        x_proj = x_proj + x
    
    return x_proj


def kernel_function(
    p_conv1_weight,
    p_bn1_weight,
    p_bn1_bias,
    p_blocks_0_depthwise_conv_0_weight,
    p_blocks_0_depthwise_conv_1_weight,
    p_blocks_0_depthwise_conv_1_bias,
    p_blocks_0_project_conv_0_weight,
    p_blocks_0_project_conv_1_weight,
    p_blocks_0_project_conv_1_bias,
    p_blocks_1_expand_conv_0_weight,
    p_blocks_1_expand_conv_1_weight,
    p_blocks_1_expand_conv_1_bias,
    p_blocks_1_depthwise_conv_0_weight,
    p_blocks_1_depthwise_conv_1_weight,
    p_blocks_1_depthwise_conv_1_bias,
    p_blocks_1_project_conv_0_weight,
    p_blocks_1_project_conv_1_weight,
    p_blocks_1_project_conv_1_bias,
    p_blocks_2_expand_conv_0_weight,
    p_blocks_2_expand_conv_1_weight,
    p_blocks_2_expand_conv_1_bias,
    p_blocks_2_depthwise_conv_0_weight,
    p_blocks_2_depthwise_conv_1_weight,
    p_blocks_2_depthwise_conv_1_bias,
    p_blocks_2_project_conv_0_weight,
    p_blocks_2_project_conv_1_weight,
    p_blocks_2_project_conv_1_bias,
    p_blocks_3_expand_conv_0_weight,
    p_blocks_3_expand_conv_1_weight,
    p_blocks_3_expand_conv_1_bias,
    p_blocks_3_depthwise_conv_0_weight,
    p_blocks_3_depthwise_conv_1_weight,
    p_blocks_3_depthwise_conv_1_bias,
    p_blocks_3_project_conv_0_weight,
    p_blocks_3_project_conv_1_weight,
    p_blocks_3_project_conv_1_bias,
    p_blocks_4_expand_conv_0_weight,
    p_blocks_4_expand_conv_1_weight,
    p_blocks_4_expand_conv_1_bias,
    p_blocks_4_depthwise_conv_0_weight,
    p_blocks_4_depthwise_conv_1_weight,
    p_blocks_4_depthwise_conv_1_bias,
    p_blocks_4_project_conv_0_weight,
    p_blocks_4_project_conv_1_weight,
    p_blocks_4_project_conv_1_bias,
    p_blocks_5_expand_conv_0_weight,
    p_blocks_5_expand_conv_1_weight,
    p_blocks_5_expand_conv_1_bias,
    p_blocks_5_depthwise_conv_0_weight,
    p_blocks_5_depthwise_conv_1_weight,
    p_blocks_5_depthwise_conv_1_bias,
    p_blocks_5_project_conv_0_weight,
    p_blocks_5_project_conv_1_weight,
    p_blocks_5_project_conv_1_bias,
    p_blocks_6_expand_conv_0_weight,
    p_blocks_6_expand_conv_1_weight,
    p_blocks_6_expand_conv_1_bias,
    p_blocks_6_depthwise_conv_0_weight,
    p_blocks_6_depthwise_conv_1_weight,
    p_blocks_6_depthwise_conv_1_bias,
    p_blocks_6_project_conv_0_weight,
    p_blocks_6_project_conv_1_weight,
    p_blocks_6_project_conv_1_bias,
    p_blocks_7_expand_conv_0_weight,
    p_blocks_7_expand_conv_1_weight,
    p_blocks_7_expand_conv_1_bias,
    p_blocks_7_depthwise_conv_0_weight,
    p_blocks_7_depthwise_conv_1_weight,
    p_blocks_7_depthwise_conv_1_bias,
    p_blocks_7_project_conv_0_weight,
    p_blocks_7_project_conv_1_weight,
    p_blocks_7_project_conv_1_bias,
    p_blocks_8_expand_conv_0_weight,
    p_blocks_8_expand_conv_1_weight,
    p_blocks_8_expand_conv_1_bias,
    p_blocks_8_depthwise_conv_0_weight,
    p_blocks_8_depthwise_conv_1_weight,
    p_blocks_8_depthwise_conv_1_bias,
    p_blocks_8_project_conv_0_weight,
    p_blocks_8_project_conv_1_weight,
    p_blocks_8_project_conv_1_bias,
    p_blocks_9_expand_conv_0_weight,
    p_blocks_9_expand_conv_1_weight,
    p_blocks_9_expand_conv_1_bias,
    p_blocks_9_depthwise_conv_0_weight,
    p_blocks_9_depthwise_conv_1_weight,
    p_blocks_9_depthwise_conv_1_bias,
    p_blocks_9_project_conv_0_weight,
    p_blocks_9_project_conv_1_weight,
    p_blocks_9_project_conv_1_bias,
    p_blocks_10_expand_conv_0_weight,
    p_blocks_10_expand_conv_1_weight,
    p_blocks_10_expand_conv_1_bias,
    p_blocks_10_depthwise_conv_0_weight,
    p_blocks_10_depthwise_conv_1_weight,
    p_blocks_10_depthwise_conv_1_bias,
    p_blocks_10_project_conv_0_weight,
    p_blocks_10_project_conv_1_weight,
    p_blocks_10_project_conv_1_bias,
    p_blocks_11_expand_conv_0_weight,
    p_blocks_11_expand_conv_1_weight,
    p_blocks_11_expand_conv_1_bias,
    p_blocks_11_depthwise_conv_0_weight,
    p_blocks_11_depthwise_conv_1_weight,
    p_blocks_11_depthwise_conv_1_bias,
    p_blocks_11_project_conv_0_weight,
    p_blocks_11_project_conv_1_weight,
    p_blocks_11_project_conv_1_bias,
    p_blocks_12_expand_conv_0_weight,
    p_blocks_12_expand_conv_1_weight,
    p_blocks_12_expand_conv_1_bias,
    p_blocks_12_depthwise_conv_0_weight,
    p_blocks_12_depthwise_conv_1_weight,
    p_blocks_12_depthwise_conv_1_bias,
    p_blocks_12_project_conv_0_weight,
    p_blocks_12_project_conv_1_weight,
    p_blocks_12_project_conv_1_bias,
    p_conv2_weight,
    p_bn2_weight,
    p_bn2_bias,
    p_fc_weight,
    p_fc_bias,
    b_bn1_running_mean,
    b_bn1_running_var,
    b_blocks_0_depthwise_conv_1_running_mean,
    b_blocks_0_depthwise_conv_1_running_var,
    b_blocks_0_project_conv_1_running_mean,
    b_blocks_0_project_conv_1_running_var,
    b_blocks_1_expand_conv_1_running_mean,
    b_blocks_1_expand_conv_1_running_var,
    b_blocks_1_depthwise_conv_1_running_mean,
    b_blocks_1_depthwise_conv_1_running_var,
    b_blocks_1_project_conv_1_running_mean,
    b_blocks_1_project_conv_1_running_var,
    b_blocks_2_expand_conv_1_running_mean,
    b_blocks_2_expand_conv_1_running_var,
    b_blocks_2_depthwise_conv_1_running_mean,
    b_blocks_2_depthwise_conv_1_running_var,
    b_blocks_2_project_conv_1_running_mean,
    b_blocks_2_project_conv_1_running_var,
    b_blocks_3_expand_conv_1_running_mean,
    b_blocks_3_expand_conv_1_running_var,
    b_blocks_3_depthwise_conv_1_running_mean,
    b_blocks_3_depthwise_conv_1_running_var,
    b_blocks_3_project_conv_1_running_mean,
    b_blocks_3_project_conv_1_running_var,
    b_blocks_4_expand_conv_1_running_mean,
    b_blocks_4_expand_conv_1_running_var,
    b_blocks_4_depthwise_conv_1_running_mean,
    b_blocks_4_depthwise_conv_1_running_var,
    b_blocks_4_project_conv_1_running_mean,
    b_blocks_4_project_conv_1_running_var,
    b_blocks_5_expand_conv_1_running_mean,
    b_blocks_5_expand_conv_1_running_var,
    b_blocks_5_depthwise_conv_1_running_mean,
    b_blocks_5_depthwise_conv_1_running_var,
    b_blocks_5_project_conv_1_running_mean,
    b_blocks_5_project_conv_1_running_var,
    b_blocks_6_expand_conv_1_running_mean,
    b_blocks_6_expand_conv_1_running_var,
    b_blocks_6_depthwise_conv_1_running_mean,
    b_blocks_6_depthwise_conv_1_running_var,
    b_blocks_6_project_conv_1_running_mean,
    b_blocks_6_project_conv_1_running_var,
    b_blocks_7_expand_conv_1_running_mean,
    b_blocks_7_expand_conv_1_running_var,
    b_blocks_7_depthwise_conv_1_running_mean,
    b_blocks_7_depthwise_conv_1_running_var,
    b_blocks_7_project_conv_1_running_mean,
    b_blocks_7_project_conv_1_running_var,
    b_blocks_8_expand_conv_1_running_mean,
    b_blocks_8_expand_conv_1_running_var,
    b_blocks_8_depthwise_conv_1_running_mean,
    b_blocks_8_depthwise_conv_1_running_var,
    b_blocks_8_project_conv_1_running_mean,
    b_blocks_8_project_conv_1_running_var,
    b_blocks_9_expand_conv_1_running_mean,
    b_blocks_9_expand_conv_1_running_var,
    b_blocks_9_depthwise_conv_1_running_mean,
    b_blocks_9_depthwise_conv_1_running_var,
    b_blocks_9_project_conv_1_running_mean,
    b_blocks_9_project_conv_1_running_var,
    b_blocks_10_expand_conv_1_running_mean,
    b_blocks_10_expand_conv_1_running_var,
    b_blocks_10_depthwise_conv_1_running_mean,
    b_blocks_10_depthwise_conv_1_running_var,
    b_blocks_10_project_conv_1_running_mean,
    b_blocks_10_project_conv_1_running_var,
    b_blocks_11_expand_conv_1_running_mean,
    b_blocks_11_expand_conv_1_running_var,
    b_blocks_11_depthwise_conv_1_running_mean,
    b_blocks_11_depthwise_conv_1_running_var,
    b_blocks_11_project_conv_1_running_mean,
    b_blocks_11_project_conv_1_running_var,
    b_blocks_12_expand_conv_1_running_mean,
    b_blocks_12_expand_conv_1_running_var,
    b_blocks_12_depthwise_conv_1_running_mean,
    b_blocks_12_depthwise_conv_1_running_var,
    b_blocks_12_project_conv_1_running_mean,
    b_blocks_12_project_conv_1_running_var,
    b_bn2_running_mean,
    b_bn2_running_var,
    x,
):
    import torch.nn.functional as F
    
    # Initial conv + BN + ReLU
    x = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    x = apply_bn_relu(x, p_bn1_weight, p_bn1_bias, b_bn1_running_mean, b_bn1_running_var)
    
    # Block 0: MBConv1 (32->16, stride=1, expand=1, kernel=3)
    # No expand conv for expand_ratio=1
    x_dw = F.conv2d(x, p_blocks_0_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=32)
    x_dw = apply_bn_hardtanh(x_dw, p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
    x_proj = F.conv2d(x_dw, p_blocks_0_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = apply_bn(x_proj, p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
    # No skip for block 0 (in_channels=32 != out_channels=16)
    
    # Block 1: MBConv6 (16->24, stride=2, expand=6, kernel=3)
    x_exp = F.conv2d(x, p_blocks_1_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x_exp = apply_bn_hardtanh(x_exp, p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
    x_dw = F.conv2d(x_exp, p_blocks_1_depthwise_conv_0_weight, bias=None, stride=2, padding=1, groups=96)
    x_dw = apply_bn_hardtanh(x_dw, p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
    x_proj = F.conv2d(x_dw, p_blocks_1_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = apply_bn(x_proj, p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
    # No skip (stride=2, different channels)
    
    # Block 2: MBConv6 (24->24, stride=1, expand=6, kernel=3) - has skip
    x_in = x
    x_exp = F.conv2d(x, p_blocks_2_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x_exp = apply_bn_hardtanh(x_exp, p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
    x_dw = F.conv2d(x_exp, p_blocks_2_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=144)
    x_dw = apply_bn_hardtanh(x_dw, p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
    x_proj = F.conv2d(x_dw, p_blocks_2_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = apply_bn(x_proj, p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
    x = x + x_in  # skip connection
    
    # Block 3