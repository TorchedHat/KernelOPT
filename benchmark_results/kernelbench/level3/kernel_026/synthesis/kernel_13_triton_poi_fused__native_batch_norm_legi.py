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
    key=['numel'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
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
    key=['numel'],
)
@triton.jit
def bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
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
    key=['numel'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    numel,
    N, C, H, W,
    groups,
    channels_per_group,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # offsets index into output tensor (N, C, H, W) in NCHW layout
    w_idx = offsets % W
    tmp = offsets // W
    h_idx = tmp % H
    tmp = tmp // H
    c_out = tmp % C
    n_idx = tmp // C
    
    # In channel shuffle: output channel c_out comes from input channel
    # g = c_out // channels_per_group (group index in output)
    # k = c_out % channels_per_group (position within group)
    # input channel = k * groups + g
    g = c_out // channels_per_group
    k = c_out % channels_per_group
    c_in = k * groups + g
    
    in_offset = w_idx + W * h_idx + W * H * c_in + W * H * C * n_idx
    
    val = tl.load(x_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


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
def add_relu_kernel(
    x_ptr, y_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = x + y
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
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    # Simple linear: out[m, n] = sum_k x[m,k] * w[n,k] + b[n]
    # This is a pointwise epilogue after matmul - but we need full matmul
    # We'll handle this differently
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    tl.store(out_ptr + offsets, tl.load(x_ptr + offsets, mask=mask, other=0.0), mask=mask)


def fused_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = x.numel()
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias, out,
    )
    return out


def fused_bn(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = x.numel()
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, mean, var, weight, bias, out,
    )
    return out


def channel_shuffle(x, groups):
    N, C, H, W = x.shape
    channels_per_group = C // groups
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
        x, out,
        numel,
        N, C, H, W,
    )
    return out


def shufflenet_unit(x, conv1_w, bn1_mean, bn1_var, bn1_w, bn1_b,
                    conv2_w, bn2_mean, bn2_var, bn2_w, bn2_b,
                    conv3_w, bn3_mean, bn3_var, bn3_w, bn3_b,
                    shortcut_conv_w=None, shortcut_bn_mean=None, shortcut_bn_var=None,
                    shortcut_bn_w=None, shortcut_bn_b=None,
                    groups=3, in_channels=None, out_channels=None):
    # conv1: 1x1 group conv
    out = F.conv2d(x, conv1_w, bias=None, stride=1, padding=0, groups=groups)
    out = fused_bn_relu(out, bn1_mean, bn1_var, bn1_w, bn1_b)
    
    # conv2: depthwise 3x3
    mid_channels = out.shape[1]
    out = F.conv2d(out, conv2_w, bias=None, stride=1, padding=1, groups=mid_channels)
    out = fused_bn(out, bn2_mean, bn2_var, bn2_w, bn2_b)
    
    # channel shuffle
    out = channel_shuffle(out, groups)
    
    # conv3: 1x1 group conv
    out = F.conv2d(out, conv3_w, bias=None, stride=1, padding=0, groups=groups)
    out = fused_bn_relu(out, bn3_mean, bn3_var, bn3_w, bn3_b)
    
    # shortcut
    if shortcut_conv_w is not None:
        sc = F.conv2d(x, shortcut_conv_w, bias=None, stride=1, padding=0)
        sc = fused_bn(sc, shortcut_bn_mean, shortcut_bn_var, shortcut_bn_w, shortcut_bn_b)
        out = out + sc
    else:
        out = out + x
    
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
    # Initial conv + bn + relu + maxpool
    out = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    out = fused_bn_relu(out, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias)
    out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

    # Stage 2, block 0 (with shortcut conv)
    out = shufflenet_unit(
        out,
        p_stage2_0_conv1_weight, b_stage2_0_bn1_running_mean, b_stage2_0_bn1_running_var, p_stage2_0_bn1_weight, p_stage2_0_bn1_bias,
        p_stage2_0_conv2_weight, b_stage2_0_bn2_running_mean, b_stage2_0_bn2_running_var, p_stage2_0_bn2_weight, p_stage2_0_bn2_bias,
        p_stage2_0_conv3_weight, b_stage2_0_bn3_running_mean, b_stage2_0_bn3_running_var, p_stage2_0_bn3_weight, p_stage2_0_bn3_bias,
        shortcut_conv_w=p_stage2_0_shortcut_0_weight,
        shortcut_bn_mean=b_stage2_0_shortcut_1_running_mean,
        shortcut_bn_var=b_stage2_0_shortcut_1_running_var,
        shortcut_bn_w=p_stage2_0_shortcut_1_weight,
        shortcut_bn_b=p_stage2_0_shortcut_1_bias,
        groups=3,
    )

    # Stage 2, block 1
    out = shufflenet_unit(
        out,
        p_stage2_1_conv1_weight, b_stage2_1_bn1_running_mean, b_stage2_1_bn1_running_var, p_stage2_1_bn1_weight, p_stage2_1_bn1_bias,
        p_stage2_1_conv2_weight, b_stage2_1_bn2_running_mean, b_stage2_1_bn2_running_var, p_stage2_1_bn2_weight, p_stage2_1_bn2_bias,
        p_stage2_1_conv3_weight, b_stage2_1_bn3_running_mean, b_stage2_1_bn3_running_var, p_stage2_1_bn3_weight, p_stage2_1_bn3_bias,
        groups=3,
    )

    # Stage 2, block 2
    out = shufflenet_unit(
        out,
        p_stage2_2_conv1_weight, b_stage2_2_bn1_running_mean, b_stage2_2_bn1_running_var, p_stage2_2_bn1_weight, p_stage2_2_bn1_bias,
        p_stage2_2_conv2_weight, b_stage2_2_bn2_running_mean, b_stage2_2_bn2_running_var, p_stage2_2_bn2_weight, p_stage2_2_bn2_bias,
        p_stage2_2_conv3_weight, b_stage2_2_bn3_running_mean, b_stage2_2_bn3_running_var, p_stage2_2_bn3_weight, p_stage2_2_bn3_bias,
        groups=3,
    )

    # Stage 3, block 0 (with shortcut conv)
    out = shufflenet_unit(
        out,
        p_stage3_0_conv1_weight, b_stage3_0_bn1_running_mean, b_stage3_0_bn1_running_var, p_stage3_0_bn1_weight, p_stage3_0_bn1_bias,
        p_stage3_0_conv2_weight, b_stage3_0_bn2_running_mean, b_stage3_0_bn2_running_var, p_stage3_0_bn2_weight, p_stage3_0_bn2_bias,
        p_stage3_0_conv3_weight, b_stage3_0_bn3_running_mean, b_stage3_0_bn3_running_var, p_stage3_0_bn3_weight, p_stage3_0_bn3_bias,
        shortcut_conv_w=p_stage3_0_shortcut_0_weight,
        shortcut_bn_mean=b_stage3_0_shortcut_1_running_mean,
        shortcut_bn_var=b_stage3_0_shortcut_1_running_var,
        shortcut_bn_w=p_stage3_0_shortcut_1_weight,
        shortcut_bn_b=p_stage3_0_shortcut_1_bias,
        groups=3,
    )

    # Stage 3, blocks 1-6
    for conv1_w, bn1_m, bn1_v, bn1_w, bn1_b, conv2_w, bn2_m, bn2_v, bn2_w, bn2_b, conv3_w, bn3_m, bn3_v, bn3_w, bn3_b in [
        (p_stage3_1_conv1_weight, b_stage3_1_bn1_running_mean, b_stage3_1_bn1_running_var, p_stage3_1_bn1_weight, p_stage3_1_bn1_bias,
         p_stage3_1_conv2_weight, b_stage3_1_bn2_running_mean, b_stage3_1_bn2_running_var, p_stage3_1_bn2_weight, p_stage3_1_bn2_bias,