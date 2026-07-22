import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm + Activation kernel
# activation: 0=none, 1=relu, 2=hardtanh6
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
def fused_bn_act_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels,
    eps,
    activation: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # channel index: layout is (N, C, H, W) so channel = (offset // (H*W)) % C
    # but we pass spatial_size = H*W separately
    # Actually we compute channel from offset and num_channels
    # offset layout: offset = n * C * HW + c * HW + hw
    # so c = (offset // HW) % C, but we don't have HW directly
    # Instead: c = (offset // (numel // (numel // num_channels))) ... complex
    # Simpler: channel_stride = numel / num_channels (= N*H*W per channel? No)
    # numel = N*C*H*W, channel index = (offset / (H*W)) % C
    # We'll pass spatial_size = H*W
    # Actually let's just use: c = (offset % (num_channels * spatial)) // spatial
    # We pass num_channels and compute from offset
    # offset = n*(C*HW) + c*HW + hw => c = (offset // HW) % C
    # We need HW. Let's pass it as numel // (N*C) but we don't have N separately.
    # Simplest: pass channel_stride = numel // num_channels (= N*H*W)
    # Then c = (offset // channel_stride) ... no that's wrong too
    # offset = n*C*HW + c*HW + hw
    # offset // HW = n*C + c => (offset // HW) % C = c
    # So we need HW. Let's add it as a parameter.
    # We'll restructure - see wrapper below
    c = (offsets // tl.load(tl.make_block_ptr(var_ptr, [1], [1], [0], [1], [0]))) 
    # This approach is getting complex. Let me use a simpler parameterization.
    pass


# Simpler approach: pass spatial_size directly
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
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // spatial_size) % 65536  # channel index, bounded
    # Load per-channel params
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
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
def fused_bn_hardtanh6_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // spatial_size) % 65536
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.minimum(tl.maximum(out, 0.0), 6.0)
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
def fused_bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, spatial_size,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // spatial_size) % 65536
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    tl.store(out_ptr + offsets, out, mask=mask)


def bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    spatial_size = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, spatial_size,
        eps,
    )
    return out


def bn_hardtanh6(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    spatial_size = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_hardtanh6_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, spatial_size,
        eps,
    )
    return out


def bn_only(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    spatial_size = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, spatial_size,
        eps,
    )
    return out


def conv2d_op(x, weight, stride=1, padding=0, groups=1):
    return F.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


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
    # Initial conv + BN + ReLU
    t = conv2d_op(x, p_conv1_weight, stride=2, padding=1)
    t = bn_relu(
        t, p_bn1_weight, p_bn1_bias,
        b_bn1_running_mean, b_bn1_running_var,
    )

    # Block 0: MBConv1 (32->16, k=3, s=1, expand=1)
    # depthwise conv (groups=32)
    t = conv2d_op(t, p_blocks_0_depthwise_conv_0_weight, stride=1, padding=1, groups=32)
    t = bn_hardtanh6(
        t, p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
        b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
    )
    t = conv2d_op(t, p_blocks_0_project_conv_0_weight, stride=1, padding=0)
    t = bn_only(
        t, p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
        b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
    )

    # Block 1: MBConv6 (16->24, k=3, s=2, expand=6)
    t1 = conv2d_op(t, p_blocks_1_expand_conv_0_weight, stride=1, padding=0)
    t1 = bn_hardtanh6(
        t1, p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
        b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
    )
    t1 = conv2d_op(t1, p_blocks_1_depthwise_conv_0_weight, stride=2, padding=1, groups=96)
    t1 = bn_hardtanh6(
        t1, p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
        b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
    )
    t1 = conv2d_op(t1, p_blocks_1_project_conv_0_weight, stride=1, padding=0)
    t1 = bn_only(
        t1, p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
        b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
    )
    t = t1

    # Block 2: MBConv6 (24->24, k=3, s=1, expand=6) with residual
    residual = t
    t2 = conv2d_op(t, p_blocks_2_expand_conv_0_weight, stride=1, padding=0)
    t2 = bn_hardtanh6(
        t2, p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
        b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
    )
    t2 = conv2d_op(t2, p_blocks_2_depthwise_conv_0_weight, stride=1, padding=1, groups=144)
    t2 = bn_hardtanh6(
        t2, p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
        b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
    )
    t2 = conv2d_op(t2, p_blocks_2_project_conv_0_weight, stride=1, padding=0)
    t2 = bn_only(
        t2, p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
        b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
    )
    t2 = t2 + residual
    t = t2

    # Block 3: MBConv6 (24->40, k=5, s=2, expand=6)
    t3 = conv2d_op(t, p_blocks_3_expand_conv_0_weight, stride=1, padding=0)
    t3 = bn_hardtanh6(
        t3, p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
        b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
    )
    t3 = conv2d_op(t3, p_blocks_3_depthwise_conv_0_weight, stride=2, padding=2, groups=144)
    t3 = bn_hardtanh6(
        t3, p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
        b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
    )
    t3 = conv2d_op(t3, p_blocks_3_project_conv_0_weight, stride=1, padding=0)
    t3 = bn_only(
        t3, p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
        b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
    )
    t = t3

    # Block 4: MBConv6 (40->40, k=5, s=1, expand=6) with residual
    residual = t
    t4 = conv2d_op(t, p_blocks_4_expand_conv_0_weight, stride=1, padding=0)
    t4 = bn_hardtanh6(
        t4, p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
        b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
    )
    t4 = conv2d_op(t4, p_blocks_4_depthwise_conv_0_weight, stride=1, padding=2, groups=240)
    t4 = bn_hardtanh6(
        t4, p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
        b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var,
    )
    t4 = conv2d_op(t4, p_blocks_4_project_conv_0_weight, stride=1, padding=0)
    t4 = bn_only(
        t4, p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,
        b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var,
    )
    t4 = t4 + residual
    t = t4

    # Block 5: MBConv6 (40->80, k=3, s=2, expand=6)
    t5 = conv2d_op(t, p_blocks_5_expand_conv