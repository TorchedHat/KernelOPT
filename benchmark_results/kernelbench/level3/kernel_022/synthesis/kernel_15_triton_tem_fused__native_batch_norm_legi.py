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
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    ACTIVATION: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # channel index: layout is (N, C, H, W) -> channel = (offset / spatial) % num_channels
    chan = (offsets // spatial) % num_channels
    chan_mask = chan < num_channels

    mean = tl.load(mean_ptr + chan, mask=mask & chan_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + chan, mask=mask & chan_mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + chan, mask=mask & chan_mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + chan, mask=mask & chan_mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    if ACTIVATION == 1:  # ReLU
        out = tl.maximum(out, 0.0)
    elif ACTIVATION == 2:  # HardTanh [-6, 6]
        out = tl.minimum(tl.maximum(out, -6.0), 6.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def apply_bn_act(x, mean, var, weight, bias, eps=1e-5, activation=0):
    """Apply batch norm + optional activation in one Triton kernel."""
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, num_channels, spatial,
        eps,
        activation,
    )
    return out


def conv2d_bn_act(x, conv_weight, mean, var, bn_weight, bn_bias,
                  stride=1, padding=0, groups=1, activation=0, eps=1e-5):
    """Conv2d + BN + activation fused (conv via cuDNN, BN+act via Triton)."""
    y = F.conv2d(x, conv_weight, bias=None, stride=stride, padding=padding, groups=groups)
    return apply_bn_act(y, mean, var, bn_weight, bn_bias, eps=eps, activation=activation)


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
    # RELU=1, HARDTANH=2, NONE=0
    RELU = 1
    HARDTANH = 2
    NONE = 0

    # Initial conv + BN + ReLU
    x = conv2d_bn_act(x, p_conv1_weight, b_bn1_running_mean, b_bn1_running_var,
                      p_bn1_weight, p_bn1_bias, stride=2, padding=1, activation=RELU)

    # Block 0: MBConv1 (32->16, k=3, s=1, expand=1)
    # depthwise conv + BN + HardTanh
    x0 = conv2d_bn_act(x, p_blocks_0_depthwise_conv_0_weight,
                       b_blocks_0_depthwise_conv_1_running_mean,
                       b_blocks_0_depthwise_conv_1_running_var,
                       p_blocks_0_depthwise_conv_1_weight,
                       p_blocks_0_depthwise_conv_1_bias,
                       stride=1, padding=1, groups=32, activation=HARDTANH)
    # project conv + BN (no activation)
    x = conv2d_bn_act(x0, p_blocks_0_project_conv_0_weight,
                      b_blocks_0_project_conv_1_running_mean,
                      b_blocks_0_project_conv_1_running_var,
                      p_blocks_0_project_conv_1_weight,
                      p_blocks_0_project_conv_1_bias,
                      stride=1, padding=0, activation=NONE)

    # Block 1: MBConv6 (16->24, k=3, s=2, expand=6)
    # expand conv + BN + HardTanh
    x1 = conv2d_bn_act(x, p_blocks_1_expand_conv_0_weight,
                       b_blocks_1_expand_conv_1_running_mean,
                       b_blocks_1_expand_conv_1_running_var,
                       p_blocks_1_expand_conv_1_weight,
                       p_blocks_1_expand_conv_1_bias,
                       stride=1, padding=0, activation=HARDTANH)
    # depthwise conv + BN + HardTanh
    x1 = conv2d_bn_act(x1, p_blocks_1_depthwise_conv_0_weight,
                       b_blocks_1_depthwise_conv_1_running_mean,
                       b_blocks_1_depthwise_conv_1_running_var,
                       p_blocks_1_depthwise_conv_1_weight,
                       p_blocks_1_depthwise_conv_1_bias,
                       stride=2, padding=1, groups=96, activation=HARDTANH)
    # project conv + BN (no activation)
    x = conv2d_bn_act(x1, p_blocks_1_project_conv_0_weight,
                      b_blocks_1_project_conv_1_running_mean,
                      b_blocks_1_project_conv_1_running_var,
                      p_blocks_1_project_conv_1_weight,
                      p_blocks_1_project_conv_1_bias,
                      stride=1, padding=0, activation=NONE)

    # Block 2: MBConv6 (24->24, k=3, s=1, expand=6) with residual
    residual = x
    x2 = conv2d_bn_act(x, p_blocks_2_expand_conv_0_weight,
                       b_blocks_2_expand_conv_1_running_mean,
                       b_blocks_2_expand_conv_1_running_var,
                       p_blocks_2_expand_conv_1_weight,
                       p_blocks_2_expand_conv_1_bias,
                       stride=1, padding=0, activation=HARDTANH)
    x2 = conv2d_bn_act(x2, p_blocks_2_depthwise_conv_0_weight,
                       b_blocks_2_depthwise_conv_1_running_mean,
                       b_blocks_2_depthwise_conv_1_running_var,
                       p_blocks_2_depthwise_conv_1_weight,
                       p_blocks_2_depthwise_conv_1_bias,
                       stride=1, padding=1, groups=144, activation=HARDTANH)
    x2 = conv2d_bn_act(x2, p_blocks_2_project_conv_0_weight,
                       b_blocks_2_project_conv_1_running_mean,
                       b_blocks_2_project_conv_1_running_var,
                       p_blocks_2_project_conv_1_weight,
                       p_blocks_2_project_conv_1_bias,
                       stride=1, padding=0, activation=NONE)
    x = residual + x2

    # Block 3: MBConv6 (24->40, k=5, s=2, expand=6)
    x3 = conv2d_bn_act(x, p_blocks_3_expand_conv_0_weight,
                       b_blocks_3_expand_conv_1_running_mean,
                       b_blocks_3_expand_conv_1_running_var,
                       p_blocks_3_expand_conv_1_weight,
                       p_blocks_3_expand_conv_1_bias,
                       stride=1, padding=0, activation=HARDTANH)
    x3 = conv2d_bn_act(x3, p_blocks_3_depthwise_conv_0_weight,
                       b_blocks_3_depthwise_conv_1_running_mean,
                       b_blocks_3_depthwise_conv_1_running_var,
                       p_blocks_3_depthwise_conv_1_weight,
                       p_blocks_3_depthwise_conv_1_bias,
                       stride=2, padding=2, groups=144, activation=HARDTANH)
    x = conv2d_bn_act(x3, p_blocks_3_project_conv_0_weight,
                      b_blocks_3_project_conv_1_running_mean,
                      b_blocks_3_project_conv_1_running_var,
                      p_blocks_3_project_conv_1_weight,
                      p_blocks_3_project_conv_1_bias,
                      stride=1, padding=0, activation=NONE)

    # Block 4: MBConv6 (40->40, k=5, s=1, expand=6) with residual
    residual = x
    x4 = conv2d_bn_act(x, p_blocks_4_expand_conv_0_weight,
                       b_blocks_4_expand_conv_1_running_mean,
                       b_blocks_4_expand_conv_1_running_var,
                       p_blocks_4_expand_conv_1_weight,
                       p_blocks_4_expand_conv_1_bias,
                       stride=1, padding=0, activation=HARDTANH)
    x4 = conv2d_bn_act(x4, p_blocks_4_depthwise_conv_0_weight,
                       b_blocks_4_depthwise_conv_1_running_mean,
                       b_blocks_4_depthwise_conv_1_running_var,
                       p_blocks_4_depthwise_conv_1_weight,
                       p_blocks_4_depthwise_conv_1_bias,
                       stride=1, padding=2, groups=240, activation=HARDTANH)
    x4 = conv2d_bn_act(x4, p_blocks_4_project_conv_0_weight,
                       b_blocks_4_project_conv_1_running_mean,
                       b_blocks_4_project_conv_1_running_var,
                       p_blocks_4_project_conv_1_weight,
                       p_blocks_4_project_conv_1_bias,
                       stride=1, padding=0, activation=NONE)
    x = residual + x4

    # Block 5: MBConv6 (40->80, k=3, s=2, expand=6)
    x5 = conv2d_bn_act(x, p_blocks_5_expand_conv_0_weight,
                       b_blocks_5_expand_conv_1_running_mean,
                       b_blocks_5_expand_conv_1_running_var,
                       p_blocks_5_expand_conv_1_weight,
                       p_blocks_5_expand_conv_1_bias,
                       stride=1, padding=0, activation=HARDTANH)
    x5 = conv2d_bn_act(x5, p_blocks_5_depthwise_conv_0_weight,
                       b_blocks_5_depthwise_conv_1_running_mean,
                       b_blocks_5_depthwise_conv_1_running_var,
                       p_blocks_5_depthwise_conv_1_weight,
                       p_blocks_5_depthwise_conv_1_bias,
                       stride=2, padding=1, groups=240, activation=HARDTANH)
    x = conv2d_bn_act(x5, p_blocks_5_project_conv_0_weight,
                      b_blocks_5_project_conv_1_running_mean,
                      b_blocks_5_project_conv_1_running_var,
                      p_blocks_5_project_conv_1_weight,
                      p_blocks_5_project_conv_1_bias,
                      stride=1, padding=0, activation=NONE)

    # Block 6: MBConv6 (80->80, k=3, s=1, expand=6) with residual
    residual = x
    x6 = conv2d_bn_act(x, p_blocks_6_expand_conv_0_weight,
                       b_blocks_6_expand_conv_1_running_mean,
                       b_blocks_6_expand_conv_1_running_var,
                       p_blocks_6_expand_conv_1_weight,
                       p_blocks_6_expand_conv_1_bias,
                       stride=1, padding=0, activation=HARDTANH)
    x6 = conv2d_bn_act(x6, p_blocks_6_depthwise_conv_0_weight,
                       b_blocks_6_depthwise_conv_1_running_mean,
                       b_blocks_6_depthwise_conv_1_running_var,
                       p_blocks_6_depthwise_conv_1_weight,
                       p_blocks_6_depthwise_conv_1_bias,
                       stride=1, padding=1, groups=480, activation=HARDTANH)
    x6 = conv2d_bn_act(x6, p_blocks_6_project_conv_0_weight,
                       b_blocks_6_project_conv_1_running_mean,
                       b_blocks_6_project_conv_1_running_var,
                       p_blocks_6_project_conv_1_weight,
                       p_blocks_6_project_conv_1_bias,
                       stride=1, padding=0, activation=NONE)
    x = residual + x6

    # Block 7: MBConv6 (80->112, k=5, s=1, expand=6)
    x7 = conv2d_bn_act(x, p_blocks_7_expand_conv_0_weight,
                       b_blocks_7_expand_conv_1_running_mean,
                       b_blocks_7_expand_conv_1_running_var,
                       p_blocks_7_expand_conv_1_weight,
                       p_blocks_7_expand_conv_1_bias,
                       stride=1, padding=0, activation=HARDTANH)
    x7 = conv2d_bn_act(x7, p_blocks_7_depthwise_conv_0_weight,
                       b_blocks_7_depthwise_conv_1_running_mean,
                       b_blocks_7_depthwise_conv_1_running_var,
                       p_blocks_7_depthwise_conv_1_weight,
                       p_blocks_7_depthwise_conv_1_bias,
                       stride=1, padding=2, groups=480, activation=HARDTANH)
    x = conv2d_bn_act(x7, p_blocks_7_project_conv_0_weight,
                      b_blocks_7_project_conv_1_running_mean,
                      b_blocks_7_project_conv_1_running_var,
                      p_blocks_7_project_conv_1_weight,
                      p_blocks_7_project_conv_1_bias,
                      stride=1, padding=0, activation=NONE)

    # Block 8: MBConv6 (112->112, k=5, s=1, expand=6) with residual
    residual = x
    x8 = conv2d_bn_act(x, p_blocks_8_expand_conv_0_weight,
                       b_blocks_8_expand_conv_1_running_mean,
                       b_blocks_8_expand_conv_1_running_var,
                       p_blocks_8_expand_conv_1_weight,
                       p_blocks_8_expand_conv_1_bias,
                       stride=1, padding=0, activation=HARDTANH)
    x8 = conv2d_bn_act(x8, p_blocks_8_depthwise_conv_0_weight,
                       b_blocks_8_depthwise_conv_1_running_mean,
                       b_blocks_8_depthwise_conv_1_running_var,
                       p_blocks_8_depthwise_conv_1_weight,
                       p_blocks_8_depthwise_conv_1_bias,
                       stride=1, padding=2, groups=672, activation=HARDTANH)
    x8 = conv2d_bn_act(x8, p_blocks_8_project_conv_0_weight,
                       b_blocks_8_project_conv_1_running_mean,
                       b_blocks_8_project_conv_1_running_var,
                       p_blocks_8_project_conv_1_weight,
                       p_blocks_8_project_conv_1_bias,
                       stride=1, padding=0, activation=NONE)
    x = residual + x8

    # Block 9: MBConv6