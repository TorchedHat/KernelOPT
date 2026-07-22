import torch
import triton
import triton.language as tl

# Fused BatchNorm (inference) + ReLU6 (hardtanh(0,6)) kernel
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
def fused_bn_relu6_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    numel,
    num_channels,
    eps,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Each element belongs to channel = (offset // spatial_size) % num_channels
    # But we store in NCHW: offset = n * C * H * W + c * H * W + h * W + w
    # channel index = (offset // (H*W)) % C
    # We compute channel from offset and num_channels
    # spatial_size = numel / (batch * num_channels) but we don't have batch separately
    # Instead: channel = (offset // (numel // (numel // num_channels))) % num_channels
    # Simpler: for NCHW layout, channel = (offset // spatial_hw) % C
    # spatial_hw = numel / (N * C), but we can compute:
    # total elements = N * C * H * W
    # channel index in flat NCHW = (offset // (H*W)) % C
    # We pass spatial_hw as numel // (N * C) but we don't have N separately
    # Actually: channel = (offset // spatial_hw) % num_channels
    # We'll compute spatial_hw = numel // num_channels (wrong if batch > 1)
    # Correct: channel index = (offset // spatial_hw) % num_channels
    # where spatial_hw = H * W
    # We need to pass spatial_hw separately

    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index
    # For NCHW: flat_idx = n*(C*H*W) + c*(H*W) + h*W + w
    # channel = (flat_idx // (H*W)) % C
    # We pass spatial_hw
    # Actually let's just use: channel = (offset // spatial_hw) % num_channels
    # But we don't have spatial_hw here... let's add it as parameter

    # Simpler approach: channel = (offset // (numel // num_channels_total)) % num_channels
    # This doesn't work for batched...
    # Let's just pass spatial_hw explicitly

    # We'll restructure to pass spatial_hw
    x = x  # placeholder, restructure below

    tl.store(out_ptr + offsets, x, mask=mask)


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
def fused_bn_relu6_kernel_v2(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    numel,
    num_channels,
    spatial_hw,
    eps,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # Compute channel index for NCHW layout
    c = (offsets // spatial_hw) % num_channels

    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

    # Batch norm: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def apply_bn_relu6(x, mean, var, weight, bias, eps=1e-5, apply_relu6=True):
    """Apply batch norm + optional ReLU6 using Triton kernel."""
    N, C, H, W = x.shape
    numel = x.numel()
    spatial_hw = H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu6_kernel_v2[grid](
        x, out,
        mean, var, weight, bias,
        numel, C, spatial_hw, eps,
        apply_relu6,
    )
    return out


def conv2d_op(x, weight, stride=1, padding=0, groups=1):
    """Wrapper for conv2d using PyTorch (cuDNN)."""
    import torch.nn.functional as F
    return F.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


def kernel_function(
    p_features_0_weight,
    p_features_1_weight,
    p_features_1_bias,
    p_features_3_0_weight,
    p_features_3_1_weight,
    p_features_3_1_bias,
    p_features_3_3_weight,
    p_features_3_4_weight,
    p_features_3_4_bias,
    p_features_4_0_weight,
    p_features_4_1_weight,
    p_features_4_1_bias,
    p_features_4_3_weight,
    p_features_4_4_weight,
    p_features_4_4_bias,
    p_features_4_6_weight,
    p_features_4_7_weight,
    p_features_4_7_bias,
    p_features_5_0_weight,
    p_features_5_1_weight,
    p_features_5_1_bias,
    p_features_5_3_weight,
    p_features_5_4_weight,
    p_features_5_4_bias,
    p_features_5_6_weight,
    p_features_5_7_weight,
    p_features_5_7_bias,
    p_features_6_0_weight,
    p_features_6_1_weight,
    p_features_6_1_bias,
    p_features_6_3_weight,
    p_features_6_4_weight,
    p_features_6_4_bias,
    p_features_6_6_weight,
    p_features_6_7_weight,
    p_features_6_7_bias,
    p_features_7_0_weight,
    p_features_7_1_weight,
    p_features_7_1_bias,
    p_features_7_3_weight,
    p_features_7_4_weight,
    p_features_7_4_bias,
    p_features_7_6_weight,
    p_features_7_7_weight,
    p_features_7_7_bias,
    p_features_8_0_weight,
    p_features_8_1_weight,
    p_features_8_1_bias,
    p_features_8_3_weight,
    p_features_8_4_weight,
    p_features_8_4_bias,
    p_features_8_6_weight,
    p_features_8_7_weight,
    p_features_8_7_bias,
    p_features_9_0_weight,
    p_features_9_1_weight,
    p_features_9_1_bias,
    p_features_9_3_weight,
    p_features_9_4_weight,
    p_features_9_4_bias,
    p_features_9_6_weight,
    p_features_9_7_weight,
    p_features_9_7_bias,
    p_features_10_0_weight,
    p_features_10_1_weight,
    p_features_10_1_bias,
    p_features_10_3_weight,
    p_features_10_4_weight,
    p_features_10_4_bias,
    p_features_10_6_weight,
    p_features_10_7_weight,
    p_features_10_7_bias,
    p_features_11_0_weight,
    p_features_11_1_weight,
    p_features_11_1_bias,
    p_features_11_3_weight,
    p_features_11_4_weight,
    p_features_11_4_bias,
    p_features_11_6_weight,
    p_features_11_7_weight,
    p_features_11_7_bias,
    p_features_12_0_weight,
    p_features_12_1_weight,
    p_features_12_1_bias,
    p_features_12_3_weight,
    p_features_12_4_weight,
    p_features_12_4_bias,
    p_features_12_6_weight,
    p_features_12_7_weight,
    p_features_12_7_bias,
    p_features_13_0_weight,
    p_features_13_1_weight,
    p_features_13_1_bias,
    p_features_13_3_weight,
    p_features_13_4_weight,
    p_features_13_4_bias,
    p_features_13_6_weight,
    p_features_13_7_weight,
    p_features_13_7_bias,
    p_features_14_0_weight,
    p_features_14_1_weight,
    p_features_14_1_bias,
    p_features_14_3_weight,
    p_features_14_4_weight,
    p_features_14_4_bias,
    p_features_14_6_weight,
    p_features_14_7_weight,
    p_features_14_7_bias,
    p_features_15_0_weight,
    p_features_15_1_weight,
    p_features_15_1_bias,
    p_features_15_3_weight,
    p_features_15_4_weight,
    p_features_15_4_bias,
    p_features_15_6_weight,
    p_features_15_7_weight,
    p_features_15_7_bias,
    p_features_16_0_weight,
    p_features_16_1_weight,
    p_features_16_1_bias,
    p_features_16_3_weight,
    p_features_16_4_weight,
    p_features_16_4_bias,
    p_features_16_6_weight,
    p_features_16_7_weight,
    p_features_16_7_bias,
    p_features_17_0_weight,
    p_features_17_1_weight,
    p_features_17_1_bias,
    p_features_17_3_weight,
    p_features_17_4_weight,
    p_features_17_4_bias,
    p_features_17_6_weight,
    p_features_17_7_weight,
    p_features_17_7_bias,
    p_features_18_0_weight,
    p_features_18_1_weight,
    p_features_18_1_bias,
    p_features_18_3_weight,
    p_features_18_4_weight,
    p_features_18_4_bias,
    p_features_18_6_weight,
    p_features_18_7_weight,
    p_features_18_7_bias,
    p_features_19_0_weight,
    p_features_19_1_weight,
    p_features_19_1_bias,
    p_features_19_3_weight,
    p_features_19_4_weight,
    p_features_19_4_bias,
    p_features_19_6_weight,
    p_features_19_7_weight,
    p_features_19_7_bias,
    p_features_20_weight,
    p_features_21_weight,
    p_features_21_bias,
    p_classifier_1_weight,
    p_classifier_1_bias,
    b_features_1_running_mean,
    b_features_1_running_var,
    b_features_3_1_running_mean,
    b_features_3_1_running_var,
    b_features_3_4_running_mean,
    b_features_3_4_running_var,
    b_features_4_1_running_mean,
    b_features_4_1_running_var,
    b_features_4_4_running_mean,
    b_features_4_4_running_var,
    b_features_4_7_running_mean,
    b_features_4_7_running_var,
    b_features_5_1_running_mean,
    b_features_5_1_running_var,
    b_features_5_4_running_mean,
    b_features_5_4_running_var,
    b_features_5_7_running_mean,
    b_features_5_7_running_var,
    b_features_6_1_running_mean,
    b_features_6_1_running_var,
    b_features_6_4_running_mean,
    b_features_6_4_running_var,
    b_features_6_7_running_mean,
    b_features_6_7_running_var,
    b_features_7_1_running_mean,
    b_features_7_1_running_var,
    b_features_7_4_running_mean,
    b_features_7_4_running_var,
    b_features_7_7_running_mean,
    b_features_7_7_running_var,
    b_features_8_1_running_mean,
    b_features_8_1_running_var,
    b_features_8_4_running_mean,
    b_features_8_4_running_var,
    b_features_8_7_running_mean,
    b_features_8_7_running_var,
    b_features_9_1_running_mean,
    b_features_9_1_running_var,
    b_features_9_4_running_mean,
    b_features_9_4_running_var,
    b_features_9_7_running_mean,
    b_features_9_7_running_var,
    b_features_10_1_running_mean,
    b_features_10_1_running_var,
    b_features_10_4_running_mean,
    b_features_10_4_running_var,
    b_features_10_7_running_mean,
    b_features_10_7_running_var,
    b_features_11_1_running_mean,
    b_features_11_1_running_var,
    b_features_11_4_running_mean,
    b_features_11_4_running_var,
    b_features_11_7_running_mean,
    b_features_11_7_running_var,
    b_features_12_1_running_mean,
    b_features_12_1_running_var,
    b_features_12_4_running_mean,
    b_features_12_4_running_var,
    b_features_12_7_running_mean,
    b_features_12_7_running_var,
    b_features_13_1_running_mean,
    b_features_13_1_running_var,
    b_features_13_4_running_mean,
    b_features_13_4_running_var,
    b_features_13_7_running_mean,
    b_features_13_7_running_var,
    b_features_14_1_running_mean,
    b_features_14_1_running_var,
    b_features_14_4_running_mean,
    b_features_14_4_running_var,
    b_features_14_7_running_mean,
    b_features_14_7_running_var,
    b_features_15_1_running_mean,
    b_features_15_1_running_var,
    b_features_15_4_running_mean,
    b_features_15_4_running_var,
    b_features_15_7_running_mean,
    b_features_15_7_running_var,
    b_features_16_1_running_mean,
    b_features_16_1_running_var,
    b_features_16_4_running_mean,
    b_features_16_4_running_var,
    b_features_16_7_running_mean,
    b_features_16_7_running_var,
    b_features_17_1_running_mean,
    b_features_17_1_running_var,
    b_features_17_4_running_mean,
    b_features_17_4_running_var,
    b_features_17_7_running_mean,
    b_features_17_7_running_var,
    b_features_18_1_running_mean,
    b_features_18_1_running_var,
    b_features_18_4_running_mean,
    b_features_18_4_running_var,
    b_features_18_7_running_mean,
    b_features_18_7_running_var,
    b_features_19_1_running_mean,
    b_features_19_1_running_var,
    b_features_19_4_running_mean,
    b_features_19_4_running_var,
    b_features_19_7_running_mean,
    b_features_19_7_running_var,
    b_features_21_running_mean,
    b_features_21_running_var,
    x,
):
    import torch.nn.functional as F

    # features[0]: Conv2d(3, 32, 3, stride=2, padding=1)
    out = F.conv2d(x, p_features_0_weight, bias=None, stride=2, padding=1)
    # features[1]: BN + ReLU6
    out = apply_bn_relu6(out, b_features_1_running_mean, b_features_1_running_var,
                         p_features_1_weight, p_features_1_bias, apply_relu6=True)

    # features[3]: InvertedResidual t=1, c=16, n=1, s=1
    # No expand (t=1), depthwise conv, BN+ReLU6, pointwise, BN
    # Depthwise conv (groups=32)
    out = F.conv2d(out, p_features_3_0_weight, bias=None, stride=1, padding=1, groups=32)
    out = apply_bn_relu6(out, b_features_3_1_running_mean, b_features_3_1_running_var,
                         p_features_3_1_weight, p_features_3_1_bias, apply_relu6=True)
    # Pointwise conv
    out = F.conv2d(out, p_features_3_3_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_3_4_running_mean, b_features_3_4_running_var,
                         p_features_3_4_weight, p_features_3_4_bias, apply_relu6=False)

    # features[4]: InvertedResidual t=6, c=24, n=2, s=2 (first block, stride=2)
    # Expand: 16 -> 96
    out = F.conv2d(out, p_features_4_0_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_4_1_running_mean, b_features_4_1_running_var,
                         p_features_4_1_weight, p_features_4_1_bias, apply_relu6=True)
    # Depthwise stride=2
    out = F.conv2d(out, p_features_4_3_weight, bias=None, stride=2, padding=1, groups=96)
    out = apply_bn_relu6(out, b_features_4_4_running_mean, b_features_4_4_running_var,
                         p_features_4_4_weight, p_features_4_4_bias, apply_relu6=True)
    # Pointwise: 96 -> 24
    out = F.conv2d(out, p_features_4_6_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_4_7_running_mean, b_features_4_7_running_var,
                         p_features_4_7_weight, p_features_4_7_bias, apply_relu6=False)

    # features[5]: InvertedResidual t=6, c=24, n=2, s=2 (second block, stride=1, residual)
    residual = out
    # Expand: 24 -> 144
    out = F.conv2d(out, p_features_5_0_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_5_1_running_mean, b_features_5_1_running_var,
                         p_features_5_1_weight, p_features_5_1_bias, apply_relu6=True)
    # Depthwise stride=1
    out = F.conv2d(out, p_features_5_3_weight, bias=None, stride=1, padding=1, groups=144)
    out = apply_bn_relu6(out, b_features_5_4_running_mean, b_features_5_4_running_var,
                         p_features_5_4_weight, p_features_5_4_bias, apply_relu6=True)
    # Pointwise: 144 -> 24
    out = F.conv2d(out, p_features_5_6_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_5_7_running_mean, b_features_5_7_running_var,
                         p_features_5_7_weight, p_features_5_7_bias, apply_relu6=False)
    out = out + residual

    # features[6]: InvertedResidual t=6, c=32, n=3, s=2 (first block, stride=2)
    # Expand: 24 -> 144
    out = F.conv2d(out, p_features_6_0_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_6_1_running_mean, b_features_6_1_running_var,
                         p_features_6_1_weight, p_features_6_1_bias, apply_relu6=True)
    # Depthwise stride=2
    out = F.conv2d(out, p_features_6_3_weight, bias=None, stride=2, padding=1, groups=144)
    out = apply_bn_relu6(out, b_features_6_4_running_mean, b_features_6_4_running_var,
                         p_features_6_4_weight, p_features_6_4_bias, apply_relu6=True)
    # Pointwise: 144 -> 32
    out = F.conv2d(out, p_features_6_6_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_6_7_running_mean, b_features_6_7_running_var,
                         p_features_6_7_weight, p_features_6_7_bias, apply_relu6=False)

    # features[7]: InvertedResidual t=6, c=32, n=3, s=2 (second block, stride=1, residual)
    residual = out
    # Expand: 32 -> 192
    out = F.conv2d(out, p_features_7_0_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_7_1_running_mean, b_features_7_1_running_var,
                         p_features_7_1_weight, p_features_7_1_bias, apply_relu6=True)
    # Depthwise stride=1
    out = F.conv2d(out, p_features_7_3_weight, bias=None, stride=1, padding=1, groups=192)
    out = apply_bn_relu6(out, b_features_7_4_running_mean, b_features_7_4_running_var,
                         p_features_7_4_weight, p_features_7_4_bias, apply_relu6=True)
    # Pointwise: 192 -> 32
    out = F.conv2d(out, p_features_7_6_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_7_7_running_mean, b_features_7_7_running_var,
                         p_features_7_7_weight, p_features_7_7_bias, apply_relu6=False)
    out = out + residual

    # features[8]: InvertedResidual t=6, c=32, n=3, s=2 (third block, stride=1, residual)
    residual = out
    # Expand: 32 -> 192
    out = F.conv2d(out, p_features_8_0_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_8_1_running_mean, b_features_8_1_running_var,
                         p_features_8_1_weight, p_features_8_1_bias, apply_relu6=True)
    # Depthwise stride=1
    out = F.conv2d(out, p_features_8_3_weight, bias=None, stride=1, padding=1, groups=192)
    out = apply_bn_relu6(out, b_features_8_4_running_mean, b_features_8_4_running_var,
                         p_features_8_4_weight, p_features_8_4_bias, apply_relu6=True)
    # Pointwise: 192 -> 32
    out = F.conv2d(out, p_features_8_6_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_8_7_running_mean, b_features_8_7_running_var,
                         p_features_8_7_weight, p_features_8_7_bias, apply_relu6=False)
    out = out + residual

    # features[9]: InvertedResidual t=6, c=64, n=4, s=2 (first block, stride=2)
    # Expand: 32 -> 192
    out = F.conv2d(out, p_features_9_0_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_9_1_running_mean, b_features_9_1_running_var,
                         p_features_9_1_weight, p_features_9_1_bias, apply_relu6=True)
    # Depthwise stride=2
    out = F.conv2d(out, p_features_9_3_weight, bias=None, stride=2, padding=1, groups=192)
    out = apply_bn_relu6(out, b_features_9_4_running_mean, b_features_9_4_running_var,
                         p_features_9_4_weight, p_features_9_4_bias, apply_relu6=True)
    # Pointwise: 192 -> 64
    out = F.conv2d(out, p_features_9_6_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_9_7_running_mean, b_features_9_7_running_var,
                         p_features_9_7_weight, p_features_9_7_bias, apply_relu6=False)

    # features[10]: InvertedResidual t=6, c=64, n=4, s=2 (second block, stride=1, residual)
    residual = out
    # Expand: 64 -> 384
    out = F.conv2d(out, p_features_10_0_weight, bias=None, stride=1, padding=0)
    out = apply_bn_relu6(out, b_features_10_1_running_mean, b_features_10_1_running_var,
                         p_features_10_1_weight, p_features_10_1_bias, apply_relu6=True)
    # Depthwise stride=1
    out = F.conv2d(out, p_features_10_3_weight, bias=None, stride=1, padding=1, groups=384)
    out = apply_bn_relu6(out, b_features_10_4_running_mean,