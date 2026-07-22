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
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, num_channels,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # channel index: layout is (N, C, H, W) contiguous
    # offsets = n * C*H*W + c * H*W + hw
    # c = (offsets // spatial) % C, but we need spatial = numel / (N*C)
    # Actually for contiguous NCHW: offset = n*(C*H*W) + c*(H*W) + h*W + w
    # c = (offset % (C*H*W)) // (H*W)
    # We pass num_channels and compute spatial = numel / num_channels (not exact per batch)
    # Better: c = (offsets // (numel // num_channels)) % num_channels
    # Actually numel = N*C*H*W, spatial_per_channel = numel // num_channels = N*H*W
    # c index: offsets // spatial_per_channel gives n*C + c_within... no
    # For NCHW contiguous: linear_idx = n*C*H*W + c*H*W + h*W + w
    # c = (linear_idx // (H*W)) % C
    # H*W = numel // (N*C) -- we don't have N separately
    # Let's pass hw_size = H*W directly
    # We'll use a different approach: c = (offsets // hw_size) % num_channels
    # But we don't have hw_size here... let's restructure

    # Actually we pass numel = N*C*H*W and num_channels = C
    # spatial = numel / num_channels = N*H*W (not H*W)
    # For NCHW: c = (offset // (H*W)) % C
    # We need H*W. Let's just pass it as a parameter.
    # For now use: c_idx = (offsets // (numel // num_channels)) % num_channels
    # numel // num_channels = N*H*W, so (offset // (N*H*W)) = n... that's wrong
    # 
    # Correct for NCHW contiguous:
    # offset = n*C*H*W + c*H*W + spatial_idx
    # c = (offset // hw) % C where hw = H*W
    # We need hw. Let's just pass total / (N*C) but we don't have N...
    # 
    # Simplest: pass hw_size separately
    # But autotune key doesn't include it... let's just use num_channels and numel
    # and compute hw = numel // num_channels // batch... 
    # Actually the cleanest: c_idx = (offsets % (num_channels * (numel // num_channels))) // (numel // num_channels)
    # = offsets // (numel // num_channels) ... only if numel // num_channels = N*H*W
    # That gives n*C + c_within_n... no still wrong.
    #
    # For NCHW: offset = n*(C*HW) + c*HW + hw_idx
    # So c = (offset // HW) % C
    # We MUST pass HW. Let me restructure the kernel.
    c_idx = offsets  # placeholder, will fix below
    tl.store(out_ptr + offsets, x, mask=mask)


# Better kernel with hw_size parameter
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
def fused_bn_relu_kernel_v2(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # For NCHW contiguous: c = (offset // hw_size) % num_channels
    c_idx = (offsets // hw_size) % num_channels

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    weight = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias

    # ReLU
    out = tl.maximum(out, tl.zeros_like(out))

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def apply_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    """Apply fused BatchNorm + ReLU using Triton kernel."""
    N, C, H, W = x.shape
    numel = x.numel()
    out = torch.empty_like(x)
    hw_size = H * W

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel_v2[grid](
        x, running_mean, running_var, weight, bias,
        out,
        numel,
        hw_size,
        C,
        eps,
    )
    return out


def dense_layer_forward(x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight, eps=1e-5):
    """BN + ReLU + Conv2d + (no dropout in inference)"""
    # Fused BN + ReLU
    out = apply_bn_relu(x, bn_weight, bn_bias, bn_mean, bn_var, eps)
    # Conv2d (3x3, padding=1)
    out = F.conv2d(out, conv_weight, bias=None, stride=1, padding=1)
    return out


def transition_forward(x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight, eps=1e-5):
    """BN + ReLU + Conv2d (1x1) + AvgPool2d"""
    out = apply_bn_relu(x, bn_weight, bn_bias, bn_mean, bn_var, eps)
    out = F.conv2d(out, conv_weight, bias=None, stride=1, padding=0)
    out = F.avg_pool2d(out, kernel_size=2, stride=2)
    return out


def kernel_function(
    p_features_0_weight,
    p_features_1_weight,
    p_features_1_bias,
    p_dense_blocks_0_layers_0_0_weight,
    p_dense_blocks_0_layers_0_0_bias,
    p_dense_blocks_0_layers_0_2_weight,
    p_dense_blocks_0_layers_1_0_weight,
    p_dense_blocks_0_layers_1_0_bias,
    p_dense_blocks_0_layers_1_2_weight,
    p_dense_blocks_0_layers_2_0_weight,
    p_dense_blocks_0_layers_2_0_bias,
    p_dense_blocks_0_layers_2_2_weight,
    p_dense_blocks_0_layers_3_0_weight,
    p_dense_blocks_0_layers_3_0_bias,
    p_dense_blocks_0_layers_3_2_weight,
    p_dense_blocks_0_layers_4_0_weight,
    p_dense_blocks_0_layers_4_0_bias,
    p_dense_blocks_0_layers_4_2_weight,
    p_dense_blocks_0_layers_5_0_weight,
    p_dense_blocks_0_layers_5_0_bias,
    p_dense_blocks_0_layers_5_2_weight,
    p_transition_layers_0_transition_0_weight,
    p_transition_layers_0_transition_0_bias,
    p_transition_layers_0_transition_2_weight,
    p_dense_blocks_1_layers_0_0_weight,
    p_dense_blocks_1_layers_0_0_bias,
    p_dense_blocks_1_layers_0_2_weight,
    p_dense_blocks_1_layers_1_0_weight,
    p_dense_blocks_1_layers_1_0_bias,
    p_dense_blocks_1_layers_1_2_weight,
    p_dense_blocks_1_layers_2_0_weight,
    p_dense_blocks_1_layers_2_0_bias,
    p_dense_blocks_1_layers_2_2_weight,
    p_dense_blocks_1_layers_3_0_weight,
    p_dense_blocks_1_layers_3_0_bias,
    p_dense_blocks_1_layers_3_2_weight,
    p_dense_blocks_1_layers_4_0_weight,
    p_dense_blocks_1_layers_4_0_bias,
    p_dense_blocks_1_layers_4_2_weight,
    p_dense_blocks_1_layers_5_0_weight,
    p_dense_blocks_1_layers_5_0_bias,
    p_dense_blocks_1_layers_5_2_weight,
    p_dense_blocks_1_layers_6_0_weight,
    p_dense_blocks_1_layers_6_0_bias,
    p_dense_blocks_1_layers_6_2_weight,
    p_dense_blocks_1_layers_7_0_weight,
    p_dense_blocks_1_layers_7_0_bias,
    p_dense_blocks_1_layers_7_2_weight,
    p_dense_blocks_1_layers_8_0_weight,
    p_dense_blocks_1_layers_8_0_bias,
    p_dense_blocks_1_layers_8_2_weight,
    p_dense_blocks_1_layers_9_0_weight,
    p_dense_blocks_1_layers_9_0_bias,
    p_dense_blocks_1_layers_9_2_weight,
    p_dense_blocks_1_layers_10_0_weight,
    p_dense_blocks_1_layers_10_0_bias,
    p_dense_blocks_1_layers_10_2_weight,
    p_dense_blocks_1_layers_11_0_weight,
    p_dense_blocks_1_layers_11_0_bias,
    p_dense_blocks_1_layers_11_2_weight,
    p_transition_layers_1_transition_0_weight,
    p_transition_layers_1_transition_0_bias,
    p_transition_layers_1_transition_2_weight,
    p_dense_blocks_2_layers_0_0_weight,
    p_dense_blocks_2_layers_0_0_bias,
    p_dense_blocks_2_layers_0_2_weight,
    p_dense_blocks_2_layers_1_0_weight,
    p_dense_blocks_2_layers_1_0_bias,
    p_dense_blocks_2_layers_1_2_weight,
    p_dense_blocks_2_layers_2_0_weight,
    p_dense_blocks_2_layers_2_0_bias,
    p_dense_blocks_2_layers_2_2_weight,
    p_dense_blocks_2_layers_3_0_weight,
    p_dense_blocks_2_layers_3_0_bias,
    p_dense_blocks_2_layers_3_2_weight,
    p_dense_blocks_2_layers_4_0_weight,
    p_dense_blocks_2_layers_4_0_bias,
    p_dense_blocks_2_layers_4_2_weight,
    p_dense_blocks_2_layers_5_0_weight,
    p_dense_blocks_2_layers_5_0_bias,
    p_dense_blocks_2_layers_5_2_weight,
    p_dense_blocks_2_layers_6_0_weight,
    p_dense_blocks_2_layers_6_0_bias,
    p_dense_blocks_2_layers_6_2_weight,
    p_dense_blocks_2_layers_7_0_weight,
    p_dense_blocks_2_layers_7_0_bias,
    p_dense_blocks_2_layers_7_2_weight,
    p_dense_blocks_2_layers_8_0_weight,
    p_dense_blocks_2_layers_8_0_bias,
    p_dense_blocks_2_layers_8_2_weight,
    p_dense_blocks_2_layers_9_0_weight,
    p_dense_blocks_2_layers_9_0_bias,
    p_dense_blocks_2_layers_9_2_weight,
    p_dense_blocks_2_layers_10_0_weight,
    p_dense_blocks_2_layers_10_0_bias,
    p_dense_blocks_2_layers_10_2_weight,
    p_dense_blocks_2_layers_11_0_weight,
    p_dense_blocks_2_layers_11_0_bias,
    p_dense_blocks_2_layers_11_2_weight,
    p_dense_blocks_2_layers_12_0_weight,
    p_dense_blocks_2_layers_12_0_bias,
    p_dense_blocks_2_layers_12_2_weight,
    p_dense_blocks_2_layers_13_0_weight,
    p_dense_blocks_2_layers_13_0_bias,
    p_dense_blocks_2_layers_13_2_weight,
    p_dense_blocks_2_layers_14_0_weight,
    p_dense_blocks_2_layers_14_0_bias,
    p_dense_blocks_2_layers_14_2_weight,
    p_dense_blocks_2_layers_15_0_weight,
    p_dense_blocks_2_layers_15_0_bias,
    p_dense_blocks_2_layers_15_2_weight,
    p_dense_blocks_2_layers_16_0_weight,
    p_dense_blocks_2_layers_16_0_bias,
    p_dense_blocks_2_layers_16_2_weight,
    p_dense_blocks_2_layers_17_0_weight,
    p_dense_blocks_2_layers_17_0_bias,
    p_dense_blocks_2_layers_17_2_weight,
    p_dense_blocks_2_layers_18_0_weight,
    p_dense_blocks_2_layers_18_0_bias,
    p_dense_blocks_2_layers_18_2_weight,
    p_dense_blocks_2_layers_19_0_weight,
    p_dense_blocks_2_layers_19_0_bias,
    p_dense_blocks_2_layers_19_2_weight,
    p_dense_blocks_2_layers_20_0_weight,
    p_dense_blocks_2_layers_20_0_bias,
    p_dense_blocks_2_layers_20_2_weight,
    p_dense_blocks_2_layers_21_0_weight,
    p_dense_blocks_2_layers_21_0_bias,
    p_dense_blocks_2_layers_21_2_weight,
    p_dense_blocks_2_layers_22_0_weight,
    p_dense_blocks_2_layers_22_0_bias,
    p_dense_blocks_2_layers_22_2_weight,
    p_dense_blocks_2_layers_23_0_weight,
    p_dense_blocks_2_layers_23_0_bias,
    p_dense_blocks_2_layers_23_2_weight,
    p_dense_blocks_2_layers_24_0_weight,
    p_dense_blocks_2_layers_24_0_bias,
    p_dense_blocks_2_layers_24_2_weight,
    p_dense_blocks_2_layers_25_0_weight,
    p_dense_blocks_2_layers_25_0_bias,
    p_dense_blocks_2_layers_25_2_weight,
    p_dense_blocks_2_layers_26_0_weight,
    p_dense_blocks_2_layers_26_0_bias,
    p_dense_blocks_2_layers_26_2_weight,
    p_dense_blocks_2_layers_27_0_weight,
    p_dense_blocks_2_layers_27_0_bias,
    p_dense_blocks_2_layers_27_2_weight,
    p_dense_blocks_2_layers_28_0_weight,
    p_dense_blocks_2_layers_28_0_bias,
    p_dense_blocks_2_layers_28_2_weight,
    p_dense_blocks_2_layers_29_0_weight,
    p_dense_blocks_2_layers_29_0_bias,
    p_dense_blocks_2_layers_29_2_weight,
    p_dense_blocks_2_layers_30_0_weight,
    p_dense_blocks_2_layers_30_0_bias,
    p_dense_blocks_2_layers_30_2_weight,
    p_dense_blocks_2_layers_31_0_weight,
    p_dense_blocks_2_layers_31_0_bias,
    p_dense_blocks_2_layers_31_2_weight,
    p_dense_blocks_2_layers_32_0_weight,
    p_dense_blocks_2_layers_32_0_bias,
    p_dense_blocks_2_layers_32_2_weight,
    p_dense_blocks_2_layers_33_0_weight,
    p_dense_blocks_2_layers_33_0_bias,
    p_dense_blocks_2_layers_33_2_weight,
    p_dense_blocks_2_layers_34_0_weight,
    p_dense_blocks_2_layers_34_0_bias,
    p_dense_blocks_2_layers_34_2_weight,
    p_dense_blocks_2_layers_35_0_weight,
    p_dense_blocks_2_layers_35_0_bias,
    p_dense_blocks_2_layers_35_2_weight,
    p_dense_blocks_2_layers_36_0_weight,
    p_dense_blocks_2_layers_36_0_bias,
    p_dense_blocks_2_layers_36_2_weight,
    p_dense_blocks_2_layers_37_0_weight,
    p_dense_blocks_2_layers_37_0_bias,
    p_dense_blocks_2_layers_37_2_weight,
    p_dense_blocks_2_layers_38_0_weight,
    p_dense_blocks_2_layers_38_0_bias,
    p_dense_blocks_2_layers_38_2_weight,
    p_dense_blocks_2_layers_39_0_weight,
    p_dense_blocks_2_layers_39_0_bias,
    p_dense_blocks_2_layers_39_2_weight,
    p_dense_blocks_2_layers_40_0_weight,
    p_dense_blocks_2_layers_40_0_bias,
    p_dense_blocks_2_layers_40_2_weight,
    p_dense_blocks_2_layers_41_0_weight,
    p_dense_blocks_2_layers_41_0_bias,
    p_dense_blocks_2_layers_41_2_weight,
    p_dense_blocks_2_layers_42_0_weight,
    p_dense_blocks_2_layers_42_0_bias,
    p_dense_blocks_2_layers_42_2_weight,
    p_dense_blocks_2_layers_43_0_weight,
    p_dense_blocks_2_layers_43_0_bias,
    p_dense_blocks_2_layers_43_2_weight,
    p_dense_blocks_2_layers_44_0_weight,
    p_dense_blocks_2_layers_44_0_bias,
    p_dense_blocks_2_layers_44_2_weight,
    p_dense_blocks_2_layers_45_0_weight,
    p_dense_blocks_2_layers_45_0_bias,
    p_dense_blocks_2_layers_45_2_weight,
    p_dense_blocks_2_layers_46_0_weight,
    p_dense_blocks_2_layers_46_0_bias,
    p_dense_blocks_2_layers_46_2_weight,
    p_dense_blocks_2_layers_47_0_weight,
    p_dense_blocks_2_layers_47_0_bias,
    p_dense_blocks_2_layers_47_2_weight,
    p_transition_layers_2_transition_0_weight,
    p_transition_layers_2_transition_0_bias,
    p_transition_layers_2_transition_2_weight,
    p_dense_blocks_3_layers_0_0_weight,
    p_dense_blocks_3_layers_0_0_bias,
    p_dense_blocks_3_layers_0_2_weight,
    p_dense_blocks_3_layers_1_0_weight,
    p_dense_blocks_3_layers_1_0_bias,
    p_dense_blocks_3_layers_1_2_weight,
    p_dense_blocks_3_layers_2_0_weight,
    p_dense_blocks_3_layers_2_0_bias,
    p_dense_blocks_3_layers_2_2_weight,
    p_dense_blocks_3_layers_3_0_weight,
    p_dense_blocks_3_layers_3_0_bias,
    p_dense_blocks_3_layers_3_2_weight,
    p_dense_blocks_3_layers_4_0_weight,
    p_dense_blocks_3_layers_4_0_bias,
    p_dense_blocks_3_layers_4_2_weight,
    p_dense_blocks_3_layers_5_0_weight,
    p_dense_blocks_3_layers_5_0_bias,
    p_dense_blocks_3_layers_5_2_weight,
    p_dense_blocks_3_layers_6_0_weight,
    p_dense_blocks_3_layers_6_0_bias,
    p_dense_blocks_3_layers_6_2_weight,
    p_dense_blocks_3_layers_7_0_weight,
    p_dense_blocks_3_layers_7_0_bias,
    p_dense_blocks_3_layers_7_2_weight,
    p_dense_blocks_3_layers_8_0_weight,
    p_dense_blocks_3_layers_8_0_bias,
    p_dense_blocks_3_layers_8_2_weight,
    p_dense_blocks_3_layers_9_0_weight,
    p_dense_blocks_3_layers_9_0_bias,
    p_dense_blocks_3_layers_9_2_weight,
    p_dense_blocks_3_layers_10_0_weight,
    p_dense_blocks_3_layers_10_0_bias,
    p_dense_blocks_3_layers_10_2_weight,
    p_dense_blocks_3_layers_11_0_weight,
    p_dense_blocks_3_layers_11_0_bias,
    p_dense_blocks_3_layers_11_2_weight,
    p_dense_blocks_3_layers_12_0_weight,
    p_dense_blocks_3_layers_12_0_bias,
    p_dense_blocks_3_layers_12_2_weight,
    p_dense_blocks_3_layers_13_0_weight,
    p_dense_blocks_3_layers_13_0_bias,
    p_dense_blocks_3_layers_13_2_weight,
    p_dense_blocks_3_layers_14_0_weight,
    p_dense_blocks_3_layers_14_0_bias,
    p_dense_blocks_3_layers_14_2_weight,
    p_dense_blocks_3_layers_15_0_weight,
    p_dense_blocks_3_layers_15_0_bias,
    p_dense_blocks_3_layers_15_2_weight,
    p_dense_blocks_3_layers_16_0_weight,
    p_dense_blocks_3_layers_16_0_bias,
    p_dense_blocks_3_layers_16_2_weight,
    p_dense_blocks_3_layers_17_0_weight,
    p_dense_blocks_3_layers_17_0_bias,
    p_dense_blocks_3_layers_17_2_weight,
    p_dense_blocks_3_layers_18_0_weight,
    p_dense_blocks_3_layers_18_0_bias,
    p_dense_blocks_3_layers_18_2_weight,
    p_dense_blocks_3_layers_19_0_weight,
    p_dense_blocks_3_layers_19_0_bias,
    p_dense_blocks_3_layers_19_2_weight,
    p_dense_blocks_3_layers_20_0_weight,
    p_dense_blocks_3_layers_20_0_bias,
    p_dense_blocks_3_layers_20_2_weight,
    p_dense_blocks_3_layers_21_0_weight,
    p_dense_blocks_3_layers_21_0_bias,
    p_dense_blocks_3_layers_21_2_weight,
    p_dense_blocks_3_layers_22_0_weight,
    p_dense_blocks_3_layers_22_0_bias,
    p_dense_blocks_3_layers_22_2_weight,
    p_dense_blocks_3_layers_23_0_weight,
    p_dense_blocks_3_layers_23_0_bias,
    p_dense_blocks_3_layers_23_2_weight,
    p_dense_blocks_3_layers_24_0_weight,
    p_dense_blocks_3_layers_24_0_bias,
    p_dense_blocks_3_layers_24_2_weight,
    p_dense_blocks_3_layers_25_0_weight,
    p_dense_blocks_3_layers_25_0_bias,
    p_dense_blocks_3_layers_25_2_weight,
    p_dense_blocks_3_layers_26_0_weight,
    p_dense_blocks_3_layers_26_0_bias,
    p_dense_blocks_3_layers_26_2_weight,
    p_dense_blocks_3_layers_27_0_weight,
    p_dense_blocks_3_layers_27_0_bias,
    p_dense_blocks_3_layers_27_2_weight,
    p_dense_blocks_3_layers_28_0_weight,
    p_dense_blocks_3_layers_28_0_bias,
    p_dense_blocks_3_layers_28_2_weight,
    p_dense_blocks_3_layers_29_0_weight,
    p_dense_blocks_3_layers_29_0_bias,
    p_dense_blocks_3_layers_29_2_weight,
    p_dense_blocks_3_layers_30_0_weight,
    p_dense_blocks_3_layers_30_0_bias,
    p_dense_blocks_3_layers_30_2_weight,
    p_dense_blocks_3_layers_31_0_weight,
    p_dense_blocks_3_layers_31_0_bias,
    p_dense_blocks_3_layers_31_2_weight,
    p_final_bn_weight,
    p_final_bn_bias,
    p_classifier_weight,
    p_classifier_bias,
    b_features_1_running_mean,
    b_features_1_running_var,
    b_dense_blocks_0_layers_0_0_running_mean,
    b_dense_blocks_0_layers_0_0_running_var,
    b_dense_blocks_0_layers_1_0_running_mean,
    b_dense_blocks_0_layers_1_0_running_var,
    b_dense_blocks_0_layers_2_0_running_mean,
    b_dense_blocks_0_layers_2_0_running_var,
    b_dense_blocks_0_layers_3_0_running_mean,
    b_dense_blocks_0_layers_3_0_running_var,
    b_dense_blocks_0_layers_4_0_running_mean,
    b_dense_blocks_0_layers_4_0_running_var,
    b_dense_blocks_0_layers_5_0_running_mean,
    b_dense_blocks_0_layers_5_0_running_var,
    b_transition_layers_0_transition_0_running_mean,
    b_transition_layers_0_transition_0_running_var,
    b_dense_blocks_1_layers_0_0_running_mean,
    b_dense_blocks_1_layers_0_0_running_var,
    b_dense_blocks_1_layers_1_0_running_mean,
    b_dense_blocks_1_layers_1_0_running_var,