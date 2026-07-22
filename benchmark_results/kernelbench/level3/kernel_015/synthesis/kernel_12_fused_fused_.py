import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index
    # Layout: (N, C, H, W) -> channel = (offset / spatial) % num_channels
    chan = (offsets // spatial) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)

    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b
    # ReLU
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial,
        eps,
    )
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
    b_dense_blocks_1_layers_2_0_running_mean,
    b_dense_blocks_1_layers_2_0_running_var,
    b_dense_blocks_1_layers_3_0_running_mean,
    b_dense_blocks_1_layers_3_0_running_var,
    b_dense_blocks_1_layers_4_0_running_mean,
    b_dense_blocks_1_layers_4_0_running_var,
    b_dense_blocks_1_layers_5_0_running_mean,
    b_dense_blocks_1_layers_5_0_running_var,
    b_dense_blocks_1_layers_6_0_running_mean,
    b_dense_blocks_1_layers_6_0_running_var,
    b_dense_blocks_1_layers_7_0_running_mean,
    b_dense_blocks_1_layers_7_0_running_var,
    b_dense_blocks_1_layers_8_0_running_mean,
    b_dense_blocks_1_layers_8_0_running_var,
    b_dense_blocks_1_layers_9_0_running_mean,
    b_dense_blocks_1_layers_9_0_running_var,
    b_dense_blocks_1_layers_10_0_running_mean,
    b_dense_blocks_1_layers_10_0_running_var,
    b_dense_blocks_1_layers_11_0_running_mean,
    b_dense_blocks_1_layers_11_0_running_var,
    b_transition_layers_1_transition_0_running_mean,
    b_transition_layers_1_transition_0_running_var,
    b_dense_blocks_2_layers_0_0_running_mean,
    b_dense_blocks_2_layers_0_0_running_var,
    b_dense_blocks_2_layers_1_0_running_mean,
    b_dense_blocks_2_layers_1_0_running_var,
    b_dense_blocks_2_layers_2_0_running_mean,
    b_dense_blocks_2_layers_2_0_running_var,
    b_dense_blocks_2_layers_3_0_running_mean,
    b_dense_blocks_2_layers_3_0_running_var,
    b_dense_blocks_2_layers_4_0_running_mean,
    b_dense_blocks_2_layers_4_0_running_var,
    b_dense_blocks_2_layers_5_0_running_mean,
    b_dense_blocks_2_layers_5_0_running_var,
    b_dense_blocks_2_layers_6_0_running_mean,
    b_dense_blocks_2_layers_6_0_running_var,
    b_dense_blocks_2_layers_7_0_running_mean,
    b_dense_blocks_2_layers_7_0_running_var,
    b_dense_blocks_2_layers_8_0_running_mean,
    b_dense_blocks_2_layers_8_0_running_var,
    b_dense_blocks_2_layers_9_0_running_mean,
    b_dense_blocks_2_layers_9_0_running_var,
    b_dense_blocks_2_layers_10_0_running_mean,
    b_dense_blocks_2_layers_10_0_running_var,
    b_dense_blocks_2_layers_11_0_running_mean,
    b_dense_blocks_2_layers_11_0_running_var,
    b_dense_blocks_2_layers_12_0_running_mean,
    b_dense_blocks_2_layers_12_0_running_var,
    b_dense_blocks_2_layers_13_0_running_mean,
    b_dense_blocks_2_layers_13_0_running_var,
    b_dense_blocks_2_layers_14_0_running_mean,
    b_dense_blocks_2_layers_14_0_running_var,
    b_dense_blocks_2_layers_15_0_running_mean,
    b_dense_blocks_2_layers_15_0_running_var,
    b_dense_blocks_2_layers_16_0_running_mean,
    b_dense_blocks_2_layers_16_0_running_var,
    b_dense_blocks_2_layers_17_0_running_mean,
    b_dense_blocks_2_layers_17_0_running_var,
    b_dense_blocks_2_layers_18_0_running_mean,
    b_dense_blocks_2_layers_18_0_running_var,
    b_dense_blocks_2_layers_19_0_running_mean,
    b_dense_blocks_2_layers_19_0_running_var,
    b_dense_blocks_2_layers_20_0_running_mean,
    b_dense_blocks_2_layers_20_0_running_var,
    b_dense_blocks_2_layers_21_0_running_mean,
    b_dense_blocks_2_layers_21_0_running_var,
    b_dense_blocks_2_layers_22_0_running_mean,
    b_dense_blocks_2_layers_22_0_running_var,
    b_dense_blocks_2_layers_23_0_running_mean,
    b_dense_blocks_2_layers_23_0_running_var,
    b_transition_layers_2_transition_0_running_mean,
    b_transition_layers_2_transition_0_running_var,
    b_dense_blocks_3_layers_0_0_running_mean,
    b_dense_blocks_3_layers_0_0_running_var,
    b_dense_blocks_3_layers_1_0_running_mean,
    b_dense_blocks_3_layers_1_0_running_var,
    b_dense_blocks_3_layers_2_0_running_mean,
    b_dense_blocks_3_layers_2_0_running_var,
    b_dense_blocks_3_layers_3_0_running_mean,
    b_dense_blocks_3_layers_3_0_running_var,
    b_dense_blocks_3_layers_4_0_running_mean,
    b_dense_blocks_3_layers_4_0_running_var,
    b_dense_blocks_3_layers_5_0_running_mean,
    b_dense_blocks_3_layers_5_0_running_var,
    b_dense_blocks_3_layers_6_0_running_mean,
    b_dense_blocks_3_layers_6_0_running_var,
    b_dense_blocks_3_layers_7_0_running_mean,
    b_dense_blocks_3_layers_7_0_running_var,
    b_dense_blocks_3_layers_8_0_running_mean,
    b_dense_blocks_3_layers_8_0_running_var,
    b_dense_blocks_3_layers_9_0_running_mean,
    b_dense_blocks_3_layers_9_0_running_var,
    b_dense_blocks_3_layers_10_0_running_mean,
    b_dense_blocks_3_layers_10_0_running_var,
    b_dense_blocks_3_layers_11_0_running_mean,
    b_dense_blocks_3_layers_11_0_running_var,
    b_dense_blocks_3_layers_12_0_running_mean,
    b_dense_blocks_3_layers_12_0_running_var,
    b_dense_blocks_3_layers_13_0_running_mean,
    b_dense_blocks_3_layers_13_0_running_var,
    b_dense_blocks_3_layers_14_0_running_mean,
    b_dense_blocks_3_layers_14_0_running_var,
    b_dense_blocks_3_layers_15_0_running_mean,
    b_dense_blocks_3_layers_15_0_running_var,
    b_final_bn_running_mean,
    b_final_bn_running_var,
    x,
):
    # Initial conv + BN + ReLU + MaxPool
    out = F.conv2d(x, p_features_0_weight, None, stride=2, padding=3)
    out = fused_bn_relu(out, p_features_1_weight, p_features_1_bias,
    out = F.max_pool2d(out, kernel_size=3, stride=2, padding=1)

    # Dense Block 0 (6 layers, growth_rate=32, starting channels=64)
    # Layer 0: in=64, out=32
    feat = fused_bn_relu(out, p_dense_blocks_0_layers_0_0_weight, p_dense_blocks_0_layers_0_0_bias,
    new_feat = F.conv2d(feat, p_dense_blocks_0_layers_0_2_weight, None, padding=1)
    out = torch.cat([out, new_feat], dim=1)

    # Layer 1: in=96, out=32
    feat = fused_bn_relu(out, p_dense_blocks_0_layers_1_0_weight, p_dense_blocks_0_layers_1_0_bias,
    new_feat = F.conv2d(feat, p_dense_blocks_0_layers_1_2_weight, None, padding=1)
    out = torch.cat([out, new_feat], dim=1)

    # Layer 2: in=128, out=32
    feat = fused_bn_relu(out, p_dense_blocks_0_layers_2_0_weight, p_dense_blocks_0_layers_2_0_bias,
    new_feat = F.conv2d(feat, p_dense_blocks_0_layers_2_2_weight, None, padding=1)
    out = torch.cat([out, new_feat], dim=1)

    # Layer 3: in=160, out=32
    feat = fused_bn_relu(out, p_dense_blocks_0_layers_3_0_weight, p_dense_blocks_0_layers_3_0_bias,
    new_feat = F.conv2d(feat, p_dense_blocks_0_layers_3_2_weight, None, padding=1)
    out = torch.cat([out, new_feat], dim=1)

    # Layer 4: in=192, out=32
    feat = fused_bn_relu(out, p_dense_blocks_0_layers_4_0_weight, p_dense_blocks_0_layers_4_0_bias,
    new_feat = F.conv2d(feat, p_dense_blocks_0_layers_4_2_weight, None, padding=1)
    out = torch.cat([out, new_feat], dim=1)

    # Layer 5: in=224, out=32
    feat = fused_bn_relu(out, p_dense_blocks_0_layers_5_0_weight, p_dense_blocks_0_layers_5_0_bias,
    new_feat = F.conv2d(feat, p_dense_blocks_0_layers_5_2_weight, None, padding=1)
    out = torch.cat([out, new_feat], dim=1)

    # Transition 0: in=256, out=128
    out = fused_bn_relu(out, p_transition_layers_0_transition_0_weight, p_transition_layers_0_transition_0_bias,
    out = F.conv2d(out, p_transition_layers_0_transition_2_weight, None)
    out = F.avg_pool2d(out, kernel_size=2, stride=2)

    # Dense Block 1 (12 layers, growth_rate=32, starting channels=128)
    # Layer 0: in=128, out=32
    feat = fused_bn_relu(out, p_dense_blocks_1_layers_0_0_weight, p_dense_blocks_1_layers_0_0_bias,
    new_feat = F.conv2d(feat, p_dense_blocks_1_layers_0_2_weight, None, padding=1)
    out = torch.cat([out, new_feat], dim=1)

    # Layer 1: in=160, out=32
    feat = fused_bn_relu(out, p_dense_blocks_1_layers_1_0_weight, p_dense_blocks_1_layers_1_0_bias,
    new_feat = F.conv2d(feat, p_dense_blocks_1_layers_1_2_weight, None, padding=1)
    out = torch.cat([out, new_feat], dim=1)

    # Layer 2: in=192, out=32
    feat = fused_bn_relu(out, p_dense_blocks_1_