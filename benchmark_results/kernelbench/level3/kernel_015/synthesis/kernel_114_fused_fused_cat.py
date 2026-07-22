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
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        numel, C, HW,
        eps,
    )
    return out


def kernel_function(
    p_features_0_weight, p_features_1_weight, p_features_1_bias,
    p_dense_blocks_0_layers_0_0_weight, p_dense_blocks_0_layers_0_0_bias,
    p_dense_blocks_0_layers_0_2_weight,
    p_dense_blocks_0_layers_1_0_weight, p_dense_blocks_0_layers_1_0_bias,
    p_dense_blocks_0_layers_1_2_weight,
    p_dense_blocks_0_layers_2_0_weight, p_dense_blocks_0_layers_2_0_bias,
    p_dense_blocks_0_layers_2_2_weight,
    p_dense_blocks_0_layers_3_0_weight, p_dense_blocks_0_layers_3_0_bias,
    p_dense_blocks_0_layers_3_2_weight,
    p_dense_blocks_0_layers_4_0_weight, p_dense_blocks_0_layers_4_0_bias,
    p_dense_blocks_0_layers_4_2_weight,
    p_dense_blocks_0_layers_5_0_weight, p_dense_blocks_0_layers_5_0_bias,
    p_dense_blocks_0_layers_5_2_weight,
    p_transition_layers_0_transition_0_weight, p_transition_layers_0_transition_0_bias,
    p_transition_layers_0_transition_2_weight,
    p_dense_blocks_1_layers_0_0_weight, p_dense_blocks_1_layers_0_0_bias,
    p_dense_blocks_1_layers_0_2_weight,
    p_dense_blocks_1_layers_1_0_weight, p_dense_blocks_1_layers_1_0_bias,
    p_dense_blocks_1_layers_1_2_weight,
    p_dense_blocks_1_layers_2_0_weight, p_dense_blocks_1_layers_2_0_bias,
    p_dense_blocks_1_layers_2_2_weight,
    p_dense_blocks_1_layers_3_0_weight, p_dense_blocks_1_layers_3_0_bias,
    p_dense_blocks_1_layers_3_2_weight,
    p_dense_blocks_1_layers_4_0_weight, p_dense_blocks_1_layers_4_0_bias,
    p_dense_blocks_1_layers_4_2_weight,
    p_dense_blocks_1_layers_5_0_weight, p_dense_blocks_1_layers_5_0_bias,
    p_dense_blocks_1_layers_5_2_weight,
    p_dense_blocks_1_layers_6_0_weight, p_dense_blocks_1_layers_6_0_bias,
    p_dense_blocks_1_layers_6_2_weight,
    p_dense_blocks_1_layers_7_0_weight, p_dense_blocks_1_layers_7_0_bias,
    p_dense_blocks_1_layers_7_2_weight,
    p_dense_blocks_1_layers_8_0_weight, p_dense_blocks_1_layers_8_0_bias,
    p_dense_blocks_1_layers_8_2_weight,
    p_dense_blocks_1_layers_9_0_weight, p_dense_blocks_1_layers_9_0_bias,
    p_dense_blocks_1_layers_9_2_weight,
    p_dense_blocks_1_layers_10_0_weight, p_dense_blocks_1_layers_10_0_bias,
    p_dense_blocks_1_layers_10_2_weight,
    p_dense_blocks_1_layers_11_0_weight, p_dense_blocks_1_layers_11_0_bias,
    p_dense_blocks_1_layers_11_2_weight,
    p_transition_layers_1_transition_0_weight, p_transition_layers_1_transition_0_bias,
    p_transition_layers_1_transition_2_weight,
    p_dense_blocks_2_layers_0_0_weight, p_dense_blocks_2_layers_0_0_bias,
    p_dense_blocks_2_layers_0_2_weight,
    p_dense_blocks_2_layers_1_0_weight, p_dense_blocks_2_layers_1_0_bias,
    p_dense_blocks_2_layers_1_2_weight,
    p_dense_blocks_2_layers_2_0_weight, p_dense_blocks_2_layers_2_0_bias,
    p_dense_blocks_2_layers_2_2_weight,
    p_dense_blocks_2_layers_3_0_weight, p_dense_blocks_2_layers_3_0_bias,
    p_dense_blocks_2_layers_3_2_weight,
    p_dense_blocks_2_layers_4_0_weight, p_dense_blocks_2_layers_4_0_bias,
    p_dense_blocks_2_layers_4_2_weight,
    p_dense_blocks_2_layers_5_0_weight, p_dense_blocks_2_layers_5_0_bias,
    p_dense_blocks_2_layers_5_2_weight,
    p_dense_blocks_2_layers_6_0_weight, p_dense_blocks_2_layers_6_0_bias,
    p_dense_blocks_2_layers_6_2_weight,
    p_dense_blocks_2_layers_7_0_weight, p_dense_blocks_2_layers_7_0_bias,
    p_dense_blocks_2_layers_7_2_weight,
    p_dense_blocks_2_layers_8_0_weight, p_dense_blocks_2_layers_8_0_bias,
    p_dense_blocks_2_layers_8_2_weight,
    p_dense_blocks_2_layers_9_0_weight, p_dense_blocks_2_layers_9_0_bias,
    p_dense_blocks_2_layers_9_2_weight,
    p_dense_blocks_2_layers_10_0_weight, p_dense_blocks_2_layers_10_0_bias,
    p_dense_blocks_2_layers_10_2_weight,
    p_dense_blocks_2_layers_11_0_weight, p_dense_blocks_2_layers_11_0_bias,
    p_dense_blocks_2_layers_11_2_weight,
    p_dense_blocks_2_layers_12_0_weight, p_dense_blocks_2_layers_12_0_bias,
    p_dense_blocks_2_layers_12_2_weight,
    p_dense_blocks_2_layers_13_0_weight, p_dense_blocks_2_layers_13_0_bias,
    p_dense_blocks_2_layers_13_2_weight,
    p_dense_blocks_2_layers_14_0_weight, p_dense_blocks_2_layers_14_0_bias,
    p_dense_blocks_2_layers_14_2_weight,
    p_dense_blocks_2_layers_15_0_weight, p_dense_blocks_2_layers_15_0_bias,
    p_dense_blocks_2_layers_15_2_weight,
    p_dense_blocks_2_layers_16_0_weight, p_dense_blocks_2_layers_16_0_bias,
    p_dense_blocks_2_layers_16_2_weight,
    p_dense_blocks_2_layers_17_0_weight, p_dense_blocks_2_layers_17_0_bias,
    p_dense_blocks_2_layers_17_2_weight,
    p_dense_blocks_2_layers_18_0_weight, p_dense_blocks_2_layers_18_0_bias,
    p_dense_blocks_2_layers_18_2_weight,
    p_dense_blocks_2_layers_19_0_weight, p_dense_blocks_2_layers_19_0_bias,
    p_dense_blocks_2_layers_19_2_weight,
    p_dense_blocks_2_layers_20_0_weight, p_dense_blocks_2_layers_20_0_bias,
    p_dense_blocks_2_layers_20_2_weight,
    p_dense_blocks_2_layers_21_0_weight, p_dense_blocks_2_layers_21_0_bias,
    p_dense_blocks_2_layers_21_2_weight,
    p_dense_blocks_2_layers_22_0_weight, p_dense_blocks_2_layers_22_0_bias,
    p_dense_blocks_2_layers_22_2_weight,
    p_dense_blocks_2_layers_23_0_weight, p_dense_blocks_2_layers_23_0_bias,
    p_dense_blocks_2_layers_23_2_weight,
    p_transition_layers_2_transition_0_weight, p_transition_layers_2_transition_0_bias,
    p_transition_layers_2_transition_2_weight,
    p_dense_blocks_3_layers_0_0_weight, p_dense_blocks_3_layers_0_0_bias,
    p_dense_blocks_3_layers_0_2_weight,
    p_dense_blocks_3_layers_1_0_weight, p_dense_blocks_3_layers_1_0_bias,
    p_dense_blocks_3_layers_1_2_weight,
    p_dense_blocks_3_layers_2_0_weight, p_dense_blocks_3_layers_2_0_bias,
    p_dense_blocks_3_layers_2_2_weight,
    p_dense_blocks_3_layers_3_0_weight, p_dense_blocks_3_layers_3_0_bias,
    p_dense_blocks_3_layers_3_2_weight,
    p_dense_blocks_3_layers_4_0_weight, p_dense_blocks_3_layers_4_0_bias,
    p_dense_blocks_3_layers_4_2_weight,
    p_dense_blocks_3_layers_5_0_weight, p_dense_blocks_3_layers_5_0_bias,
    p_dense_blocks_3_layers_5_2_weight,
    p_dense_blocks_3_layers_6_0_weight, p_dense_blocks_3_layers_6_0_bias,
    p_dense_blocks_3_layers_6_2_weight,
    p_dense_blocks_3_layers_7_0_weight, p_dense_blocks_3_layers_7_0_bias,
    p_dense_blocks_3_layers_7_2_weight,
    p_dense_blocks_3_layers_8_0_weight, p_dense_blocks_3_layers_8_0_bias,
    p_dense_blocks_3_layers_8_2_weight,
    p_dense_blocks_3_layers_9_0_weight, p_dense_blocks_3_layers_9_0_bias,
    p_dense_blocks_3_layers_9_2_weight,
    p_dense_blocks_3_layers_10_0_weight, p_dense_blocks_3_layers_10_0_bias,
    p_dense_blocks_3_layers_10_2_weight,
    p_dense_blocks_3_layers_11_0_weight, p_dense_blocks_3_layers_11_0_bias,
    p_dense_blocks_3_layers_11_2_weight,
    p_dense_blocks_3_layers_12_0_weight, p_dense_blocks_3_layers_12_0_bias,
    p_dense_blocks_3_layers_12_2_weight,
    p_dense_blocks_3_layers_13_0_weight, p_dense_blocks_3_layers_13_0_bias,
    p_dense_blocks_3_layers_13_2_weight,
    p_dense_blocks_3_layers_14_0_weight, p_dense_blocks_3_layers_14_0_bias,
    p_dense_blocks_3_layers_14_2_weight,
    p_dense_blocks_3_layers_15_0_weight, p_dense_blocks_3_layers_15_0_bias,
    p_dense_blocks_3_layers_15_2_weight,
    p_final_bn_weight, p_final_bn_bias,
    p_classifier_weight, p_classifier_bias,
    b_features_1_running_mean, b_features_1_running_var,
    b_dense_blocks_0_layers_0_0_running_mean, b_dense_blocks_0_layers_0_0_running_var,
    b_dense_blocks_0_layers_1_0_running_mean, b_dense_blocks_0_layers_1_0_running_var,
    b_dense_blocks_0_layers_2_0_running_mean, b_dense_blocks_0_layers_2_0_running_var,
    b_dense_blocks_0_layers_3_0_running_mean, b_dense_blocks_0_layers_3_0_running_var,
    b_dense_blocks_0_layers_4_0_running_mean, b_dense_blocks_0_layers_4_0_running_var,
    b_dense_blocks_0_layers_5_0_running_mean, b_dense_blocks_0_layers_5_0_running_var,
    b_transition_layers_0_transition_0_running_mean, b_transition_layers_0_transition_0_running_var,
    b_dense_blocks_1_layers_0_0_running_mean, b_dense_blocks_1_layers_0_0_running_var,
    b_dense_blocks_1_layers_1_0_running_mean, b_dense_blocks_1_layers_1_0_running_var,
    b_dense_blocks_1_layers_2_0_running_mean, b_dense_blocks_1_layers_2_0_running_var,
    b_dense_blocks_1_layers_3_0_running_mean, b_dense_blocks_1_layers_3_0_running_var,
    b_dense_blocks_1_layers_4_0_running_mean, b_dense_blocks_1_layers_4_0_running_var,
    b_dense_blocks_1_layers_5_0_running_mean, b_dense_blocks_1_layers_5_0_running_var,
    b_dense_blocks_1_layers_6_0_running_mean, b_dense_blocks_1_layers_6_0_running_var,
    b_dense_blocks_1_layers_7_0_running_mean, b_dense_blocks_1_layers_7_0_running_var,
    b_dense_blocks_1_layers_8_0_running_mean, b_dense_blocks_1_layers_8_0_running_var,
    b_dense_blocks_1_layers_9_0_running_mean, b_dense_blocks_1_layers_9_0_running_var,
    b_dense_blocks_1_layers_10_0_running_mean, b_dense_blocks_1_layers_10_0_running_var,
    b_dense_blocks_1_layers_11_0_running_mean, b_dense_blocks_1_layers_11_0_running_var,
    b_transition_layers_1_transition_0_running_mean, b_transition_layers_1_transition_0_running_var,
    b_dense_blocks_2_layers_0_0_running_mean, b_dense_blocks_2_layers_0_0_running_var,
    b_dense_blocks_2_layers_1_0_running_mean, b_dense_blocks_2_layers_1_0_running_var,
    b_dense_blocks_2_layers_2_0_running_mean, b_dense_blocks_2_layers_2_0_running_var,
    b_dense_blocks_2_layers_3_0_running_mean, b_dense_blocks_2_layers_3_0_running_var,
    b_dense_blocks_2_layers_4_0_running_mean, b_dense_blocks_2_layers_4_0_running_var,
    b_dense_blocks_2_layers_5_0_running_mean, b_dense_blocks_2_layers_5_0_running_var,
    b_dense_blocks_2_layers_6_0_running_mean, b_dense_blocks_2_layers_6_0_running_var,
    b_dense_blocks_2_layers_7_0_running_mean, b_dense_blocks_2_layers_7_0_running_var,
    b_dense_blocks_2_layers_8_0_running_mean, b_dense_blocks_2_layers_8_0_running_var,
    b_dense_blocks_2_layers_9_0_running_mean, b_dense_blocks_2_layers_9_0_running_var,
    b_dense_blocks_2_layers_10_0_running_mean, b_dense_blocks_2_layers_10_0_running_var,
    b_dense_blocks_2_layers_11_0_running_mean, b_dense_blocks_2_layers_11_0_running_var,
    b_dense_blocks_2_layers_12_0_running_mean, b_dense_blocks_2_layers_12_0_running_var,
    b_dense_blocks_2_layers_13_0_running_mean, b_dense_blocks_2_layers_13_0_running_var,
    b_dense_blocks_2_layers_14_0_running_mean, b_dense_blocks_2_layers_14_0_running_var,
    b_dense_blocks_2_layers_15_0_running_mean, b_dense_blocks_2_layers_15_0_running_var,
    b_dense_blocks_2_layers_16_0_running_mean, b_dense_blocks_2_layers_16_0_running_var,
    b_dense_blocks_2_layers_17_0_running_mean, b_dense_blocks_2_layers_17_0_running_var,
    b_dense_blocks_2_layers_18_0_running_mean, b_dense_blocks_2_layers_18_0_running_var,
    b_dense_blocks_2_layers_19_0_running_mean, b_dense_blocks_2_layers_19_0_running_var,
    b_dense_blocks_2_layers_20_0_running_mean, b_dense_blocks_2_layers_20_0_running_var,
    b_dense_blocks_2_layers_21_0_running_mean, b_dense_blocks_2_layers_21_0_running_var,
    b_dense_blocks_2_layers_22_0_running_mean, b_dense_blocks_2_layers_22_0_running_var,
    b_dense_blocks_2_layers_23_0_running_mean, b_dense_blocks_2_layers_23_0_running_var,
    b_transition_layers_2_transition_0_running_mean, b_transition_layers_2_transition_0_running_var,
    b_dense_blocks_3_layers_0_0_running_mean, b_dense_blocks_3_layers_0_0_running_var,
    b_dense_blocks_3_layers_1_0_running_mean, b_dense_blocks_3_layers_1_0_running_var,
    b_dense_blocks_3_layers_2_0_running_mean, b_dense_blocks_3_layers_2_0_running_var,
    b_dense_blocks_3_layers_3_0_running_mean, b_dense_blocks_3_layers_3_0_running_var,
    b_dense_blocks_3_layers_4_0_running_mean, b_dense_blocks_3_layers_4_0_running_var,
    b_dense_blocks_3_layers_5_0_running_mean, b_dense_blocks_3_layers_5_0_running_var,
    b_dense_blocks_3_layers_6_0_running_mean, b_dense_blocks_3_layers_6_0_running_var,
    b_dense_blocks_3_layers_7_0_running_mean, b_dense_blocks_3_layers_7_0_running_var,
    b_dense_blocks_3_layers_8_0_running_mean, b_dense_blocks_3_layers_8_0_running_var,
    b_dense_blocks_3_layers_9_0_running_mean, b_dense_blocks_3_layers_9_0_running_var,
    b_dense_blocks_3_layers_10_0_running_mean, b_dense_blocks_3_layers_10_0_running_var,
    b_dense_blocks_3_layers_11_0_running_mean, b_dense_blocks_3_layers_11_0_running_var,
    b_dense_blocks_3_layers_12_0_running_mean, b_dense_blocks_3_layers_12_0_running_var,
    b_dense_blocks_3_layers_13_0_running_mean, b_dense_blocks_3_layers_13_0_running_var,
    b_dense_blocks_3_layers_14_0_running_mean, b_dense_blocks_3_layers_14_0_running_var,
    b_dense_blocks_3_layers_15_0_running_mean, b_dense_blocks_3_layers_15_0_running_var,
    b_final_bn_running_mean, b_final_bn_running_var,
    x,
):
    # Initial features: conv7x7 + BN + ReLU + MaxPool
    # conv2d: stride=2, padding=3, kernel=7x7
    x = F.conv2d(x, p_features_0_weight, bias=None, stride=2, padding=3)
    x = fused_bn_relu(x, p_features_1_weight, p_features_1_bias,
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

    # Dense Block 0: 6 layers, growth_rate=32, initial channels=64
    # Layer 0: BN+ReLU+Conv(64->32)
    def dense_layer(x_in, bn_w, bn_b, rm, rv, conv_w):
        h = fused_bn_relu(x_in, bn_w, bn_b, rm, rv)
        h = F.conv2d(h, conv_w, bias=None, padding=1)
        return h

    # Dense Block 0
    x0 = x
    h = dense_layer(x0, p_dense_blocks_0_layers_0_0_weight, p_dense_blocks_0_layers_0_0_bias,
                    b_dense_blocks_0_layers_0_0_running_mean, b_dense_blocks_0_layers_0_0_running_var,
    x1 = torch.cat([x0, h], dim=1)
    
    h = dense_layer(x1, p_dense_blocks_0_layers_1_0_weight, p_dense_blocks_0_layers_1_0_bias,
                    b_dense_blocks_0_layers_1_0_running_mean, b_dense_blocks_0_layers_1_0_running_var,
    x2 = torch.cat([x1, h], dim=1)
    
    h = dense_layer(x2, p_dense_blocks_0_layers_2_0_weight, p_dense_blocks_0_layers_2_0_bias,
                    b_dense_blocks_0_layers_2_0_running_mean, b_dense_blocks_0_layers_2_0_running_var,
    x3 = torch.cat([x2, h], dim=1)
    
    h = dense_layer(x3, p_dense_blocks_0_layers_3_0_weight, p_dense_blocks_0_layers_3_0_bias,
                    b_dense_blocks_0_layers_3_0_running_mean, b_dense_blocks_0_layers_3_0_running_var,
    x4 = torch.cat([x3, h], dim=1)
    
    h = dense_layer(x4, p_dense_blocks_0_layers_4_0_weight, p_dense_blocks_0_layers_4_0_bias,
                    b_dense_blocks_0_layers_4_0_running_mean, b_dense_blocks_0_layers_4_0_running_var,
    x5 = torch.cat([x4, h], dim=1)
    
    h = dense_layer(x5, p_dense_blocks_0_layers_5_0_weight, p_dense_blocks_0_layers_5_0_bias,
                    b_dense_blocks_0_layers_5_0_running_mean, b_dense_blocks_0_layers_5_0_running_var,
    x = torch.cat([x5, h], dim=1)  # [10, 256, 56, 56]

    # Transition 0: BN+ReLU+Conv(256->128)+AvgPool
    x = fused_bn_relu(x, p_transition_layers_0_transition_0_weight, p_transition_layers_0_transition_0_bias,
    x = F.conv2d(x, p_transition_layers_0_transition_2_weight, bias=None)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)  # [10, 128, 28, 28]

    # Dense Block 1: 12 layers
    x0 = x
    h = dense_layer(x0, p_dense_blocks_1_layers_0_0_weight, p_dense_blocks_1_layers_0_0_bias,
                    b_dense_blocks_1_layers_0_0_running_mean, b_dense_blocks_1_layers_0_0_running_var,
    x1 = torch.cat([x0, h], dim=1)
    
    h = dense_layer(x1, p_dense_blocks_1_layers_1_0_weight, p_dense_blocks_1_layers_1_0_bias,
                    b_dense_blocks_1_layers_1_0_running_mean, b_dense_blocks_1_layers_1_0_running_var,
    x2 = torch.cat([x1, h], dim=1)
    
    h = dense_layer(x2, p_dense_blocks_1_layers_2_0_weight, p_dense_blocks_1_layers_2_0_bias,
                    b_dense_blocks_1_layers_2_0_running_mean, b_dense_blocks_1_layers_2_0_running_var,
    x3 = torch.cat([x2, h], dim=1)
    
    h = dense_layer(x3, p_dense_blocks_1_layers_3_0_weight, p_dense_blocks_1_layers_3_0_bias,
                    b_dense_blocks_1_layers_3_0_running_mean, b_dense_blocks_1_layers_3_0_running_var,
    x4 = torch.cat([x3, h], dim=1)
    
    h = dense_layer(x4, p_dense_blocks_1_layers_4_0_weight, p_dense_blocks_1_layers_4_0_bias,