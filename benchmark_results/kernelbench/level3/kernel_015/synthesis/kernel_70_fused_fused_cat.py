import torch
import triton
import triton.language as tl

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
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute channel index
    c = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


def bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
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


def kernel_function(*args):
    import torch.nn.functional as F
    
    idx = 0
    p_features_0_weight = args[idx]; idx+=1
    p_features_1_weight = args[idx]; idx+=1
    p_features_1_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_0_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_0_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_0_2_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_1_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_1_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_1_2_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_2_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_2_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_2_2_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_3_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_3_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_3_2_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_4_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_4_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_4_2_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_5_0_weight = args[idx]; idx+=1
    p_dense_blocks_0_layers_5_0_bias = args[idx]; idx+=1
    p_dense_blocks_0_layers_5_2_weight = args[idx]; idx+=1
    p_transition_layers_0_transition_0_weight = args[idx]; idx+=1
    p_transition_layers_0_transition_0_bias = args[idx]; idx+=1
    p_transition_layers_0_transition_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_0_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_0_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_0_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_1_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_1_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_1_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_2_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_2_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_2_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_3_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_3_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_3_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_4_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_4_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_4_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_5_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_5_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_5_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_6_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_6_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_6_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_7_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_7_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_7_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_8_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_8_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_8_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_9_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_9_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_9_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_10_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_10_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_10_2_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_11_0_weight = args[idx]; idx+=1
    p_dense_blocks_1_layers_11_0_bias = args[idx]; idx+=1
    p_dense_blocks_1_layers_11_2_weight = args[idx]; idx+=1
    p_transition_layers_1_transition_0_weight = args[idx]; idx+=1
    p_transition_layers_1_transition_0_bias = args[idx]; idx+=1
    p_transition_layers_1_transition_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_0_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_0_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_0_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_1_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_1_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_1_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_2_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_2_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_2_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_3_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_3_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_3_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_4_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_4_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_4_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_5_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_5_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_5_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_6_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_6_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_6_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_7_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_7_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_7_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_8_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_8_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_8_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_9_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_9_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_9_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_10_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_10_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_10_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_11_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_11_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_11_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_12_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_12_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_12_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_13_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_13_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_13_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_14_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_14_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_14_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_15_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_15_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_15_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_16_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_16_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_16_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_17_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_17_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_17_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_18_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_18_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_18_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_19_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_19_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_19_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_20_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_20_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_20_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_21_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_21_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_21_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_22_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_22_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_22_2_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_23_0_weight = args[idx]; idx+=1
    p_dense_blocks_2_layers_23_0_bias = args[idx]; idx+=1
    p_dense_blocks_2_layers_23_2_weight = args[idx]; idx+=1
    p_transition_layers_2_transition_0_weight = args[idx]; idx+=1
    p_transition_layers_2_transition_0_bias = args[idx]; idx+=1
    p_transition_layers_2_transition_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_0_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_0_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_0_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_1_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_1_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_1_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_2_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_2_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_2_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_3_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_3_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_3_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_4_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_4_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_4_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_5_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_5_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_5_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_6_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_6_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_6_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_7_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_7_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_7_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_8_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_8_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_8_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_9_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_9_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_9_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_10_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_10_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_10_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_11_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_11_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_11_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_12_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_12_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_12_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_13_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_13_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_13_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_14_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_14_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_14_2_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_15_0_weight = args[idx]; idx+=1
    p_dense_blocks_3_layers_15_0_bias = args[idx]; idx+=1
    p_dense_blocks_3_layers_15_2_weight = args[idx]; idx+=1
    p_final_bn_weight = args[idx]; idx+=1
    p_final_bn_bias = args[idx]; idx+=1
    p_classifier_weight = args[idx]; idx+=1
    p_classifier_bias = args[idx]; idx+=1
    b_features_1_running_mean = args[idx]; idx+=1
    b_features_1_running_var = args[idx]; idx+=1
    b_dense_blocks_0_layers_0_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_0_layers_0_0_running_var = args[idx]; idx+=1
    b_dense_blocks_0_layers_1_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_0_layers_1_0_running_var = args[idx]; idx+=1
    b_dense_blocks_0_layers_2_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_0_layers_2_0_running_var = args[idx]; idx+=1
    b_dense_blocks_0_layers_3_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_0_layers_3_0_running_var = args[idx]; idx+=1
    b_dense_blocks_0_layers_4_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_0_layers_4_0_running_var = args[idx]; idx+=1
    b_dense_blocks_0_layers_5_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_0_layers_5_0_running_var = args[idx]; idx+=1
    b_transition_layers_0_transition_0_running_mean = args[idx]; idx+=1
    b_transition_layers_0_transition_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_0_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_0_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_1_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_1_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_2_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_2_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_3_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_3_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_4_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_4_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_5_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_5_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_6_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_6_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_7_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_7_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_8_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_8_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_9_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_9_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_10_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_10_0_running_var = args[idx]; idx+=1
    b_dense_blocks_1_layers_11_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_1_layers_11_0_running_var = args[idx]; idx+=1
    b_transition_layers_1_transition_0_running_mean = args[idx]; idx+=1
    b_transition_layers_1_transition_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_0_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_0_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_1_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_1_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_2_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_2_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_3_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_3_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_4_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_4_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_5_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_5_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_6_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_6_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_7_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_7_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_8_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_8_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_9_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_9_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_10_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_10_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_11_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_11_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_12_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_12_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_13_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_13_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_14_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_14_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_15_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_15_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_16_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_16_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_17_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_17_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_18_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_18_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_19_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_19_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_20_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_20_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_21_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_21_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_22_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_22_0_running_var = args[idx]; idx+=1
    b_dense_blocks_2_layers_23_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_2_layers_23_0_running_var = args[idx]; idx+=1
    b_transition_layers_2_transition_0_running_mean = args[idx]; idx+=1
    b_transition_layers_2_transition_0_running_var = args[idx]; idx+=1
    b_dense_blocks_3_layers_0_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_3_layers_0_0_running_var = args[idx]; idx+=1
    b_dense_blocks_3_layers_1_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_3_layers_1_0_running_var = args[idx]; idx+=1
    b_dense_blocks_3_layers_2_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_3_layers_2_0_running_var = args[idx]; idx+=1
    b_dense_blocks_3_layers_3_0_running_mean = args[idx]; idx+=1
    b_dense_blocks_3_layers_3_0_running_var = args[idx]; idx+=1