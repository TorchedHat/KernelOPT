import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm (inference) + ReLU kernel
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
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b
    # ReLU
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Tiled GEMM kernel for the final linear layer: C = A @ W^T + b
# A: [M, K], W: [N, K], b: [N], C: [M, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    a_ptr, w_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(a, w, acc)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc, mask=out_mask)


def fused_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    """Apply BatchNorm (inference mode) + ReLU using Triton kernel."""
    out = torch.empty_like(x)
    numel = x.numel()
    # x is NCHW: channel is dim 1
    N, C, H, W = x.shape
    spatial = H * W
    
    # Reshape to make channel indexing work: need contiguous
    x_flat = x.contiguous()
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x_flat, out,
        mean, var, weight, bias,
        numel, C, spatial,
        eps,
    )
    return out


def triton_linear(a, w, b):
    """Linear layer: a @ w.T + b using Triton GEMM."""
    M, K = a.shape
    N, K2 = w.shape
    assert K == K2
    
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        a, w, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        w.stride(0), w.stride(1),
    )
    return c


def dense_layer_forward(x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight, eps=1e-5):
    """BN + ReLU + Conv2d + (no dropout in inference)"""
    # BN + ReLU
    out = fused_bn_relu(x, bn_mean, bn_var, bn_weight, bn_bias, eps)
    # Conv2d (3x3, padding=1)
    out = F.conv2d(out, conv_weight, bias=None, padding=1)
    return out


def transition_forward(x, bn_weight, bn_bias, bn_mean, bn_var, conv_weight, eps=1e-5):
    """BN + ReLU + Conv2d (1x1) + AvgPool2d"""
    out = fused_bn_relu(x, bn_mean, bn_var, bn_weight, bn_bias, eps)
    out = F.conv2d(out, conv_weight, bias=None)
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
    eps = 1e-5

    #