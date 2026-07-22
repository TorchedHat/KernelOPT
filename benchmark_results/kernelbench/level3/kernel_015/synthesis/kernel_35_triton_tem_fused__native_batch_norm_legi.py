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
    key=['n_channels', 'spatial'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    weight_ptr, bias_ptr, mean_ptr, var_ptr,
    n_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles BLOCK_SIZE elements
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = n_channels * spatial
    mask = offsets < total

    # Decode channel index
    c = (offsets // spatial) % n_channels

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


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    spatial = H * W
    # x is NCHW, we need to index by channel
    # Flatten to (N*C*H*W) but channel is dim 1
    # Linear index: n*C*H*W + c*H*W + h*W + w
    # channel = (linear_idx // (H*W)) % C  -- but N*C*H*W total
    # Actually channel = (linear_idx % (C*H*W)) // (H*W)
    # Let's reshape to (N*C, H*W) and process
    total = N * C * H * W
    out = torch.empty_like(x)
    
    # We need channel index from flat index
    # flat index i: channel = (i // spatial) % C
    # But N*C*H*W total, so channel = (i // (H*W)) % C
    # spatial = H*W, n_channels = C
    
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, out,
        weight, bias, running_mean, running_var,
        C, spatial,
        eps,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_channels', 'spatial'],
)
@triton.jit
def fused_bn_relu_inplace_kernel(
    x_ptr,
    weight_ptr, bias_ptr, mean_ptr, var_ptr,
    n_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = n_channels * spatial
    mask = offsets < total

    c = (offsets // spatial) % n_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b
    out = tl.maximum(out, 0.0)
    tl.store(x_ptr + offsets, out, mask=mask)


# Fused linear (matmul) kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc, mask=out_mask)


def triton_linear(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def conv2d_bn_relu(x, conv_weight, bn_weight, bn_bias, bn_mean, bn_var,
                   stride=1, padding=0, eps=1e-5):
    # Use cuDNN for conv
    x = F.conv2d(x, conv_weight, bias=None, stride=stride, padding=padding)
    # Fused BN + ReLU
    x = fused_bn_relu(x, bn_weight, bn_bias, bn_mean, bn_var, eps)
    return x


def dense_layer(x, bn_w, bn_b, bn_mean, bn_var, conv_w, eps=1e-5):
    """BN -> ReLU -> Conv -> (no dropout in inference)"""
    # BN + ReLU on input
    out = fused_bn_relu(x, bn_w, bn_b, bn_mean, bn_var, eps)
    # Conv
    out = F.conv2d(out, conv_w, bias=None, stride=1, padding=1)
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