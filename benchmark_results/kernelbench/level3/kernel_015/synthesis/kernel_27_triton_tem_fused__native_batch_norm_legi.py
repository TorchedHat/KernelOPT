import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_relu_kernel(
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
    c = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


import torch
import triton
import triton.language as tl


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
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c = (offsets // spatial) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial,
        eps,
    )
    return out


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
def fused_bn_relu_inplace_kernel(
    x_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c = (offsets // spatial) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)

    tl.store(x_ptr + offsets, out, mask=mask)


def bn_relu_inplace(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_inplace_kernel[grid](
        x,
        running_mean, running_var, weight, bias,
        numel, C, spatial,
        eps,
    )
    return x


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def triton_linear(x, weight, bias=None):
    # x: (M, K), weight: (N, K) -> out: (M, N)
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    bias_ptr = bias if bias is not None else x  # dummy
    matmul_kernel[grid](
        x, weight, out, bias_ptr if bias is not None else x,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
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