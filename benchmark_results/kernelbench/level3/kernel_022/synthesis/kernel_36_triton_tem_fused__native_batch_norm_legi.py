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
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    ACTIVATION: tl.constexpr,  # 0=none, 1=relu, 2=hardtanh6
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index
    c = (offsets // spatial) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_out = x_norm * w + b

    if ACTIVATION == 1:
        x_out = tl.maximum(x_out, 0.0)
    elif ACTIVATION == 2:
        x_out = tl.minimum(tl.maximum(x_out, 0.0), 6.0)

    tl.store(out_ptr + offsets, x_out.to(tl.float32), mask=mask)


# Fused BatchNorm + add + activation kernel
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
def bn_add_kernel(
    x_ptr, residual_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c = (offsets // spatial) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_out = x_norm * w + b + res

    tl.store(out_ptr + offsets, x_out.to(tl.float32), mask=mask)


def bn_act(x, mean, var, weight, bias, activation=0, eps=1e-5):
    """Apply batch norm + optional activation in Triton."""
    out = torch.empty_like(x)
    numel = x.numel()
    spatial = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    num_channels = x.shape[1] if x.dim() >= 2 else 1

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, num_channels, spatial,
        eps,
        ACTIVATION=activation,
    )
    return out


def bn_add(x, residual, mean, var, weight, bias, eps=1e-5):
    """Apply batch norm + residual add in Triton."""
    out = torch.empty_like(x)
    numel = x.numel()
    spatial = x.shape[2] * x.shape[3] if x.dim() == 4 else 1
    num_channels = x.shape[1] if x.dim() >= 2 else 1

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_kernel[grid](
        x, residual, out,
        mean, var, weight, bias,
        numel, num_channels, spatial,
        eps,
    )
    return out


def conv2d_bn_act(x, conv_weight, bn_mean, bn_var, bn_weight, bn_bias,
                  stride=1, padding=0, groups=1, activation=0, eps=1e-5):
    """Conv2d + BN + activation."""
    x = torch.nn.functional.conv2d(x, conv_weight, bias=None,
                                    stride=stride, padding=padding, groups=groups)
    x = bn_act(x, bn_mean, bn_var, bn_weight, bn_bias, activation=activation, eps=eps)
    return x


def conv2d_bn(x, conv_weight, bn_mean, bn_var, bn_weight, bn_bias,
              stride=1, padding=0, groups=1, eps=1e-5):
    """Conv2d + BN (no activation)."""
    x = torch.nn.functional.conv2d(x, conv_weight, bias=None,
                                    stride=stride, padding=padding, groups=groups)
    x = bn_act(x, bn_mean, bn_var, bn_weight, bn_bias, activation=0, eps=eps)
    return x


def conv2d_bn_add(x, conv_weight, bn_mean, bn_var, bn_weight, bn_bias,
                  residual, stride=1, padding=0, groups=1, eps=1e-5):
    """Conv2d + BN + residual add."""
    x = torch.nn.functional.conv2d(x, conv_weight, bias=None,
                                    stride=stride, padding=padding, groups=groups)
    x = bn_add(x, residual, bn_mean, bn_var, bn_weight, bn_bias, eps=eps)
    return x


# Linear kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
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
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_linear(x, weight, bias):
    """Linear layer using Triton."""
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


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_blocks_0_depthwise_conv_0_weight, p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
    p_blocks_0_project_conv_0_weight, p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
    p_blocks_1_expand_conv_0_weight, p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
    p_blocks_1_depthwise_conv_0_weight, p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
    p_blocks_1_project_conv_0_weight, p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
    p_blocks_2_expand_conv_0_weight, p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
    p_blocks_2_depthwise_conv_0_weight, p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
    p_blocks_2_project_conv_0_weight, p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
    p_blocks_3_expand_conv_0_weight, p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
    p_blocks_3_depthwise_conv_0_weight, p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
    p_blocks_3_project_conv_0_weight, p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
    p_blocks_4_expand_conv_0_weight, p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
    p_blocks_4_depthwise_conv_0_weight, p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
    p_blocks_4_project_conv_0_weight, p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,
    p_blocks_5_expand_conv_0_weight, p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias,
    p_blocks_5_depthwise_conv_0_weight, p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias,
    p_blocks_5_project_conv_0_weight, p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_1_bias,
    p_blocks_6_expand_conv_0_weight, p_blocks_6_expand_conv_1_weight, p_blocks_6_expand_conv_1_bias,
    p_blocks_6_depthwise_conv_0_weight, p_blocks_6_depthwise_conv_1_weight, p_blocks_6_depthwise_conv_1_bias,
    p_blocks_6_project_conv_0_weight, p_blocks_6_project_conv_1_weight, p_blocks_6_project_conv_1_bias,
    p_blocks_7_expand_conv_0_weight, p_blocks_7_expand_conv_1_weight, p_blocks_7_expand_conv_1_bias,
    p_blocks_7_depthwise_conv_0_weight, p_blocks_7_depthwise_conv_1_weight, p_blocks_7_depthwise_conv_1_bias,
    p_blocks_7_project_conv_0_weight, p_blocks_7_project_conv_1_weight, p_blocks_7_project_conv_1_bias,
    p_blocks_8_expand_conv_0_weight, p_blocks_8_expand_conv_1_weight, p_blocks_8_expand_conv_1_bias,
    p_blocks_8_depthwise_conv_0_weight, p_blocks_8_depthwise_conv_1_weight, p_blocks_8_depthwise_conv_1_bias,
    p_blocks_8_project_conv_0_weight, p_blocks_8_project_conv_1_weight, p_blocks_8_project_conv_1_bias,
    p_blocks_9_expand_conv_0_weight, p_blocks_9_expand_conv_1_weight, p_blocks_9_expand_conv_1_bias,
    p_blocks_9_depthwise_conv_0_weight, p_blocks_9_depthwise_conv_1_weight, p_blocks_9_depthwise_conv_1_bias,
    p_blocks_9_project_conv_0_weight, p_blocks_9_project_conv_1_weight, p_blocks_9_project_conv_1_bias,
    p_blocks_10_expand_conv_0_weight, p_blocks_10_expand_conv_1_weight, p_blocks_10_expand_conv_1_bias,
    p_blocks_10_depthwise_conv_0_weight, p_blocks_10_depthwise_conv_1_weight, p_blocks_10_depthwise_conv_1_bias,
    p_blocks_10_project_conv_0_weight, p_blocks_10_project_conv_1_weight, p_blocks_10_project_conv_1_bias,
    p_blocks_11_expand_conv_0_weight, p_blocks_11_expand_conv_1_weight, p_blocks_11_expand_conv_1_bias,
    p_blocks_11_depthwise_conv_0_weight, p_blocks_11_depthwise_conv_1_weight, p_blocks_11_depthwise_conv_1_bias,
    p_blocks_11_project_conv_0_weight, p_blocks_11_project_conv_1_weight, p_blocks_11_project_conv_1_bias,
    p_blocks_12_expand_conv_0_weight, p_blocks_12_expand_conv_1_weight, p_blocks_12_expand_conv_1_bias,
    p_blocks_12_depthwise_conv_0_weight, p_blocks_12_depthwise_conv_1_weight, p_blocks_12_depthwise_conv_1_bias,
    p_blocks_12_project_conv_0_weight, p_blocks_12_project_conv_1_weight, p_blocks_12_project_conv_1_bias,
    p_conv2_weight, p_bn2_weight, p_bn2_bias,
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
    x = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    x = bn_act(x, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias, activation=1)

    # Block 0: MBConv1 (32->16, stride=1, expand=1)
    # No expand conv (expand_ratio=1)
    # Depthwise conv
    x = F.conv2d(x, p_blocks_0_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=32)
    x = bn_act(x, b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
               p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias, activation=2)
    # Project conv
    x = F.conv2d(x, p_blocks_0_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = bn_act(x, b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
               p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias, activation=0)

    # Block 1: MBConv6 (16->24, stride=2, expand=6)
    # Expand conv
    x = F.conv2d(x, p_blocks_1_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x = bn_act(x, b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
               p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias, activation=2)
    # Depthwise conv
    x = F.conv2d(x, p_blocks_1_depthwise_conv_0_weight, bias=None, stride=2, padding=1, groups=96)
    x = bn_act(x, b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
               p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias, activation=2)
    # Project conv
    x = F.conv2d(x, p_blocks_1_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = bn_act(x, b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
               p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias, activation=0)

    # Block 2: MBConv6 (24->24, stride=1, expand=6) - has residual
    residual = x
    x = F.conv2d(x, p_blocks_2_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x = bn_act(x, b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
               p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias, activation=2)
    x = F.conv2d(x, p_blocks_2_depthwise_conv_0_weight, bias=None, stride=1, padding=1, groups=144)
    x = bn_act(x, b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
               p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias, activation=2)
    x = F.conv2d(x, p_blocks_2_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = bn_add(x, residual, b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,

    # Block 3: MBConv6 (24->40, stride=2, expand=6, kernel=5)
    x = F.conv2d(x, p_blocks_3_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x = bn_act(x, b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
               p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias, activation=2)
    x = F.conv2d(x, p_blocks_3_depthwise_conv_0_weight, bias=None, stride=2, padding=2, groups=144)
    x = bn_act(x, b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
               p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias, activation=2)
    x = F.conv2d(x, p_blocks_3_project_conv_0_weight, bias=None, stride=1, padding=0)
    x = bn_act(x, b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
               p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias, activation=0)

    # Block 4: MBConv6 (40->40, stride=1, expand=6, kernel=5) - has residual
    residual = x
    x = F.conv2d(x, p_blocks_4_expand_conv_0_weight, bias=None, stride=1, padding=0)
    x = bn_act(x, b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
               p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias, activation=2)
    x = F.conv2d(x, p_blocks_4_depthwise_conv_0_weight, bias=None, stride=1, padding=2, groups=240)
    x = bn_act(x, b_