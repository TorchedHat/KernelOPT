import torch
import triton
import triton.language as tl
import math

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

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


# Fused BatchNorm + HardTanh (clamp to [-6, 6] for ReLU6) kernel
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
def bn_hardtanh_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    min_val, max_val,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    chan = (offsets // spatial) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.minimum(tl.maximum(out, min_val), max_val)

    tl.store(out_ptr + offsets, out, mask=mask)


# Fused BatchNorm only (no activation) kernel
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
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    chan = (offsets // spatial) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    tl.store(out_ptr + offsets, out, mask=mask)


# Linear kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4),
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

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc += tl.dot(x, w)

    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += b[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def fused_bn_relu(x, running_mean, running_var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, num_channels, spatial,
        eps,
    )
    return out


def fused_bn_hardtanh(x, running_mean, running_var, weight, bias, eps=1e-5, min_val=0.0, max_val=6.0):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_hardtanh_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, num_channels, spatial,
        eps,
        min_val, max_val,
    )
    return out


def fused_bn(x, running_mean, running_var, weight, bias, eps=1e-5):
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, num_channels, spatial,
        eps,
    )
    return out


def triton_linear(x, weight, bias):
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


def conv2d(x, weight, bias=None, stride=1, padding=0, groups=1):
    import torch.nn.functional as F
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, groups=groups)


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
    import torch.nn.functional as F

    # Initial conv + BN + ReLU
    x = conv2d(x, p_conv1_weight, stride=2, padding=1)
    x = fused_bn_relu(x, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias)

    # Block 0: MBConv1 (32->16, stride=1, expand=1)
    # depthwise conv (groups=32)
    identity = x
    x = conv2d(x, p_blocks_0_depthwise_conv_0_weight, stride=1, padding=1, groups=32)
    x = fused_bn_hardtanh(x, b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
    x = conv2d(x, p_blocks_0_project_conv_0_weight, stride=1, padding=0)
    x = fused_bn(x, b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
    # No residual for block 0 (in_channels != out_channels: 32 != 16)

    # Block 1: MBConv6 (16->24, stride=2, expand=6)
    x = conv2d(x, p_blocks_1_expand_conv_0_weight, stride=1, padding=0)
    x = fused_bn_hardtanh(x, b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
    x = conv2d(x, p_blocks_1_depthwise_conv_0_weight, stride=2, padding=1, groups=96)
    x = fused_bn_hardtanh(x, b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
    x = conv2d(x, p_blocks_1_project_conv_0_weight, stride=1, padding=0)
    x = fused_bn(x, b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
    # No residual (stride=2)

    # Block 2: MBConv6 (24->24, stride=1, expand=6)
    identity = x
    x = conv2d(x, p_blocks_2_expand_conv_0_weight, stride=1, padding=0)
    x = fused_bn_hardtanh(x, b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
    x = conv2d(x, p_blocks_2_depthwise_conv_0_weight, stride=1, padding=1, groups=144)
    x = fused_bn_hardtanh(x, b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
    x = conv2d(x, p_blocks_2_project_conv_0_weight, stride=1, padding=0)
    x = fused_bn(x, b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
    x = x + identity

    # Block 3: MBConv6 (24->40, stride=2, expand=6, kernel=5)
    x = conv2d(x, p_blocks_3_expand_conv_0_weight, stride=1, padding=0)
    x = fused_bn_hardtanh(x, b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
    x = conv2d(x, p_blocks_3_depthwise_conv_0_weight, stride=2, padding=2, groups=144)
    x = fused_bn_hardtanh(x, b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
    x = conv2d(x, p_blocks_3_project_conv_0_weight, stride=1, padding=0)
    x = fused_bn(x, b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
    # No residual (stride=2)

    # Block 4: MBConv6 (40->40, stride=1, expand=6, kernel=5)
    identity = x
    x = conv2d(x, p_blocks_4_expand_conv_0_weight, stride=1, padding=0)
    x = fused_bn_hardtanh(x, b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
    x = conv2d(x, p_blocks_4_depthwise_conv_0_weight, stride=1, padding