import torch
import triton
import triton.language as tl


# Fused BatchNorm + ReLU kernel (pointwise)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    do_relu: tl.constexpr,
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
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Fused BatchNorm + ReLU + Add (residual) kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def bn_add_relu_kernel(
    x_ptr, residual_ptr, out_ptr,
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
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    
    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b + res
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Channel shuffle kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    groups, channels_per_group,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Output index: (n, c_out, h, w)
    # c_out = g * channels_per_group + c_in_group
    # Input index: (n, c_in, h, w) where c_in = c_in_group * groups + g
    
    w_idx = offsets % W
    tmp = offsets // W
    h_idx = tmp % H
    tmp = tmp // H
    c_out = tmp % C
    n_idx = tmp // C
    
    # c_out = g * channels_per_group + c_in_group
    g = c_out // channels_per_group
    c_in_group = c_out % channels_per_group
    # c_in = c_in_group * groups + g
    c_in = c_in_group * groups + g
    
    in_offset = w_idx + W * h_idx + W * H * c_in + W * H * C * n_idx
    
    val = tl.load(x_ptr + in_offset, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


# Adaptive avg pool + flatten kernel (pool to 1x1)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    nc_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    nc_mask = nc_offsets < N * C
    
    n_idx = nc_offsets // C
    c_idx = nc_offsets % C
    
    # Sum over spatial dimension
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # We need to iterate over HW - but HW can be large (56*56=3136)
    # Use a loop
    base = n_idx * C * HW + c_idx * HW
    
    for hw in range(HW):
        val = tl.load(x_ptr + base + hw, mask=nc_mask, other=0.0)
        acc += val
    
    avg = acc / HW
    tl.store(out_ptr + nc_offsets, avg, mask=nc_mask)


# Linear layer kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # w is (N, K), we want (K, N) for the dot
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc=acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def apply_bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    """Apply batch norm + optional relu using Triton kernel."""
    out = torch.empty_like(x)
    numel = x.numel()
    num_channels = x.shape[1]
    spatial = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, num_channels, spatial,
        eps,
        do_relu,
    )
    return out


def apply_bn(x, mean, var, weight, bias, eps=1e-5):
    return apply_bn_relu(x, mean, var, weight, bias, eps, do_relu=False)


def apply_channel_shuffle(x, groups):
    """Apply channel shuffle using Triton kernel."""
    N, C, H, W = x.shape
    channels_per_group = C // groups
    out = torch.empty_like(x)
    numel = x.numel()
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
        x, out,
        N, C, H, W,
        groups, channels_per_group,
        numel,
    )
    return out


def apply_adaptive_avg_pool(x):
    """Apply adaptive avg pool to 1x1 using Triton kernel."""
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(N * C, meta['BLOCK_SIZE']),)
    adaptive_avg_pool_kernel[grid](
        x, out,
        N, C, HW,
    )
    return out.view(N, C, 1, 1)


def apply_linear(x, weight, bias):
    """Apply linear layer using Triton kernel."""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


def conv2d_op(x, weight, stride=1, padding=0, groups=1):
    """Standard conv2d using PyTorch (cuDNN)."""
    return torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


def shufflenet_unit(x, 
                    conv1_w, bn1_mean, bn1_var, bn1_w, bn1_b,
                    conv2_w, bn2_mean, bn2_var, bn2_w, bn2_b,
                    conv3_w, bn3_mean, bn3_var, bn3_w, bn3_b,
                    shortcut_conv_w=None, sc_bn_mean=None, sc_bn_var=None, sc_bn_w=None, sc_bn_b=None,
                    groups=3):
    """ShuffleNet unit forward pass."""
    # First 1x1 group conv + BN + ReLU
    out = conv2d_op(x, conv1_w, stride=1, padding=0, groups=groups)
    out = apply_bn_relu(out, bn1_mean, bn1_var, bn1_w, bn1_b, do_relu=True)
    
    # Depthwise 3x3 conv + BN (no relu before shuffle)
    mid_channels = out.shape[1]
    out = conv2d_op(out, conv2_w, stride=1, padding=1, groups=mid_channels)
    out = apply_bn(out, bn2_mean, bn2_var, bn2_w, bn2_b)
    
    # Channel shuffle
    out = apply_channel_shuffle(out, groups)
    
    # Second 1x1 group conv + BN + ReLU
    out = conv2d_op(out, conv3_w, stride=1, padding=0, groups=groups)
    out = apply_bn_relu(out, bn3_mean, bn3_var, bn3_w, bn3_b, do_relu=True)
    
    # Shortcut
    if shortcut_conv_w is not None:
        shortcut = conv2d_op(x, shortcut_conv_w, stride=1, padding=0, groups=1)
        shortcut = apply_bn(shortcut, sc_bn_mean, sc_bn_var, sc_bn_w, sc_bn_b)
        out = out + shortcut
    else:
        out = out + x
    
    return out


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_stage2_0_conv1_weight, p_stage2_0_bn1_weight, p_stage2_0_bn1_bias,
    p_stage2_0_conv2_weight, p_stage2_0_bn2_weight, p_stage2_0_bn2_bias,
    p_stage2_0_conv3_weight, p_stage2_0_bn3_weight, p_stage2_0_bn3_bias,
    p_stage2_0_shortcut_0_weight, p_stage2_0_shortcut_1_weight, p_stage2_0_shortcut_1_bias,
    p_stage2_1_conv1_weight, p_stage2_1_bn1_weight, p_stage2_1_bn1_bias,
    p_stage2_1_conv2_weight, p_stage2_1_bn2_weight, p_stage2_1_bn2_bias,
    p_stage2_1_conv3_weight, p_stage2_1_bn3_weight, p_stage2_1_bn3_bias,
    p_stage2_2_conv1_weight, p_stage2_2_bn1_weight, p_stage2_2_bn1_bias,
    p_stage2_2_conv2_weight, p_stage2_2_bn2_weight, p_stage2_2_bn2_bias,
    p_stage2_2_conv3_weight, p_stage2_2_bn3_weight, p_stage2_2_bn3_bias,
    p_stage3_0_conv1_weight, p_stage3_0_bn1_weight, p_stage3_0_bn1_bias,
    p_stage3_0_conv2_weight, p_stage3_0_bn2_weight, p_stage3_0_bn2_bias,
    p_stage3_0_conv3_weight, p_stage3_0_bn3_weight, p_stage3_0_bn3_bias,
    p_stage3_0_shortcut_0_weight, p_stage3_0_shortcut_1_weight, p_stage3_0_shortcut_1_bias,
    p_stage3_1_conv1_weight, p_stage3_1_bn1_weight, p_stage3_1_bn1_bias,
    p_stage3_1_conv2_weight, p_stage3_1_bn2_weight, p_stage3_1_bn2_bias,
    p_stage3_1_conv3_weight, p_stage3_1_bn3_weight, p_stage3_1_bn3_bias,
    p_stage3_2_conv1_weight, p_stage3_2_bn1_weight, p_stage3_2_bn1_bias,
    p_stage3_2_conv2_weight, p_stage3_2_bn2_weight, p_stage3_2_bn2_bias,
    p_stage3_2_conv3_weight, p_stage3_2_bn3_weight, p_stage3_2_bn3_bias,
    p_stage3_3_conv1_weight, p_stage3_3_bn1_weight, p_stage3_3_bn1_bias,
    p_stage3_3_conv2_weight, p_stage3_3_bn2_weight, p_stage3_3_bn2_bias,
    p_stage3_3_conv3_weight, p_stage3_3_bn3_weight, p_stage3_3_bn3_bias,
    p_stage3_4_conv1_weight, p_stage3_4_bn1_weight, p_stage3_4_bn1_bias,
    p_stage3_4_conv2_weight, p_stage3_4_bn2_weight, p_stage3_4_bn2_bias,
    p_stage3_4_conv3_weight, p_stage3_4_bn3_weight, p_stage3_4_bn3_bias,
    p_stage3_5_conv1_weight, p_stage3_5_bn1_weight, p_stage3_5_bn1_bias,
    p_stage3_5_conv2_weight, p_stage3_5_bn2_weight, p_stage3_5_bn2_bias,
    p_stage3_5_conv3_weight, p_stage3_5_bn3_weight, p_stage3_5_bn3_bias,
    p_stage3_6_conv1_weight, p_stage3_6_bn1_weight, p_stage3_6_bn1_bias,
    p_stage3_6_conv2_weight, p_stage3_6_bn2_weight, p_stage3_6_bn2_bias,
    p_stage3_6_conv3_weight, p_stage3_6_bn3_weight, p_stage3_6_bn3_bias,
    p_stage4_0_conv1_weight, p_stage4_0_bn1_weight, p_stage4_0_bn1_bias,
    p_stage4_0_conv2_weight, p_stage4_0_bn2_weight, p_stage4_0_bn2_bias,
    p_stage4_0_conv3_weight, p_stage4_0_bn3_weight, p_stage4_0_bn3_bias,
    p_stage4_0_shortcut_0_weight, p_stage4_0_shortcut_1_weight, p_stage4_0_shortcut_1_bias,
    p_stage4_1_conv1_weight, p_stage4_1_bn1_weight, p_stage4_1_bn1_bias,
    p_stage4_1_conv2_weight, p_stage4_1_bn2_weight, p_stage4_1_bn2_bias,
    p_stage4_1_conv3_weight, p_stage4_1_bn3_weight, p_stage4_1_bn3_bias,
    p_stage4_2_conv1_weight, p_stage4_2_bn1_weight, p_stage4_2_bn1_bias,
    p_stage4_2_conv2_weight, p_stage4_2_bn2_weight, p_stage4_2_bn2_bias,
    p_stage4_2_conv3_weight, p_stage4_2_bn3_weight, p_stage4_2_bn3_bias,
    p_conv5_weight, p_bn5_weight, p_bn5_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_stage2_0_bn1_running_mean, b_stage2_0_bn1_running_var,
    b_stage2_0_bn2_running_mean, b_stage2_0_bn2_running_var,
    b_stage2_0_bn3_running_mean, b_stage2_0_bn3_running_var,
    b_stage2_0_shortcut_1_running_mean, b_stage2_0_shortcut_1_running_var,
    b_stage2_1_bn1_running_mean, b_stage2_1_bn1_running_var,
    b_stage2_1_bn2_running_mean, b_stage2_1_bn2_running_var,
    b_stage2_1_bn3_running_mean, b_stage2_1_bn3_running_var,
    b_stage2_2_bn1_running_mean, b_stage2_2_bn1_running_var,
    b_stage2_2_bn2_running_mean, b_stage2_2_bn2_running_var,
    b_stage2_2_bn3_running_mean, b_stage2_2_bn3_running_var,
    b_stage3_0_bn1_running_mean, b_stage3_0_bn1_running_var,
    b_stage3_0_bn2_running_mean, b_stage3_0_bn2_running_var,
    b_stage3_0_bn3_running_mean, b_stage3_0_bn3_running_var,
    b_stage3_0_shortcut_1_running_mean, b_stage3_0_shortcut_1_running_var,
    b_stage3_1_bn1_running_mean, b_stage3_1_bn1_running_var,
    b_stage3_1_bn2_running_mean, b_stage3_1_bn2_running_var,
    b_stage3_1_bn3_running_mean, b_stage3_1_bn3_running_var,
    b_stage3_2_bn1_running_mean, b_stage3_2_bn1_running_var,
    b_stage3_2_bn2_running_mean, b_stage3_2_bn2_running_var,
    b_stage3_2_bn3_running_mean, b_stage3_2_bn3_running_var,
    b_stage3_3_bn1_running_mean, b_stage3_3_bn1_running_var,
    b_stage3_3_bn2_running_mean, b_stage3_3_bn2_running_var,
    b_stage3_3_bn3_running_mean, b_stage3_3_bn3_running_var,
    b_stage3_4_bn1_running_mean, b_stage3_4_bn1_running_var,
    b_stage3_4_bn2_running_mean, b_stage3_4_bn2_running_var,
    b_stage3_4_bn3_running_mean, b_stage3_4_bn3_running_var,
    b_stage3_5_bn1_running_mean, b_stage3_5_bn1_running_var,
    b_stage3_5_bn2_running_mean, b_stage3_5_bn2_running_var,
    b_stage3_5_bn3_running_mean, b_stage3_5_bn3_running_var,
    b_stage3_6_bn1_running_mean, b_stage3_6_bn1_running_var,
    b_stage3_6_bn2_running_mean, b_stage3_6_bn2_running_var,
    b_stage3_6_bn3_running_mean, b_stage3_6_bn3_running_var,
    b_stage4_0_bn1_running_mean, b_stage4_0_bn1_running_var,
    b_stage4_0_bn2_running_mean, b_stage4_0_bn2_running_var,
    b_stage4_0_bn3_running_mean, b_stage4_0_bn3_running_var,
    b_stage4_0_shortcut_1_running_mean, b_stage4_0_shortcut_1_running_var,
    b_stage4_1_bn1_running_mean, b_stage4_1_bn1_running_var,
    b_stage4_1_bn2_running_mean, b_stage4_1_bn2_running_var,
    b_stage4_1_bn3_running_mean, b_stage4_1_bn3_running_var,
    b_stage4_2_bn1_running_mean, b_stage4_2_bn1_running_var,
    b_stage4_2_bn2_running_mean, b_stage4_2_bn2_running_var,
    b_stage4_2_bn3_running_mean, b_stage4_2_bn3_running_var,
    b_bn5_running_mean, b_bn5_running_var,
    x,
):
    # Initial conv + bn + relu + maxpool
    x = conv2d_op(x, p_conv1_weight, stride=2, padding=1, groups=1)
    x = apply_bn_relu(x, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias, do_relu=True)
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # Stage 2 - block 0 (with shortcut projection)
    x = shufflenet_unit(
        x,
        p_stage2_0_conv1_weight, b_stage2_0_bn1_running_mean, b_stage2_0_bn1_running_var,
        p_stage2_0_bn1_weight, p_stage2_0_bn1_bias,
        p_stage2_0_conv2_weight, b_stage2_0_bn2_running_mean, b_stage2_0_bn2_running_var,
        p_stage2_0_bn2_weight, p_stage2_0_bn2_bias,
        p_stage2_0_conv3_weight, b_stage2_0_bn3_running_mean, b_stage2_0_bn3_running_var,
        p_stage2_0_bn3_weight, p_stage2_0_bn3_bias,
        p_stage2_0_shortcut_0_weight, b_stage2_0_shortcut_1_running_mean,
        b_stage2_0_shortcut_1_running_var, p_stage2_0_shortcut_1_weight, p_stage2_0_shortcut_1_bias,
        groups=3,
    )
    
    # Stage 2 - block 1
    x = shufflenet_unit(
        x,
        p_stage2_1_conv1_weight, b_stage2_1_bn1_running_mean, b_stage2_1_bn1_running_var,
        p_stage2_1_bn1_weight, p_stage2_1_bn1_bias,
        p_stage2_1_conv2_weight, b_stage2_1_bn2_running_mean, b_stage2_1_bn2_running_var,
        p_stage2_1_bn2_weight, p_stage2_1_bn2_bias,
        p_stage2_1_conv3_weight, b_stage2_1_bn3_running_mean, b_stage2_1_bn3_running_var,
        p_stage2_1_bn3_weight, p_stage2_1_bn3_bias,
        groups=3,
    )
    
    # Stage 2 - block 2
    x = shufflenet_unit(
        x,
        p_stage2_2_conv1_weight, b_stage2_2_bn1_running_mean, b_stage2_2_bn1_running_var,