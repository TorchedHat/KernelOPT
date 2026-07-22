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
    c = (offsets // spatial) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Fused BatchNorm + Add + (optional ReLU) kernel
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
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b + res
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Linear layer kernel (GEMM: [M, K] x [K, N] -> [M, N])
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
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
        
        acc = tl.dot(x, w, acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def apply_bn_relu(x, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5, do_relu=True):
    """Apply batch norm + optional relu using Triton kernel."""
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    spatial = H * W
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        bn_mean, bn_var, bn_weight, bn_bias,
        numel, C, spatial,
        eps,
        do_relu,
    )
    return out


def apply_bn_add(x, residual, bn_weight, bn_bias, bn_mean, bn_var, eps=1e-5):
    """Apply batch norm + add residual using Triton kernel."""
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    spatial = H * W
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_kernel[grid](
        x, residual, out,
        bn_mean, bn_var, bn_weight, bn_bias,
        numel, C, spatial,
        eps,
    )
    return out


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


def channel_shuffle(x, groups):
    """Channel shuffle operation."""
    N, C, H, W = x.shape
    channels_per_group = C // groups
    x = x.view(N, groups, channels_per_group, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(N, C, H, W)
    return x


def shufflenet_unit(x, conv1_w, bn1_w, bn1_b, bn1_rm, bn1_rv,
                    conv2_w, bn2_w, bn2_b, bn2_rm, bn2_rv,
                    conv3_w, bn3_w, bn3_b, bn3_rm, bn3_rv,
                    shortcut_conv_w=None, shortcut_bn_w=None, shortcut_bn_b=None,
                    shortcut_bn_rm=None, shortcut_bn_rv=None,
                    groups=3, eps=1e-5):
    """ShuffleNet unit with fused BN+ReLU."""
    # Save input for residual
    identity = x
    
    # Conv1 + BN + ReLU
    out = F.conv2d(x, conv1_w, bias=None, stride=1, padding=0, groups=groups)
    out = apply_bn_relu(out, bn1_w, bn1_b, bn1_rm, bn1_rv, eps=eps, do_relu=True)
    
    # Conv2 (depthwise) + BN
    mid_channels = conv2_w.shape[0]
    out = F.conv2d(out, conv2_w, bias=None, stride=1, padding=1, groups=mid_channels)
    out = apply_bn_relu(out, bn2_w, bn2_b, bn2_rm, bn2_rv, eps=eps, do_relu=False)
    
    # Channel shuffle
    out = channel_shuffle(out, groups)
    
    # Conv3 + BN
    out = F.conv2d(out, conv3_w, bias=None, stride=1, padding=0, groups=groups)
    
    # Shortcut
    if shortcut_conv_w is not None:
        shortcut = F.conv2d(identity, shortcut_conv_w, bias=None, stride=1, padding=0)
        shortcut = apply_bn_relu(shortcut, shortcut_bn_w, shortcut_bn_b,
                                  shortcut_bn_rm, shortcut_bn_rv, eps=eps, do_relu=False)
    else:
        shortcut = identity
    
    # BN + Add + ReLU (fused)
    out = apply_bn_add(out, shortcut, bn3_w, bn3_b, bn3_rm, bn3_rv, eps=eps)
    out = torch.relu(out)
    
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
    eps = 1e-5
    groups = 3

    # Initial conv + bn + relu
    x = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    x = apply_bn_relu(x, p_bn1_weight, p_bn1_bias, b_bn1_running_mean, b_bn1_running_var, eps=eps, do_relu=True)
    
    # Max pool
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # Stage 2, block 0 (with shortcut conv)
    x = shufflenet_unit(
        x,
        p_stage2_0_conv1_weight, p_stage2_0_bn1_weight, p_stage2_0_bn1_bias,
        b_stage2_0_bn1_running_mean, b_stage2_0_bn1_running_var,
        p_stage2_0_conv2_weight, p_stage2_0_bn2_weight, p_stage2_0_bn2_bias,
        b_stage2_0_bn2_running_mean, b_stage2_0_bn2_running_var,
        p_stage2_0_conv3_weight, p_stage2_0_bn3_weight, p_stage2_0_bn3_bias,
        b_stage2_0_bn3_running_mean, b_stage2_0_bn3_running_var,
        shortcut_conv_w=p_stage2_0_shortcut_0_weight,
        shortcut_bn_w=p_stage2_0_shortcut_1_weight,
        shortcut_bn_b=p_stage2_0_shortcut_1_bias,
        shortcut_bn_rm=b_stage2_0_shortcut_1_running_mean,
        shortcut_bn_rv=b_stage2_0_shortcut_1_running_var,
        groups=groups, eps=eps,
    )
    
    # Stage 2, block 1 (no shortcut conv)
    x = shufflenet_unit(
        x,
        p_stage2_1_conv1_weight, p_stage2_1_bn1_weight, p_stage2_1_bn1_bias,
        b_stage2_1_bn1_running_mean, b_stage2_1_bn1_running_var,
        p_stage2_1_conv2_weight, p_stage2_1_bn2_weight, p_stage2_1_bn2_bias,
        b_stage2_1_bn2_running_mean, b_stage2_1_bn2_running_var,
        p_stage2_1_conv3_weight, p_stage2_1_bn3_weight, p_stage2_1_bn3_bias,
        b_stage2_1_bn3_running_mean, b_stage2_1_bn3_running_var,
        groups=groups, eps=eps,
    )
    
    # Stage 2, block 2 (no shortcut conv)
    x = shufflenet_unit(
        x,
        p_stage2_2_conv1_weight, p_stage2_2_bn1_weight, p_stage2_2_bn1_bias,
        b_stage2_2_bn1_running_mean, b_stage2_2_bn1_running_var,
        p_stage2_2_conv2_weight, p_stage2_2_bn2_weight, p_stage2_2_bn2_bias,
        b_stage2_2_bn2_running_mean, b_stage2_2_bn2_running_var,
        p_stage2_2_conv3_weight, p_stage2_2_bn3_weight, p_stage2_2_bn3_bias,
        b_stage2_2_bn3_running_mean, b_stage2_2_bn3_running_var,
        groups=groups, eps=eps,
    )
    
    # Stage 3, block 0 (with shortcut conv)
    x = shufflenet_unit(
        x,
        p_stage3_0_conv1_weight, p_stage3_0_bn1_weight, p_stage3_0_bn1_bias,
        b_stage3_0_bn1_running_mean, b_stage3_0_bn1_running_var,
        p_stage3_0_conv2_weight, p_stage3_0_bn2_weight, p_stage3_0_bn2_bias,
        b_stage3_0_bn2_running_mean, b_stage3_0_bn2_running_var,
        p_stage3_0_conv3_weight, p_stage3_0_bn3_weight, p_stage3_0_bn3_bias,
        b_stage3_0_bn3_running_mean, b_stage3_0_bn3_running_var,
        shortcut_conv_w=p_stage3_0_shortcut_0_weight,
        shortcut_bn_w=p_stage3_0_shortcut_1_weight,
        shortcut_bn_b=p_stage3_0_shortcut_1_bias,
        shortcut_bn_rm=b_stage3_0_shortcut_1_running_mean,
        shortcut_bn_rv=b_stage3_0_shortcut_1_running_var,
        groups=groups, eps=eps,
    )
    
    # Stage 3, blocks 1-6 (no shortcut conv)
    for conv1_w, bn1_w, bn1_b, bn1_rm, bn1_rv, \
        conv2_w, bn2_w, bn2_b, bn2_rm, bn2_rv, \
        (p_stage3_1_conv1_weight, p_stage3_1_bn1_weight, p_stage3_1_bn1_bias,
         b_stage3_1_bn1_running_mean, b_stage3_1_bn1_running_var,
         p_stage3_1_conv2_weight, p_stage3_1_bn2_weight, p_stage3_1_bn2_bias,
         b_stage3_1_bn2_running_mean, b_stage3_1_bn2_running_var,
         p_stage3_1_conv3_weight, p_stage3_1_bn3_weight, p_stage3_1_bn3_bias,
         b_stage3_1_bn3_running_mean, b_stage3_1_bn3_running_var),
        (p_stage3_2_conv1_weight, p_stage3_2_bn1_weight, p_stage3_2_bn1_bias,
         b_stage3_2_bn1_running_mean, b_stage3_2_bn1_running_var,