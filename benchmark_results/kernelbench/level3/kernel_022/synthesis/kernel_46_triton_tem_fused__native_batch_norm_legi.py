import torch
import triton
import triton.language as tl


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
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    ACTIVATION: tl.constexpr,  # 0=none, 1=relu, 2=hardtanh
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
    var = tl.load(var_ptr + chan, mask=mask, other=0.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)

    x = x.to(tl.float32)
    mean = mean.to(tl.float32)
    var = var.to(tl.float32)
    w = w.to(tl.float32)
    b = b.to(tl.float32)

    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std
    x_out = x_norm * w + b

    if ACTIVATION == 1:
        x_out = tl.maximum(x_out, 0.0)
    elif ACTIVATION == 2:
        x_out = tl.minimum(tl.maximum(x_out, 0.0), 6.0)

    tl.store(out_ptr + offsets, x_out, mask=mask)


def apply_bn_act(x, mean, var, weight, bias, eps=1e-5, activation=0):
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
        activation,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
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

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    bias = tl.load(bias_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m_out[:, None] * stride_om + offs_n_out[None, :] * stride_on
    tl.store(out_ptrs, acc, mask=out_mask)


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


def conv2d_bn_act(x, conv_w, bn_mean, bn_var, bn_w, bn_b,
                  stride=1, padding=0, groups=1, dilation=1,
                  activation=0, eps=1e-5):
    # Use PyTorch's cuDNN for convolution
    import torch.nn.functional as F
    x = F.conv2d(x, conv_w, bias=None, stride=stride, padding=padding,
                 groups=groups, dilation=dilation)
    x = apply_bn_act(x, bn_mean, bn_var, bn_w, bn_b, eps=eps, activation=activation)
    return x


def mbconv_block(x, expand_w, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b,
                 dw_w, dw_bn_mean, dw_bn_var, dw_bn_w, dw_bn_b,
                 proj_w, proj_bn_mean, proj_bn_var, proj_bn_w, proj_bn_b,
    import torch.nn.functional as F
    identity = x
    in_channels = x.shape[1]
    expanded_channels = in_channels * expand_ratio

    if expand_ratio != 1:
        # Expand conv
        x = F.conv2d(x, expand_w, bias=None, stride=1, padding=0)
        x = apply_bn_act(x, expand_bn_mean, expand_bn_var, expand_bn_w, expand_bn_b, activation=2)

    # Depthwise conv
    pad = (kernel_size - 1) // 2
    x = F.conv2d(x, dw_w, bias=None, stride=stride, padding=pad, groups=expanded_channels)
    x = apply_bn_act(x, dw_bn_mean, dw_bn_var, dw_bn_w, dw_bn_b, activation=2)

    # Project conv
    x = F.conv2d(x, proj_w, bias=None, stride=1, padding=0)
    x = apply_bn_act(x, proj_bn_mean, proj_bn_var, proj_bn_w, proj_bn_b, activation=0)

    if use_residual:
        x = x + identity

    return x


def kernel_function(
    p_conv1_weight,
    p_bn1_weight, p_bn1_bias,
    p_blocks_0_depthwise_conv_0_weight,
    p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
    p_blocks_0_project_conv_0_weight,
    p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
    p_blocks_1_expand_conv_0_weight,
    p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
    p_blocks_1_depthwise_conv_0_weight,
    p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
    p_blocks_1_project_conv_0_weight,
    p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
    p_blocks_2_expand_conv_0_weight,
    p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
    p_blocks_2_depthwise_conv_0_weight,
    p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
    p_blocks_2_project_conv_0_weight,
    p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
    p_blocks_3_expand_conv_0_weight,
    p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
    p_blocks_3_depthwise_conv_0_weight,
    p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
    p_blocks_3_project_conv_0_weight,
    p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
    p_blocks_4_expand_conv_0_weight,
    p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
    p_blocks_4_depthwise_conv_0_weight,
    p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
    p_blocks_4_project_conv_0_weight,
    p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,
    p_blocks_5_expand_conv_0_weight,
    p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias,
    p_blocks_5_depthwise_conv_0_weight,
    p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias,
    p_blocks_5_project_conv_0_weight,
    p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_1_bias,
    p_blocks_6_expand_conv_0_weight,
    p_blocks_6_expand_conv_1_weight, p_blocks_6_expand_conv_1_bias,
    p_blocks_6_depthwise_conv_0_weight,
    p_blocks_6_depthwise_conv_1_weight, p_blocks_6_depthwise_conv_1_bias,
    p_blocks_6_project_conv_0_weight,
    p_blocks_6_project_conv_1_weight, p_blocks_6_project_conv_1_bias,
    p_blocks_7_expand_conv_0_weight,
    p_blocks_7_expand_conv_1_weight, p_blocks_7_expand_conv_1_bias,
    p_blocks_7_depthwise_conv_0_weight,
    p_blocks_7_depthwise_conv_1_weight, p_blocks_7_depthwise_conv_1_bias,
    p_blocks_7_project_conv_0_weight,
    p_blocks_7_project_conv_1_weight, p_blocks_7_project_conv_1_bias,
    p_blocks_8_expand_conv_0_weight,
    p_blocks_8_expand_conv_1_weight, p_blocks_8_expand_conv_1_bias,
    p_blocks_8_depthwise_conv_0_weight,
    p_blocks_8_depthwise_conv_1_weight, p_blocks_8_depthwise_conv_1_bias,
    p_blocks_8_project_conv_0_weight,
    p_blocks_8_project_conv_1_weight, p_blocks_8_project_conv_1_bias,
    p_blocks_9_expand_conv_0_weight,
    p_blocks_9_expand_conv_1_weight, p_blocks_9_expand_conv_1_bias,
    p_blocks_9_depthwise_conv_0_weight,
    p_blocks_9_depthwise_conv_1_weight, p_blocks_9_depthwise_conv_1_bias,
    p_blocks_9_project_conv_0_weight,
    p_blocks_9_project_conv_1_weight, p_blocks_9_project_conv_1_bias,
    p_blocks_10_expand_conv_0_weight,
    p_blocks_10_expand_conv_1_weight, p_blocks_10_expand_conv_1_bias,
    p_blocks_10_depthwise_conv_0_weight,
    p_blocks_10_depthwise_conv_1_weight, p_blocks_10_depthwise_conv_1_bias,
    p_blocks_10_project_conv_0_weight,
    p_blocks_10_project_conv_1_weight, p_blocks_10_project_conv_1_bias,
    p_blocks_11_expand_conv_0_weight,
    p_blocks_11_expand_conv_1_weight, p_blocks_11_expand_conv_1_bias,
    p_blocks_11_depthwise_conv_0_weight,
    p_blocks_11_depthwise_conv_1_weight, p_blocks_11_depthwise_conv_1_bias,
    p_blocks_11_project_conv_0_weight,
    p_blocks_11_project_conv_1_weight, p_blocks_11_project_conv_1_bias,
    p_blocks_12_expand_conv_0_weight,
    p_blocks_12_expand_conv_1_weight, p_blocks_12_expand_conv_1_bias,
    p_blocks_12_depthwise_conv_0_weight,
    p_blocks_12_depthwise_conv_1_weight, p_blocks_12_depthwise_conv_1_bias,
    p_blocks_12_project_conv_0_weight,
    p_blocks_12_project_conv_1_weight, p_blocks_12_project_conv_1_bias,
    p_conv2_weight,
    p_bn2_weight, p_bn2_bias,
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
    x = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    x = apply_bn_act(x, b_bn1_running_mean, b_bn1_running_var,
                     p_bn1_weight, p_bn1_bias, activation=1)

    # Block 0: MBConv1 (32->16, k=3, s=1, expand=1, no residual since channels differ)
    # expand_ratio=1 means no expand conv
    x = mbconv_block(
        x,
        p_blocks_0_depthwise_conv_0_weight,
        b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
        p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
        p_blocks_0_project_conv_0_weight,
        b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
        p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
        stride=1, kernel_size=3, expand_ratio=1, use_residual=False,
    )

    # Block 1: MBConv6 (16->24, k=3, s=2, expand=6, no residual since stride=2)
    x = mbconv_block(
        x,
        p_blocks_1_expand_conv_0_weight,
        b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
        p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
        p_blocks_1_depthwise_conv_0_weight,
        b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
        p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
        p_blocks_1_project_conv_0_weight,
        b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
        p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
        stride=2, kernel_size=3, expand_ratio=6, use_residual=False,
    )

    # Block 2: MBConv6 (24->24, k=3, s=1, expand=6, residual)
    x = mbconv_block(
        x,
        p_blocks_2_expand_conv_0_weight,
        b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
        p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
        p_blocks_2_depthwise_conv_0_weight,
        b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
        p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
        p_blocks_2_project_conv_0_weight,
        b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
        p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
        stride=1, kernel_size=3, expand_ratio=6, use_residual=True,
    )

    # Block 3: MBConv6 (24->40, k=5, s=2, expand=6, no residual)
    x = mbconv_block(
        x,
        p_blocks_3_expand_conv_0_weight,
        b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
        p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
        p_blocks_3_depthwise_conv_0_weight,
        b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
        p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
        p_blocks_3_project_conv_0_weight,
        b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
        p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
        stride=2, kernel_size=5, expand_ratio=6, use_residual=False,
    )

    # Block 4: MBConv6 (40->40, k=5, s=1, expand=6, residual)
    x = mbconv_block(
        x,
        p_blocks_4_expand_conv_0_weight,
        b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
        p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
        p_blocks_4_depthwise_conv_0_weight,
        b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var,
        p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
        p_blocks_4_project_conv_0_weight,
        b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var,
        p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,
        stride=1, kernel_size=5, expand_ratio=6, use_residual=True,
    )

    # Block 5: MBConv6 (40->80, k=3, s=2, expand=6, no residual)
    x = mbconv_block(
        x,
        p_blocks_5_expand_conv_0_weight,
        b_blocks_5_expand_conv_1_running_mean, b_blocks_5_expand_conv_1_running_var,
        p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias,
        p_blocks_5_depthwise_conv_0_weight,
        b_blocks_5_depthwise_conv_1_running_mean, b_blocks_5_depthwise_conv_1_running_var,
        p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias,
        p_blocks_5_project_conv_0_weight,
        b_blocks_5_project_conv_1_running_mean, b_blocks_5_project_conv_1_running_var,
        p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_1_bias,
        stride=2, kernel_size=3, expand_ratio=6, use_residual=False,
    )

    # Block 6: MBConv6 (80->80, k=3, s=1, expand=6, residual)
    x = mbconv_block(
        x,
        p_blocks_6_expand_conv_0_weight,