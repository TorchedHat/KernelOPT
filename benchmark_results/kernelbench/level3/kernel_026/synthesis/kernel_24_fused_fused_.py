import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
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

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channel index: offsets // spatial % num_channels
    # For NCHW layout: index = n * C * H * W + c * H * W + h * W + w
    # channel = (offsets // spatial) % num_channels
    chan = (offsets // spatial) % num_channels

    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    chan = (offsets // spatial) % num_channels

    mean = tl.load(mean_ptr + chan, mask=mask, other=0.0)
    var = tl.load(var_ptr + chan, mask=mask, other=1.0)
    w = tl.load(weight_ptr + chan, mask=mask, other=1.0)
    b = tl.load(bias_ptr + chan, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def relu_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def add_relu_kernel(
    x_ptr, y_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    out = tl.maximum(x + y, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['N', 'C', 'H', 'W', 'groups'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    N, C, H, W, groups,
    BLOCK_SIZE: tl.constexpr,
):
    """
    """
    pid = tl.program_id(0)
    numel = N * C * H * W
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Decode output index (n, c_out, h, w) where c_out = cpg * groups + g
    w_idx = offsets % W
    tmp = offsets // W
    h_idx = tmp % H
    tmp = tmp // H
    c_out = tmp % C
    n_idx = tmp // C

    cpg = C // groups
    # c_out = cpg_idx * groups + g_idx
    cpg_idx = c_out % cpg
    g_idx = c_out // cpg

    # Input index: n, g_idx, cpg_idx, h, w
    in_idx = n_idx * (C * H * W) + g_idx * (cpg * H * W) + cpg_idx * (H * W) + h_idx * W + w_idx

    val = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    KH, KW, SH, SW, PH, PW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    numel = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    w_idx = offsets % OW
    tmp = offsets // OW
    h_idx = tmp % OH
    tmp = tmp // OH
    c_idx = tmp % C
    n_idx = tmp // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = h_idx * SH + kh - PH
            iw = w_idx * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            in_idx = n_idx * (C * IH * IW) + c_idx * (IH * IW) + ih * IW + iw
            val = tl.load(x_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['N', 'C'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, IH, IW,
    BLOCK_SIZE: tl.constexpr,
):
    """Adaptive avg pool to 1x1"""
    pid = tl.program_id(0)
    numel = N * C
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c_idx = offsets % C
    n_idx = offsets // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    total = IH * IW

    for h in range(IH):
        for w in range(IW):
            in_idx = n_idx * (C * IH * IW) + c_idx * (IH * IW) + h * IW + w
            val = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
            acc = acc + val

    out = acc / total
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['M', 'N'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple linear layer: out = x @ w.T + b, x: (M, K), w: (N, K), out: (M, N)"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M * N

    n_idx = offsets % N
    m_idx = offsets // N

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for k in range(K):
        x_val = tl.load(x_ptr + m_idx * K + k, mask=mask, other=0.0)
        w_val = tl.load(w_ptr + n_idx * K + k, mask=mask, other=0.0)
        acc = acc + x_val * w_val

    b_val = tl.load(b_ptr + n_idx, mask=mask, other=0.0)
    out = acc + b_val
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    spatial = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def triton_bn(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    spatial = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_kernel[grid](
        x, out, mean, var, weight, bias,
    )
    return out


def triton_channel_shuffle(x, groups):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid](
    )
    return out


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, IH, IW = x.shape
    OH = (IH + 2 * padding - kernel_size) // stride + 1
    OW = (IW + 2 * padding - kernel_size) // stride + 1
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    numel = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out, N, C, IH, IW, OH, OW,
    )
    return out


def triton_adaptive_avg_pool2d(x):
    N, C, IH, IW = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    numel = N * C
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
    )
    return out


def triton_linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    numel = M * N
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    linear_kernel[grid](
    )
    return out


def shufflenet_block(x, conv1_w, bn1_mean, bn1_var, bn1_w, bn1_b,
                     conv2_w, bn2_mean, bn2_var, bn2_w, bn2_b,
                     conv3_w, bn3_mean, bn3_var, bn3_w, bn3_b,
                     groups, has_shortcut=False,
                     sc_conv_w=None, sc_bn_mean=None, sc_bn_var=None, sc_bn_w=None, sc_bn_b=None):
    # conv1: 1x1 group conv
    out = F.conv2d(x, conv1_w, bias=None, stride=1, padding=0, groups=groups)
    out = triton_bn_relu(out, bn1_mean, bn1_var, bn1_w, bn1_b)

    # conv2: depthwise 3x3
    mid_channels = out.shape[1]
    out = F.conv2d(out, conv2_w, bias=None, stride=1, padding=1, groups=mid_channels)
    out = triton_bn(out, bn2_mean, bn2_var, bn2_w, bn2_b)

    # channel shuffle
    out = triton_channel_shuffle(out, groups)

    # conv3: 1x1 group conv
    out = F.conv2d(out, conv3_w, bias=None, stride=1, padding=0, groups=groups)
    out = triton_bn_relu(out, bn3_mean, bn3_var, bn3_w, bn3_b)

    if has_shortcut:
        shortcut = F.conv2d(x, sc_conv_w, bias=None, stride=1, padding=0)
        shortcut = triton_bn(shortcut, sc_bn_mean, sc_bn_var, sc_bn_w, sc_bn_b)
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
    groups = 3
    eps = 1e-5

    # Initial conv + bn + relu + maxpool
    out = F.conv2d(x, p_conv1_weight, bias=None, stride=2, padding=1)
    out = triton_bn_relu(out, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias, eps)
    out = triton_max_pool2d(out, kernel_size=3, stride=2, padding=1)

    # Stage 2, block 0 (has shortcut with conv)
    out = shuff