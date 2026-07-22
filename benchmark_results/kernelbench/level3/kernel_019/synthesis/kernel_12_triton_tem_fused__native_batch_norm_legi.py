import torch
import triton
import triton.language as tl
import torch.nn.functional as F


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
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    numel,
    num_channels,
    spatial_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index: layout is (N, C, H, W) -> channel = (offset / spatial_size) % num_channels
    channel_idx = (offsets // spatial_size) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)

    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    # ReLU
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def fused_bn_relu(x, running_mean, running_var, weight, bias, eps=1e-5):
    """Fused BatchNorm + ReLU using Triton kernel."""
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial_size = H * W
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial_size, eps,
    )
    return out


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
def avg_pool_flatten_kernel(
    x_ptr,
    out_ptr,
    numel,
    N,
    C,
    H,
    W,
    BLOCK_SIZE: tl.constexpr,
):
    """Average pool over H x W spatial dims, output shape (N, C)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel  # numel = N * C

    n = offsets // C
    c = offsets % C

    n_mask = mask & (n < N) & (c < C)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    hw = H * W

    # Loop over spatial positions
    for h in range(H):
        for w in range(W):
            idx = n * C * H * W + c * H * W + h * W + w
            val = tl.load(x_ptr + idx, mask=n_mask, other=0.0).to(tl.float32)
            acc = acc + val

    acc = acc / (H * W).to(tl.float32)
    tl.store(out_ptr + offsets, acc, mask=n_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
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

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

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
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def fused_linear(x, weight, bias):
    """Linear layer using Triton tl.dot."""
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


def conv2d_depthwise(x, weight, stride, padding):
    """Depthwise conv2d using PyTorch (groups=in_channels)."""
    in_channels = x.shape[1]
    return F.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=in_channels)


def conv2d_pointwise(x, weight):
    """1x1 pointwise conv2d using PyTorch."""
    return F.conv2d(x, weight, bias=None, stride=1, padding=0)


def conv2d_standard(x, weight, stride, padding):
    """Standard conv2d using PyTorch."""
    return F.conv2d(x, weight, bias=None, stride=stride, padding=padding)


def kernel_function(
    p_model_0_0_weight, p_model_0_1_weight, p_model_0_1_bias,
    p_model_1_0_weight, p_model_1_1_weight, p_model_1_1_bias,
    p_model_1_3_weight, p_model_1_4_weight, p_model_1_4_bias,
    p_model_2_0_weight, p_model_2_1_weight, p_model_2_1_bias,
    p_model_2_3_weight, p_model_2_4_weight, p_model_2_4_bias,
    p_model_3_0_weight, p_model_3_1_weight, p_model_3_1_bias,
    p_model_3_3_weight, p_model_3_4_weight, p_model_3_4_bias,
    p_model_4_0_weight, p_model_4_1_weight, p_model_4_1_bias,
    p_model_4_3_weight, p_model_4_4_weight, p_model_4_4_bias,
    p_model_5_0_weight, p_model_5_1_weight, p_model_5_1_bias,
    p_model_5_3_weight, p_model_5_4_weight, p_model_5_4_bias,
    p_model_6_0_weight, p_model_6_1_weight, p_model_6_1_bias,
    p_model_6_3_weight, p_model_6_4_weight, p_model_6_4_bias,
    p_model_7_0_weight, p_model_7_1_weight, p_model_7_1_bias,
    p_model_7_3_weight, p_model_7_4_weight, p_model_7_4_bias,
    p_model_8_0_weight, p_model_8_1_weight, p_model_8_1_bias,
    p_model_8_3_weight, p_model_8_4_weight, p_model_8_4_bias,
    p_model_9_0_weight, p_model_9_1_weight, p_model_9_1_bias,
    p_model_9_3_weight, p_model_9_4_weight, p_model_9_4_bias,
    p_model_10_0_weight, p_model_10_1_weight, p_model_10_1_bias,
    p_model_10_3_weight, p_model_10_4_weight, p_model_10_4_bias,
    p_model_11_0_weight, p_model_11_1_weight, p_model_11_1_bias,
    p_model_11_3_weight, p_model_11_4_weight, p_model_11_4_bias,
    p_model_12_0_weight, p_model_12_1_weight, p_model_12_1_bias,
    p_model_12_3_weight, p_model_12_4_weight, p_model_12_4_bias,
    p_model_13_0_weight, p_model_13_1_weight, p_model_13_1_bias,
    p_model_13_3_weight, p_model_13_4_weight, p_model_13_4_bias,
    p_fc_weight, p_fc_bias,
    b_model_0_1_running_mean, b_model_0_1_running_var,
    b_model_1_1_running_mean, b_model_1_1_running_var,
    b_model_1_4_running_mean, b_model_1_4_running_var,
    b_model_2_1_running_mean, b_model_2_1_running_var,
    b_model_2_4_running_mean, b_model_2_4_running_var,
    b_model_3_1_running_mean, b_model_3_1_running_var,
    b_model_3_4_running_mean, b_model_3_4_running_var,
    b_model_4_1_running_mean, b_model_4_1_running_var,
    b_model_4_4_running_mean, b_model_4_4_running_var,
    b_model_5_1_running_mean, b_model_5_1_running_var,
    b_model_5_4_running_mean, b_model_5_4_running_var,
    b_model_6_1_running_mean, b_model_6_1_running_var,
    b_model_6_4_running_mean, b_model_6_4_running_var,
    b_model_7_1_running_mean, b_model_7_1_running_var,
    b_model_7_4_running_mean, b_model_7_4_running_var,
    b_model_8_1_running_mean, b_model_8_1_running_var,
    b_model_8_4_running_mean, b_model_8_4_running_var,
    b_model_9_1_running_mean, b_model_9_1_running_var,
    b_model_9_4_running_mean, b_model_9_4_running_var,
    b_model_10_1_running_mean, b_model_10_1_running_var,
    b_model_10_4_running_mean, b_model_10_4_running_var,
    b_model_11_1_running_mean, b_model_11_1_running_var,
    b_model_11_4_running_mean, b_model_11_4_running_var,
    b_model_12_1_running_mean, b_model_12_1_running_var,
    b_model_12_4_running_mean, b_model_12_4_running_var,
    b_model_13_1_running_mean, b_model_13_1_running_var,
    b_model_13_4_running_mean, b_model_13_4_running_var,
    x,
):
    # Layer 0: conv_bn (3->32, stride=2)
    out = conv2d_standard(x, p_model_0_0_weight, stride=2, padding=1)
    out = fused_bn_relu(out, b_model_0_1_running_mean, b_model_0_1_running_var,

    # Layer 1: conv_dw (32->64, stride=1)
    out = conv2d_depthwise(out, p_model_1_0_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_model_1_1_running_mean, b_model_1_1_running_var,
    out = conv2d_pointwise(out, p_model_1_3_weight)
    out = fused_bn_relu(out, b_model_1_4_running_mean, b_model_1_4_running_var,

    # Layer 2: conv_dw (64->128, stride=2)
    out = conv2d_depthwise(out, p_model_2_0_weight, stride=2, padding=1)
    out = fused_bn_relu(out, b_model_2_1_running_mean, b_model_2_1_running_var,
    out = conv2d_pointwise(out, p_model_2_3_weight)
    out = fused_bn_relu(out, b_model_2_4_running_mean, b_model_2_4_running_var,

    # Layer 3: conv_dw (128->128, stride=1)
    out = conv2d_depthwise(out, p_model_3_0_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_model_3_1_running_mean, b_model_3_1_running_var,
    out = conv2d_pointwise(out, p_model_3_3_weight)
    out = fused_bn_relu(out, b_model_3_4_running_mean, b_model_3_4_running_var,

    # Layer 4: conv_dw (128->256, stride=2)
    out = conv2d_depthwise(out, p_model_4_0_weight, stride=2, padding=1)
    out = fused_bn_relu(out, b_model_4_1_running_mean, b_model_4_1_running_var,
    out = conv2d_pointwise(out, p_model_4_3_weight)
    out = fused_bn_relu(out, b_model_4_4_running_mean, b_model_4_4_running_var,

    # Layer 5: conv_dw (256->256, stride=1)
    out = conv2d_depthwise(out, p_model_5_0_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_model_5_1_running_mean, b_model_5_1_running_var,
    out = conv2d_pointwise(out, p_model_5_3_weight)
    out = fused_bn_relu(out, b_model_5_4_running_mean, b_model_5_4_running_var,

    # Layer 6: conv_dw (256->512, stride=2)
    out = conv2d_depthwise(out, p_model_6_0_weight, stride=2, padding=1)
    out = fused_bn_relu(out, b_model_6_1_running_mean, b_model_6_1_running_var,
    out = conv2d_pointwise(out, p_model_6_3_weight)
    out = fused_bn_relu(out, b_model_6_4_running_mean, b_model_6_4_running_var,

    # Layer 7: conv_dw (512->512, stride=1)
    out = conv2d_depthwise(out, p_model_7_0_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_model_7_1_running_mean, b_model_7_1_running_var,
    out = conv2d_pointwise(out, p_model_7_3_weight)
    out = fused_bn_relu(out, b_model_7_4_running_mean, b_model_7_4_running_var,

    # Layer 8: conv_dw (512->512, stride=1)
    out = conv2d_depthwise(out, p_model_8_0_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_model_8_1_running_mean, b_model_8_1_running_var,
    out = conv2d_pointwise(out, p_model_8_3_weight)
    out = fused_bn_relu(out, b_model_8_4_running_mean, b_model_8_4_running_var,

    # Layer 9: conv_dw (512->512, stride=1)
    out = conv2d_depthwise(out, p_model_9_0_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_model_9_1_running_mean, b_model_9_1_running_var,
    out = conv2d_pointwise(out, p_model_9_3_weight)
    out = fused_bn_relu(out, b_model_9_4_running_mean, b_model_9_4_running_var,

    # Layer 10: conv_dw (512->512, stride=1)
    out = conv2d_depthwise(out, p_model_10_0_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_model_10_1_running_mean, b_model_10_1_running_var,
    out = conv2d_pointwise(out, p_model_10_3_weight)
    out = fused_bn_relu(out, b_model_10_4_running_mean, b_model_10_4_running_var,

    # Layer 11: conv_dw (512->512, stride=1)
    out = conv2d_depthwise(out, p_model_11_0_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_model_11_1_running_mean, b_model_11_1_running_var,
    out = conv2d_pointwise(out, p_model_11_3_weight)
    out = fused_bn_relu(out, b_model_11_4_running_mean, b_model_11_4_running_var,

    # Layer 12: conv_dw (512->1024, stride=2)
    out = conv2d_depthwise(out, p_model_12_0_weight, stride=2, padding=1)
    out = fused_bn_relu(out, b_model_12_1_running_mean, b_model_12_1_running_var,
    out = conv2d_pointwise(out, p_model_12_3_weight)
    out = fused_bn_relu(out, b_model_12_4_running_mean, b_model_12_4_running_var,

    # Layer 13: conv_dw (1024->1024, stride=1)
    out = conv2d_depthwise(out, p_model_13_0_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_model_13_1_running_mean, b_model_13_1_running_var,
    out = conv2d_pointwise(out, p_model_13_3_weight)
    out = fused_bn_relu(out, b_model_13_4_running_mean, b_model_13_4_running_var,

    # AvgPool2d(7) -> (N, 1024, 1, 1)
    N, C, H, W = out.shape
    # Use Triton avg pool
    pool_out = torch.empty(N, C, device=out.device, dtype=out.dtype)
    numel_pool = N * C
    grid_pool = lambda meta: (triton.cdiv(numel_pool, meta['BLOCK_SIZE']),)
    avg_pool_flatten_kernel[grid_pool](
        out, pool_out,
        numel_pool, N, C, H, W,
    )

    # Linear: (N, 1024) -> (N, 1000)
    result = fused_linear(pool_out, p_fc_weight, p_fc_bias)
    return result


def get_inputs():
    device = 'cuda'

    def make_param(*shape):
        return torch.randn(*shape, device=device)

    def make_bn_params(c):
        weight = torch.ones(c, device=device)
        bias = torch.zeros(c, device=device)
        running_mean = torch.zeros(c, device=device)
        running_var = torch.ones(c, device=device)
        return weight, bias, running_mean, running_var

    p_model_0_0_weight = make_param(32, 3, 3, 3)
    w01, b01, rm01, rv01 = make_bn_params(32)

    p_model_1_0_weight = make_param(32, 1, 3, 3)
    w11, b11, rm11, rv11 = make_bn_params(32)
    p_model_1_3_weight = make_param(64, 32, 1, 1)
    w14, b14, rm14, rv14 = make_bn_params(64)

    p_model_2_0_weight = make_param(64, 1, 3, 3)
    w21, b21, rm21, rv21 = make_bn_params(64)
    p_model_2_3_weight = make_param(128, 64, 1, 1)
    w24, b24, rm24, rv24 = make_bn_params(128)

    p_model_3_0_weight = make_param(128, 1, 3, 3)
    w31, b31, rm31, rv31 = make_bn_params(128)
    p_model_3_3_weight = make_param(128, 128, 1, 1)
    w34, b34, rm34, rv34 = make_bn_params(128)

    p_model_4_0_weight = make_param(128, 1, 3, 3)
    w41, b41, rm41, rv41 = make_bn_params(128)
    p_model_4_3_weight = make_param(256, 128, 1, 1)
    w44, b44, rm44, rv44 = make_bn_params(256)

    p_model_5_0_weight = make_param(256, 1, 3, 3)
    w51, b51, rm51, rv51 = make_bn_params(256)
    p_model_5_3_weight = make_param(256, 256, 1, 1)
    w54, b54, rm54, rv54 = make_bn_params(256)

    p_model_6_0_weight = make_param(256, 1, 3, 3)
    w61, b61, rm61, rv61 = make_bn_params(256)
    p_model_6_3_weight = make_param(512, 256, 1, 1)
    w64, b64, rm64, rv64 = make_bn_params(512)

    p_model_7_0_weight = make_param(512, 1, 3, 3)
    w71, b71, rm71, rv71 = make_bn_params(512)
    p_model_7_3_weight = make_param(512, 512, 1, 1)
    w74, b74, rm74, rv74 = make_bn_params(512)

    p_model_8_0_weight = make_param(512, 1, 3, 3)
    w81, b81, rm81, rv81 = make_bn_params(512)
    p_model_8_3_weight = make_param(512, 512, 1, 1)
    w84, b84, rm84, rv84 = make_bn_params(512)

    p_model_9_0_weight = make_param(512, 1, 3, 3)
    w91, b91, rm91, rv91 = make_bn_params(512)
    p_model_9_3_weight = make_param(512, 512, 1, 1)
    w94, b94, rm94, rv94 = make_bn_params(512)

    p_model_10_0_weight = make_param(512, 1, 3, 3)
    w101, b101, rm101, rv101 = make_bn_params(512)
    p_model_10_3_weight = make_param(512, 512, 1, 1)
    w104, b104, rm104, rv104 = make_bn_params(512)

    p_model_11_0_weight = make_param(512, 1, 3, 3)
    w111, b111, rm111, rv111 = make_bn_params(512)
    p_model_11_3_weight = make_param