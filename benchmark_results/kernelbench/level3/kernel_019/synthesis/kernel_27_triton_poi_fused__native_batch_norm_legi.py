import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def bn_relu_fused_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    numel,
    spatial,
    channels,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Determine channel index: layout is (N, C, H, W) -> channel = (offset // spatial) % channels
    c_idx = (offsets // spatial) % channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_scaled = x_norm * w + b
    x_relu = tl.maximum(x_scaled, 0.0)

    tl.store(out_ptr + offsets, x_relu, mask=mask)


def apply_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = N * C * H * W
    spatial = H * W

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_fused_kernel[grid](
        x, out,
        running_mean, running_var,
        weight, bias,
        numel, spatial, C, eps,
    )
    return out


def conv_bn_relu(x, conv_weight, bn_weight, bn_bias, bn_mean, bn_var,
                 stride=1, padding=0, groups=1):
    # Use PyTorch conv (cuDNN) then fused BN+ReLU in Triton
    x = F.conv2d(x, conv_weight, bias=None, stride=stride, padding=padding, groups=groups)
    x = apply_bn_relu(x, bn_weight, bn_bias, bn_mean, bn_var)
    return x


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
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
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel  # numel = N * C

    n_idx = offsets // C
    c_idx = offsets % C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    total = H * W

    for h in range(H):
        for w in range(W):
            idx = n_idx * C * H * W + c_idx * H * W + h * W + w
            val = tl.load(x_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    acc = acc / total
    tl.store(out_ptr + offsets, acc, mask=mask)


def avg_pool_flatten(x):
    N, C, H, W = x.shape
    out = torch.empty(N, C, device=x.device, dtype=x.dtype)
    numel = N * C

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    avg_pool_flatten_kernel[grid](
        x, out,
        numel, N, C, H, W,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8),
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
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    BLOCK_K = 32
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile.T, acc)

    offs_m_out = offs_m
    offs_n_out = offs_n
    mask_out = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    bias = tl.load(b_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m_out[:, None] * N + offs_n_out[None, :]
    tl.store(out_ptrs, acc, mask=mask_out)


def apply_linear(x, weight, bias):
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
    # Layer 0: conv_bn (3->32, stride=2, 3x3, pad=1)
    x = conv_bn_relu(x, p_model_0_0_weight, p_model_0_1_weight, p_model_0_1_bias,
                     b_model_0_1_running_mean, b_model_0_1_running_var,
                     stride=2, padding=1)

    # Layer 1: conv_dw (32->32 dw, then 32->64 pw)
    x = conv_bn_relu(x, p_model_1_0_weight, p_model_1_1_weight, p_model_1_1_bias,
                     b_model_1_1_running_mean, b_model_1_1_running_var,
                     stride=1, padding=1, groups=32)
    x = conv_bn_relu(x, p_model_1_3_weight, p_model_1_4_weight, p_model_1_4_bias,
                     b_model_1_4_running_mean, b_model_1_4_running_var,
                     stride=1, padding=0)

    # Layer 2: conv_dw (64->64 dw stride=2, then 64->128 pw)
    x = conv_bn_relu(x, p_model_2_0_weight, p_model_2_1_weight, p_model_2_1_bias,
                     b_model_2_1_running_mean, b_model_2_1_running_var,
                     stride=2, padding=1, groups=64)
    x = conv_bn_relu(x, p_model_2_3_weight, p_model_2_4_weight, p_model_2_4_bias,
                     b_model_2_4_running_mean, b_model_2_4_running_var,
                     stride=1, padding=0)

    # Layer 3: conv_dw (128->128 dw, then 128->128 pw)
    x = conv_bn_relu(x, p_model_3_0_weight, p_model_3_1_weight, p_model_3_1_bias,
                     b_model_3_1_running_mean, b_model_3_1_running_var,
                     stride=1, padding=1, groups=128)
    x = conv_bn_relu(x, p_model_3_3_weight, p_model_3_4_weight, p_model_3_4_bias,
                     b_model_3_4_running_mean, b_model_3_4_running_var,
                     stride=1, padding=0)

    # Layer 4: conv_dw (128->128 dw stride=2, then 128->256 pw)
    x = conv_bn_relu(x, p_model_4_0_weight, p_model_4_1_weight, p_model_4_1_bias,
                     b_model_4_1_running_mean, b_model_4_1_running_var,
                     stride=2, padding=1, groups=128)
    x = conv_bn_relu(x, p_model_4_3_weight, p_model_4_4_weight, p_model_4_4_bias,
                     b_model_4_4_running_mean, b_model_4_4_running_var,
                     stride=1, padding=0)

    # Layer 5: conv_dw (256->256 dw, then 256->256 pw)
    x = conv_bn_relu(x, p_model_5_0_weight, p_model_5_1_weight, p_model_5_1_bias,
                     b_model_5_1_running_mean, b_model_5_1_running_var,
                     stride=1, padding=1, groups=256)
    x = conv_bn_relu(x, p_model_5_3_weight, p_model_5_4_weight, p_model_5_4_bias,
                     b_model_5_4_running_mean, b_model_5_4_running_var,
                     stride=1, padding=0)

    # Layer 6: conv_dw (256->256 dw stride=2, then 256->512 pw)
    x = conv_bn_relu(x, p_model_6_0_weight, p_model_6_1_weight, p_model_6_1_bias,
                     b_model_6_1_running_mean, b_model_6_1_running_var,
                     stride=2, padding=1, groups=256)
    x = conv_bn_relu(x, p_model_6_3_weight, p_model_6_4_weight, p_model_6_4_bias,
                     b_model_6_4_running_mean, b_model_6_4_running_var,
                     stride=1, padding=0)

    # Layer 7: conv_dw (512->512 dw, then 512->512 pw)
    x = conv_bn_relu(x, p_model_7_0_weight, p_model_7_1_weight, p_model_7_1_bias,
                     b_model_7_1_running_mean, b_model_7_1_running_var,
                     stride=1, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_7_3_weight, p_model_7_4_weight, p_model_7_4_bias,
                     b_model_7_4_running_mean, b_model_7_4_running_var,
                     stride=1, padding=0)

    # Layer 8: conv_dw (512->512 dw, then 512->512 pw)
    x = conv_bn_relu(x, p_model_8_0_weight, p_model_8_1_weight, p_model_8_1_bias,
                     b_model_8_1_running_mean, b_model_8_1_running_var,
                     stride=1, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_8_3_weight, p_model_8_4_weight, p_model_8_4_bias,
                     b_model_8_4_running_mean, b_model_8_4_running_var,
                     stride=1, padding=0)

    # Layer 9: conv_dw (512->512 dw, then 512->512 pw)
    x = conv_bn_relu(x, p_model_9_0_weight, p_model_9_1_weight, p_model_9_1_bias,
                     b_model_9_1_running_mean, b_model_9_1_running_var,
                     stride=1, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_9_3_weight, p_model_9_4_weight, p_model_9_4_bias,
                     b_model_9_4_running_mean, b_model_9_4_running_var,
                     stride=1, padding=0)

    # Layer 10: conv_dw (512->512 dw, then 512->512 pw)
    x = conv_bn_relu(x, p_model_10_0_weight, p_model_10_1_weight, p_model_10_1_bias,
                     b_model_10_1_running_mean, b_model_10_1_running_var,
                     stride=1, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_10_3_weight, p_model_10_4_weight, p_model_10_4_bias,
                     b_model_10_4_running_mean, b_model_10_4_running_var,
                     stride=1, padding=0)

    # Layer 11: conv_dw (512->512 dw, then 512->512 pw)
    x = conv_bn_relu(x, p_model_11_0_weight, p_model_11_1_weight, p_model_11_1_bias,
                     b_model_11_1_running_mean, b_model_11_1_running_var,
                     stride=1, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_11_3_weight, p_model_11_4_weight, p_model_11_4_bias,
                     b_model_11_4_running_mean, b_model_11_4_running_var,
                     stride=1, padding=0)

    # Layer 12: conv_dw (512->512 dw stride=2, then 512->1024 pw)
    x = conv_bn_relu(x, p_model_12_0_weight, p_model_12_1_weight, p_model_12_1_bias,
                     b_model_12_1_running_mean, b_model_12_1_running_var,
                     stride=2, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_12_3_weight, p_model_12_4_weight, p_model_12_4_bias,
                     b_model_12_4_running_mean, b_model_12_4_running_var,
                     stride=1, padding=0)

    # Layer 13: conv_dw (1024->1024 dw, then 1024->1024 pw)
    x = conv_bn_relu(x, p_model_13_0_weight, p_model_13_1_weight, p_model_13_1_bias,
                     b_model_13_1_running_mean, b_model_13_1_running_var,
                     stride=1, padding=1, groups=1024)
    x = conv_bn_relu(x, p_model_13_3_weight, p_model_13_4_weight, p_model_13_4_bias,
                     b_model_13_4_running_mean, b_model_13_4_running_var,
                     stride=1, padding=0)

    # AvgPool + flatten using Triton
    x = avg_pool_flatten(x)  # (N, C)

    # Linear using Triton
    x = apply_linear(x, p_fc_weight, p_fc_bias)

    return x


def get_inputs():
    device = 'cuda'

    def make_param(*shape):
        return torch.randn(*shape, device=device)

    def make_bn_params(c):
        weight = torch.ones(c, device=device)
        bias = torch.zeros(c, device=device)
        mean = torch.zeros(c, device=device)
        var = torch.ones(c, device=device)
        return weight, bias, mean, var

    p_model_0_0_weight = make_param(32, 3, 3, 3)
    p_model_0_1_weight, p_model_0_1_bias, b_model_0_1_running_mean, b_model_0_1_running_var = make_bn_params(32)

    p_model_1_0_weight = make_param(32, 1, 3, 3)
    p_model_1_1_weight, p_model_1_1_bias, b_model_1_1_running_mean, b_model_1_1_running_var = make_bn_params(32)
    p_model_1_3_weight = make_param(64, 32, 1, 1)
    p_model_1_4_weight, p_model_1_4_bias, b_model_1_4_running_mean, b_model_1_4_running_var = make_bn_params(64)

    p_model_2_0_weight = make_param(64, 1, 3, 3)
    p_model_2_1_weight, p_model_2_1_bias, b_model_2_1_running_mean, b_model_2_1_running_var = make_bn_params(64)
    p_model_2_3_weight = make_param(128, 64, 1, 1)
    p_model_2_4_weight, p_model_2_4_bias, b_model_2_4_running_mean, b_model_2_4_running_var = make_bn_params(128)

    p_model_3_0_weight = make_param(128, 1, 3, 3)
    p_model_3_1_weight, p_model_3_1_bias, b_model_3_1_running_mean, b_model_3_1_running_var = make_bn_params(128)
    p_model_3_3_weight = make_param(128, 128, 1, 1)
    p_model_3_4_weight, p_model_3_4_bias, b_model_3_4_running_mean, b_model_3_4_running_var = make_bn_params(128)

    p_model_4_0_weight = make_param(128, 1, 3, 3)
    p_model_4_1_weight, p_model_4_1_bias, b_model_4_1_running_mean, b_model_4_1_running_var = make_bn_params(128)
    p_model_4_3_weight = make_param(256, 128, 1, 1)
    p_model_4_4_weight, p_model_4_4_bias, b_model_4_4_running_mean, b_model_4_4_running_var = make_bn_params(256)

    p_model_5_0_weight = make_param(256, 1, 3, 3)
    p_model_5_1_weight, p_model_5_1_bias, b_model_5_1_running_mean, b_model_5_1_running_var = make_bn_params(256)
    p_model_5_3_weight = make_param(256, 256, 1, 1)
    p_model_5_4_weight, p_model_5_4_bias, b_model_5_4_running_mean, b_model_5_4_running_var = make_bn_params(256)

    p_model_6_0_weight = make_param(256, 1, 3, 3)
    p_model_6_1_weight, p_model_6_1_bias, b_model_6_1_running_mean, b_model_6_1_running_var = make_bn_params(256)
    p_model_6_3_weight = make_param(512, 256, 1, 1)
    p_model_6_4_weight, p_model_6_4_bias, b_model_6_4_running_mean, b_model_6_4_running_var = make_bn_params(512)

    p_model_7_0_weight = make_param(512, 1, 3, 3)
    p_model_7_1_weight, p_model_7_1_bias, b_model_7_1_running_mean, b_model_7_1_running_var = make_bn_params(512)
    p_model_7_3_weight = make_param(512, 512, 1, 1)
    p_model_7_4_weight, p_model_7_4_bias, b_model_7_4_running_mean, b_model_7_4_running_var = make_bn_params(512)

    p_model_8_0_weight = make_param(512, 1, 3, 3)
    p_model_8_1_weight, p_model_8_1_bias, b_model_8_1_running_mean, b_model_8_1_running_var = make_bn_params(512)
    p_model_8_3_weight = make_param(512, 512, 1, 1)
    p_model_8_4_weight, p_model_8_4_bias, b_model_8_4_running_mean, b_model_8_4_running_var = make_bn_params(512)

    p_model_9_0_weight = make_param(512, 1, 3, 3)