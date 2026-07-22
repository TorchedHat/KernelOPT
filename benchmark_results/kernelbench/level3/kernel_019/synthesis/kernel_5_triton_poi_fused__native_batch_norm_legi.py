import torch
import triton
import triton.language as tl
import torch.nn.functional as F

import torch
import triton
import triton.language as tl

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
def bn_relu_fused_kernel(
    x_ptr,
    out_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    numel,
    num_channels,
    HW: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index: layout is (N, C, H, W) contiguous
    # channel = (offset // HW) % num_channels
    c_idx = (offsets // HW) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def apply_bn_relu(x, running_mean, running_var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_fused_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, HW, eps,
    )
    return out


def conv2d_bn_relu(x, conv_weight, bn_mean, bn_var, bn_weight, bn_bias,
                   stride=1, padding=0, groups=1):
    # Use PyTorch's optimized conv2d (cuDNN)
    x = torch.ops.aten.convolution(x, conv_weight, None,
                                    [stride, stride], [padding, padding],
    x = apply_bn_relu(x, bn_mean, bn_var, bn_weight, bn_bias)
    return x


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
def avg_pool_flatten_kernel(
    x_ptr,
    out_ptr,
    N,
    C,
    H,
    W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel  # numel = N * C

    n = offsets // C
    c = offsets % C

    # Sum over H*W
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    HW = H * W
    base = n * C * HW + c * HW
    for i in range(H * W):
        val = tl.load(x_ptr + base + i, mask=mask, other=0.0).to(tl.float32)
        acc = acc + val
    acc = acc / (H * W).to(tl.float32)

    tl.store(out_ptr + offsets, acc.to(tl.float32), mask=mask)


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
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        x_tile = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :],
                         mask=x_mask, other=0.0).to(tl.float32)
        w_tile = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :],
                         mask=w_mask, other=0.0).to(tl.float32)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(tl.float32), mask=out_mask)


def apply_linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, weight, bias, out, M, N, K)
    return out


def apply_avg_pool_flatten(x):
    N, C, H, W = x.shape
    numel = N * C
    out = torch.empty(N, C, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    avg_pool_flatten_kernel[grid](
        x, out.view(-1),
        N, C, H, W, numel,
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
    # Layer 0: conv_bn (3->32, stride=2, pad=1, 3x3)
    x = torch.ops.aten.convolution(x, p_model_0_0_weight, None,
    x = apply_bn_relu(x, b_model_0_1_running_mean, b_model_0_1_running_var,

    # Layer 1: conv_dw (32->32 dw, then 32->64 pw)
    x = torch.ops.aten.convolution(x, p_model_1_0_weight, None,
    x = apply_bn_relu(x, b_model_1_1_running_mean, b_model_1_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_1_3_weight, None,
    x = apply_bn_relu(x, b_model_1_4_running_mean, b_model_1_4_running_var,

    # Layer 2: conv_dw (64->64 dw stride=2, then 64->128 pw)
    x = torch.ops.aten.convolution(x, p_model_2_0_weight, None,
    x = apply_bn_relu(x, b_model_2_1_running_mean, b_model_2_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_2_3_weight, None,
    x = apply_bn_relu(x, b_model_2_4_running_mean, b_model_2_4_running_var,

    # Layer 3: conv_dw (128->128 dw, then 128->128 pw)
    x = torch.ops.aten.convolution(x, p_model_3_0_weight, None,
    x = apply_bn_relu(x, b_model_3_1_running_mean, b_model_3_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_3_3_weight, None,
    x = apply_bn_relu(x, b_model_3_4_running_mean, b_model_3_4_running_var,

    # Layer 4: conv_dw (128->128 dw stride=2, then 128->256 pw)
    x = torch.ops.aten.convolution(x, p_model_4_0_weight, None,
    x = apply_bn_relu(x, b_model_4_1_running_mean, b_model_4_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_4_3_weight, None,
    x = apply_bn_relu(x, b_model_4_4_running_mean, b_model_4_4_running_var,

    # Layer 5: conv_dw (256->256 dw, then 256->256 pw)
    x = torch.ops.aten.convolution(x, p_model_5_0_weight, None,
    x = apply_bn_relu(x, b_model_5_1_running_mean, b_model_5_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_5_3_weight, None,
    x = apply_bn_relu(x, b_model_5_4_running_mean, b_model_5_4_running_var,

    # Layer 6: conv_dw (256->256 dw stride=2, then 256->512 pw)
    x = torch.ops.aten.convolution(x, p_model_6_0_weight, None,
    x = apply_bn_relu(x, b_model_6_1_running_mean, b_model_6_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_6_3_weight, None,
    x = apply_bn_relu(x, b_model_6_4_running_mean, b_model_6_4_running_var,

    # Layer 7: conv_dw (512->512 dw, then 512->512 pw)
    x = torch.ops.aten.convolution(x, p_model_7_0_weight, None,
    x = apply_bn_relu(x, b_model_7_1_running_mean, b_model_7_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_7_3_weight, None,
    x = apply_bn_relu(x, b_model_7_4_running_mean, b_model_7_4_running_var,

    # Layer 8: conv_dw (512->512 dw, then 512->512 pw)
    x = torch.ops.aten.convolution(x, p_model_8_0_weight, None,
    x = apply_bn_relu(x, b_model_8_1_running_mean, b_model_8_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_8_3_weight, None,
    x = apply_bn_relu(x, b_model_8_4_running_mean, b_model_8_4_running_var,

    # Layer 9: conv_dw (512->512 dw, then 512->512 pw)
    x = torch.ops.aten.convolution(x, p_model_9_0_weight, None,
    x = apply_bn_relu(x, b_model_9_1_running_mean, b_model_9_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_9_3_weight, None,
    x = apply_bn_relu(x, b_model_9_4_running_mean, b_model_9_4_running_var,

    # Layer 10: conv_dw (512->512 dw, then 512->512 pw)
    x = torch.ops.aten.convolution(x, p_model_10_0_weight, None,
    x = apply_bn_relu(x, b_model_10_1_running_mean, b_model_10_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_10_3_weight, None,
    x = apply_bn_relu(x, b_model_10_4_running_mean, b_model_10_4_running_var,

    # Layer 11: conv_dw (512->512 dw, then 512->512 pw)
    x = torch.ops.aten.convolution(x, p_model_11_0_weight, None,
    x = apply_bn_relu(x, b_model_11_1_running_mean, b_model_11_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_11_3_weight, None,
    x = apply_bn_relu(x, b_model_11_4_running_mean, b_model_11_4_running_var,

    # Layer 12: conv_dw (512->512 dw stride=2, then 512->1024 pw)
    x = torch.ops.aten.convolution(x, p_model_12_0_weight, None,
    x = apply_bn_relu(x, b_model_12_1_running_mean, b_model_12_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_12_3_weight, None,
    x = apply_bn_relu(x, b_model_12_4_running_mean, b_model_12_4_running_var,

    # Layer 13: conv_dw (1024->1024 dw, then 1024->1024 pw)
    x = torch.ops.aten.convolution(x, p_model_13_0_weight, None,
    x = apply_bn_relu(x, b_model_13_1_running_mean, b_model_13_1_running_var,
    x = torch.ops.aten.convolution(x, p_model_13_3_weight, None,
    x = apply_bn_relu(x, b_model_13_4_running_mean, b_model_13_4_running_var,

    # AvgPool2d(7) + flatten
    x = apply_avg_pool_flatten(x)

    # Linear
    x = apply_linear(x, p_fc_weight, p_fc_bias)

    return x


def get_inputs():
    import torch
    device = 'cuda'

    def make_tensor(*shape):
        return torch.randn(*shape, device=device)

    def make_bn_params(c):
        return (
            make_tensor(c),  # weight
            make_tensor(c),  # bias