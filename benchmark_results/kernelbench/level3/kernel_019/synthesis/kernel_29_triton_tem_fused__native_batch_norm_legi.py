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

    # Determine channel index for each element
    # Layout: (N, C, H, W) -> channel = (offset / spatial_size) % num_channels
    channel_idx = (offsets // spatial_size) % num_channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)

    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    # ReLU
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


def bn_relu(x, running_mean, running_var, weight, bias, eps=1e-5):
    """Fused BatchNorm + ReLU"""
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
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
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

    offs_m_out = offs_m
    offs_n_out = offs_n
    out_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    b = tl.load(b_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + b[None, :]

    out_ptrs = out_ptr + offs_m_out[:, None] * stride_om + offs_n_out[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_linear(x, weight, bias):
    """Linear layer using Triton"""
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


def conv_bn_relu(x, conv_weight, bn_mean, bn_var, bn_weight, bn_bias,
                 stride=1, padding=0, groups=1):
    """Conv2d + BN + ReLU with fused BN+ReLU in Triton"""
    # Use PyTorch/cuDNN for convolution
    conv_out = F.conv2d(x, conv_weight, bias=None, stride=stride, padding=padding, groups=groups)
    # Fused BN + ReLU in Triton
    return bn_relu(conv_out, bn_mean, bn_var, bn_weight, bn_bias)


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
    # Layer 0: conv_bn (3x3, stride=2, pad=1)
    x = conv_bn_relu(x, p_model_0_0_weight,
                     b_model_0_1_running_mean, b_model_0_1_running_var,
                     p_model_0_1_weight, p_model_0_1_bias,
                     stride=2, padding=1)

    # Layer 1: conv_dw (depthwise 3x3, stride=1) + pointwise 1x1
    x = conv_bn_relu(x, p_model_1_0_weight,
                     b_model_1_1_running_mean, b_model_1_1_running_var,
                     p_model_1_1_weight, p_model_1_1_bias,
                     stride=1, padding=1, groups=32)
    x = conv_bn_relu(x, p_model_1_3_weight,
                     b_model_1_4_running_mean, b_model_1_4_running_var,
                     p_model_1_4_weight, p_model_1_4_bias,
                     stride=1, padding=0)

    # Layer 2: conv_dw (stride=2) + pointwise
    x = conv_bn_relu(x, p_model_2_0_weight,
                     b_model_2_1_running_mean, b_model_2_1_running_var,
                     p_model_2_1_weight, p_model_2_1_bias,
                     stride=2, padding=1, groups=64)
    x = conv_bn_relu(x, p_model_2_3_weight,
                     b_model_2_4_running_mean, b_model_2_4_running_var,
                     p_model_2_4_weight, p_model_2_4_bias,
                     stride=1, padding=0)

    # Layer 3: conv_dw (stride=1) + pointwise
    x = conv_bn_relu(x, p_model_3_0_weight,
                     b_model_3_1_running_mean, b_model_3_1_running_var,
                     p_model_3_1_weight, p_model_3_1_bias,
                     stride=1, padding=1, groups=128)
    x = conv_bn_relu(x, p_model_3_3_weight,
                     b_model_3_4_running_mean, b_model_3_4_running_var,
                     p_model_3_4_weight, p_model_3_4_bias,
                     stride=1, padding=0)

    # Layer 4: conv_dw (stride=2) + pointwise
    x = conv_bn_relu(x, p_model_4_0_weight,
                     b_model_4_1_running_mean, b_model_4_1_running_var,
                     p_model_4_1_weight, p_model_4_1_bias,
                     stride=2, padding=1, groups=128)
    x = conv_bn_relu(x, p_model_4_3_weight,
                     b_model_4_4_running_mean, b_model_4_4_running_var,
                     p_model_4_4_weight, p_model_4_4_bias,
                     stride=1, padding=0)

    # Layer 5: conv_dw (stride=1) + pointwise
    x = conv_bn_relu(x, p_model_5_0_weight,
                     b_model_5_1_running_mean, b_model_5_1_running_var,
                     p_model_5_1_weight, p_model_5_1_bias,
                     stride=1, padding=1, groups=256)
    x = conv_bn_relu(x, p_model_5_3_weight,
                     b_model_5_4_running_mean, b_model_5_4_running_var,
                     p_model_5_4_weight, p_model_5_4_bias,
                     stride=1, padding=0)

    # Layer 6: conv_dw (stride=2) + pointwise
    x = conv_bn_relu(x, p_model_6_0_weight,
                     b_model_6_1_running_mean, b_model_6_1_running_var,
                     p_model_6_1_weight, p_model_6_1_bias,
                     stride=2, padding=1, groups=256)
    x = conv_bn_relu(x, p_model_6_3_weight,
                     b_model_6_4_running_mean, b_model_6_4_running_var,
                     p_model_6_4_weight, p_model_6_4_bias,
                     stride=1, padding=0)

    # Layer 7: conv_dw (stride=1) + pointwise
    x = conv_bn_relu(x, p_model_7_0_weight,
                     b_model_7_1_running_mean, b_model_7_1_running_var,
                     p_model_7_1_weight, p_model_7_1_bias,
                     stride=1, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_7_3_weight,
                     b_model_7_4_running_mean, b_model_7_4_running_var,
                     p_model_7_4_weight, p_model_7_4_bias,
                     stride=1, padding=0)

    # Layer 8: conv_dw (stride=1) + pointwise
    x = conv_bn_relu(x, p_model_8_0_weight,
                     b_model_8_1_running_mean, b_model_8_1_running_var,
                     p_model_8_1_weight, p_model_8_1_bias,
                     stride=1, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_8_3_weight,
                     b_model_8_4_running_mean, b_model_8_4_running_var,
                     p_model_8_4_weight, p_model_8_4_bias,
                     stride=1, padding=0)

    # Layer 9: conv_dw (stride=1) + pointwise
    x = conv_bn_relu(x, p_model_9_0_weight,
                     b_model_9_1_running_mean, b_model_9_1_running_var,
                     p_model_9_1_weight, p_model_9_1_bias,
                     stride=1, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_9_3_weight,
                     b_model_9_4_running_mean, b_model_9_4_running_var,
                     p_model_9_4_weight, p_model_9_4_bias,
                     stride=1, padding=0)

    # Layer 10: conv_dw (stride=1) + pointwise
    x = conv_bn_relu(x, p_model_10_0_weight,
                     b_model_10_1_running_mean, b_model_10_1_running_var,
                     p_model_10_1_weight, p_model_10_1_bias,
                     stride=1, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_10_3_weight,
                     b_model_10_4_running_mean, b_model_10_4_running_var,
                     p_model_10_4_weight, p_model_10_4_bias,
                     stride=1, padding=0)

    # Layer 11: conv_dw (stride=1) + pointwise
    x = conv_bn_relu(x, p_model_11_0_weight,
                     b_model_11_1_running_mean, b_model_11_1_running_var,
                     p_model_11_1_weight, p_model_11_1_bias,
                     stride=1, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_11_3_weight,
                     b_model_11_4_running_mean, b_model_11_4_running_var,
                     p_model_11_4_weight, p_model_11_4_bias,
                     stride=1, padding=0)

    # Layer 12: conv_dw (stride=2) + pointwise
    x = conv_bn_relu(x, p_model_12_0_weight,
                     b_model_12_1_running_mean, b_model_12_1_running_var,
                     p_model_12_1_weight, p_model_12_1_bias,
                     stride=2, padding=1, groups=512)
    x = conv_bn_relu(x, p_model_12_3_weight,
                     b_model_12_4_running_mean, b_model_12_4_running_var,
                     p_model_12_4_weight, p_model_12_4_bias,
                     stride=1, padding=0)

    # Layer 13: conv_dw (stride=1) + pointwise
    x = conv_bn_relu(x, p_model_13_0_weight,
                     b_model_13_1_running_mean, b_model_13_1_running_var,
                     p_model_13_1_weight, p_model_13_1_bias,
                     stride=1, padding=1, groups=1024)
    x = conv_bn_relu(x, p_model_13_3_weight,
                     b_model_13_4_running_mean, b_model_13_4_running_var,
                     p_model_13_4_weight, p_model_13_4_bias,
                     stride=1, padding=0)

    # AvgPool2d(7)
    x = F.avg_pool2d(x, 7)

    # Flatten
    x = x.view(x.size(0), -1)

    # Linear layer using Triton
    x = triton_linear(x, p_fc_weight, p_fc_bias)

    return x


def get_inputs():
    # Create model to get proper weights
    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
            super(Model, self).__init__()

            def conv_bn(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
                )

            def conv_dw(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )

            self.model = nn.Sequential(
                conv_bn(input_channels, int(32 * alpha), 2),
                conv_dw(int(32 * alpha), int(64 * alpha), 1),
                conv_dw(int(64 * alpha), int(128 * alpha), 2),
                conv_dw(int(128 * alpha), int(128 * alpha), 1),
                conv_dw(int(128 * alpha), int(256 * alpha), 2),
                conv_dw(int(256 * alpha), int(256 * alpha), 1),
                conv_dw(int(256 * alpha), int(512 * alpha), 2),
                conv_dw(int(512 * alpha), int(512 * alpha), 1),
                conv_dw(int(512 * alpha), int(512 * alpha), 1),
                conv_dw(int(512 * alpha), int(512 * alpha), 1),
                conv_dw(int(512 * alpha), int(512 * alpha), 1),
                conv_dw(int(512 * alpha), int(512 * alpha), 1),
                conv_dw(int(512 * alpha), int(1024 * alpha), 2),
                conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
                nn.AvgPool2d(7),
            )
            self.fc = nn.Linear(int(1024 * alpha), num_classes)

        def forward(self, x):
            x = self.model(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = Model().cuda().eval()

    # Extract all parameters
    m = model.model
    fc = model.fc

    inputs = [
        # Layer 0
        m[0][0].weight.data,
        m[0][1].weight.data,
        m[0][1].bias.data,
        # Layer 1
        m[1][0].weight.data,
        m[1][1].weight.data,
        m[1][1].bias.data,
        m[1][3].weight.data,
        m[1][4].weight.data,
        m[1][4].bias.data,
        # Layer 2
        m[2][0].weight.data,
        m[2][1].weight.data,
        m[2][1].bias.data,
        m[2][3].weight.data,
        m[2][4].weight.data,
        m[2][4].bias.data,
        # Layer 3
        m[3][0].weight.data,
        m[3][1].weight.data,
        m[3][1].bias.data,
        m[3][3].weight.data,
        m[3][4].weight.data,
        m[3][4].bias.data,
        # Layer 4
        m[4][0].weight.data,
        m[4][1].weight.data,
        m[4][1].bias.data,
        m[4][3].weight.data,
        m[4][4].weight.data,
        m[4][4].bias.data,
        # Layer 5
        m[5][0].weight.data,
        m[5][1].weight.data,
        m[5][1].bias.data,
        m[5][3].weight.data,
        m[5][4].weight.data,
        m[5][4].bias.data,
        # Layer 6
        m[6][0].weight.data,
        m[6][1].weight.data,
        m[6][1].bias.data,
        m[6][3].weight.data,
        m[6][4].weight.data,
        m[6][4].bias.data,
        # Layer 7
        m[7][0].weight.data,
        m[7][1].weight.data,
        m[7][1].bias.data,
        m[7][3].weight.data,
        m[7][4].weight.data,
        m[7][4].bias.data,
        # Layer 8
        m[8][0].weight.data,
        m[8][1].weight.data,
        m[8][1].bias.data,
        m[8][3].weight.data,
        m[8][4].weight.data,
        m[8][4].bias.data,
        # Layer 9
        m[9][0].weight.data,
        m[9][1].weight.data,
        m[9][1].bias.data,
        m[9][3].weight.data,
        m[9][4].weight.data,
        m[9][4].bias.data,
        # Layer 10
        m[10][0].weight.data,
        m[10][1].weight.data,
        m[10][1].bias.data,
        m[10][3].weight.data,
        m[10][4].weight.data,
        m[10][4].bias.data,
        # Layer 11
        m[11][0].weight.data,
        m[11][1].weight.data,
        m[11][1].bias.data,
        m[11][3].weight.data,
        m[11][4].weight.data,
        m[11][4].bias.data,
        # Layer 12
        m[12][0].weight.data,
        m[12][1].weight.data,
        m[12][1].bias.data,
        m[12][3].weight.data,
        m[12][4].weight.data,
        m[12][4].bias.data,
        # Layer 13
        m[13][0].weight.data,
        m[13][1].weight.data,
        m[13][1].bias.data,
        m[13][3].weight.data,
        m[13][4].weight.data,
        m[13][4].bias.data,
        # FC
        fc.weight.data,
        fc.bias.data,