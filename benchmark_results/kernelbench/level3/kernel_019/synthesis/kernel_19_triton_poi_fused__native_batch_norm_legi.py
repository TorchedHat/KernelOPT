import torch
import triton
import triton.language as tl

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

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)

    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_scaled = x_norm * weight + bias

    # ReLU
    result = tl.maximum(x_scaled, 0.0)

    tl.store(out_ptr + offsets, result, mask=mask)


def apply_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial_size = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var,
        weight, bias,
        numel, C, spatial_size, eps,
    )
    return out


def conv2d_bn_relu(x, conv_weight, bn_weight, bn_bias, bn_mean, bn_var,
                   stride=1, padding=0, groups=1, eps=1e-5):
    # Use PyTorch's optimized conv (cuDNN)
    import torch.nn.functional as F
    out = F.conv2d(x, conv_weight, bias=None, stride=stride, padding=padding, groups=groups)
    # Fuse BN + ReLU in Triton
    out = apply_bn_relu(out, bn_weight, bn_bias, bn_mean, bn_var, eps)
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
    import torch.nn.functional as F

    # Layer 0: conv_bn (3->32, stride=2, pad=1)
    x = conv2d_bn_relu(x, p_model_0_0_weight, p_model_0_1_weight, p_model_0_1_bias,
                       b_model_0_1_running_mean, b_model_0_1_running_var,
                       stride=2, padding=1, groups=1)

    # Layer 1: conv_dw (32->32 dw, then 32->64 pw)
    x = conv2d_bn_relu(x, p_model_1_0_weight, p_model_1_1_weight, p_model_1_1_bias,
                       b_model_1_1_running_mean, b_model_1_1_running_var,
                       stride=1, padding=1, groups=32)
    x = conv2d_bn_relu(x, p_model_1_3_weight, p_model_1_4_weight, p_model_1_4_bias,
                       b_model_1_4_running_mean, b_model_1_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 2: conv_dw (64->64 dw stride=2, then 64->128 pw)
    x = conv2d_bn_relu(x, p_model_2_0_weight, p_model_2_1_weight, p_model_2_1_bias,
                       b_model_2_1_running_mean, b_model_2_1_running_var,
                       stride=2, padding=1, groups=64)
    x = conv2d_bn_relu(x, p_model_2_3_weight, p_model_2_4_weight, p_model_2_4_bias,
                       b_model_2_4_running_mean, b_model_2_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 3: conv_dw (128->128 dw, then 128->128 pw)
    x = conv2d_bn_relu(x, p_model_3_0_weight, p_model_3_1_weight, p_model_3_1_bias,
                       b_model_3_1_running_mean, b_model_3_1_running_var,
                       stride=1, padding=1, groups=128)
    x = conv2d_bn_relu(x, p_model_3_3_weight, p_model_3_4_weight, p_model_3_4_bias,
                       b_model_3_4_running_mean, b_model_3_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 4: conv_dw (128->128 dw stride=2, then 128->256 pw)
    x = conv2d_bn_relu(x, p_model_4_0_weight, p_model_4_1_weight, p_model_4_1_bias,
                       b_model_4_1_running_mean, b_model_4_1_running_var,
                       stride=2, padding=1, groups=128)
    x = conv2d_bn_relu(x, p_model_4_3_weight, p_model_4_4_weight, p_model_4_4_bias,
                       b_model_4_4_running_mean, b_model_4_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 5: conv_dw (256->256 dw, then 256->256 pw)
    x = conv2d_bn_relu(x, p_model_5_0_weight, p_model_5_1_weight, p_model_5_1_bias,
                       b_model_5_1_running_mean, b_model_5_1_running_var,
                       stride=1, padding=1, groups=256)
    x = conv2d_bn_relu(x, p_model_5_3_weight, p_model_5_4_weight, p_model_5_4_bias,
                       b_model_5_4_running_mean, b_model_5_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 6: conv_dw (256->256 dw stride=2, then 256->512 pw)
    x = conv2d_bn_relu(x, p_model_6_0_weight, p_model_6_1_weight, p_model_6_1_bias,
                       b_model_6_1_running_mean, b_model_6_1_running_var,
                       stride=2, padding=1, groups=256)
    x = conv2d_bn_relu(x, p_model_6_3_weight, p_model_6_4_weight, p_model_6_4_bias,
                       b_model_6_4_running_mean, b_model_6_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 7: conv_dw (512->512 dw, then 512->512 pw)
    x = conv2d_bn_relu(x, p_model_7_0_weight, p_model_7_1_weight, p_model_7_1_bias,
                       b_model_7_1_running_mean, b_model_7_1_running_var,
                       stride=1, padding=1, groups=512)
    x = conv2d_bn_relu(x, p_model_7_3_weight, p_model_7_4_weight, p_model_7_4_bias,
                       b_model_7_4_running_mean, b_model_7_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 8: conv_dw (512->512 dw, then 512->512 pw)
    x = conv2d_bn_relu(x, p_model_8_0_weight, p_model_8_1_weight, p_model_8_1_bias,
                       b_model_8_1_running_mean, b_model_8_1_running_var,
                       stride=1, padding=1, groups=512)
    x = conv2d_bn_relu(x, p_model_8_3_weight, p_model_8_4_weight, p_model_8_4_bias,
                       b_model_8_4_running_mean, b_model_8_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 9: conv_dw (512->512 dw, then 512->512 pw)
    x = conv2d_bn_relu(x, p_model_9_0_weight, p_model_9_1_weight, p_model_9_1_bias,
                       b_model_9_1_running_mean, b_model_9_1_running_var,
                       stride=1, padding=1, groups=512)
    x = conv2d_bn_relu(x, p_model_9_3_weight, p_model_9_4_weight, p_model_9_4_bias,
                       b_model_9_4_running_mean, b_model_9_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 10: conv_dw (512->512 dw, then 512->512 pw)
    x = conv2d_bn_relu(x, p_model_10_0_weight, p_model_10_1_weight, p_model_10_1_bias,
                       b_model_10_1_running_mean, b_model_10_1_running_var,
                       stride=1, padding=1, groups=512)
    x = conv2d_bn_relu(x, p_model_10_3_weight, p_model_10_4_weight, p_model_10_4_bias,
                       b_model_10_4_running_mean, b_model_10_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 11: conv_dw (512->512 dw, then 512->512 pw)
    x = conv2d_bn_relu(x, p_model_11_0_weight, p_model_11_1_weight, p_model_11_1_bias,
                       b_model_11_1_running_mean, b_model_11_1_running_var,
                       stride=1, padding=1, groups=512)
    x = conv2d_bn_relu(x, p_model_11_3_weight, p_model_11_4_weight, p_model_11_4_bias,
                       b_model_11_4_running_mean, b_model_11_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 12: conv_dw (512->512 dw stride=2, then 512->1024 pw)
    x = conv2d_bn_relu(x, p_model_12_0_weight, p_model_12_1_weight, p_model_12_1_bias,
                       b_model_12_1_running_mean, b_model_12_1_running_var,
                       stride=2, padding=1, groups=512)
    x = conv2d_bn_relu(x, p_model_12_3_weight, p_model_12_4_weight, p_model_12_4_bias,
                       b_model_12_4_running_mean, b_model_12_4_running_var,
                       stride=1, padding=0, groups=1)

    # Layer 13: conv_dw (1024->1024 dw, then 1024->1024 pw)
    x = conv2d_bn_relu(x, p_model_13_0_weight, p_model_13_1_weight, p_model_13_1_bias,
                       b_model_13_1_running_mean, b_model_13_1_running_var,
                       stride=1, padding=1, groups=1024)
    x = conv2d_bn_relu(x, p_model_13_3_weight, p_model_13_4_weight, p_model_13_4_bias,
                       b_model_13_4_running_mean, b_model_13_4_running_var,
                       stride=1, padding=0, groups=1)

    # AvgPool2d(7)
    x = torch.nn.functional.avg_pool2d(x, 7)

    # View
    x = x.view(x.size(0), -1)

    # Linear
    x = torch.nn.functional.linear(x, p_fc_weight, p_fc_bias)

    return x


def get_inputs():
    # All model parameters
    p_model_0_0_weight = torch.randn(32, 3, 3, 3, device='cuda')
    p_model_0_1_weight = torch.randn(32, device='cuda')
    p_model_0_1_bias = torch.randn(32, device='cuda')

    p_model_1_0_weight = torch.randn(32, 1, 3, 3, device='cuda')
    p_model_1_1_weight = torch.randn(32, device='cuda')
    p_model_1_1_bias = torch.randn(32, device='cuda')
    p_model_1_3_weight = torch.randn(64, 32, 1, 1, device='cuda')
    p_model_1_4_weight = torch.randn(64, device='cuda')
    p_model_1_4_bias = torch.randn(64, device='cuda')

    p_model_2_0_weight = torch.randn(64, 1, 3, 3, device='cuda')
    p_model_2_1_weight = torch.randn(64, device='cuda')
    p_model_2_1_bias = torch.randn(64, device='cuda')
    p_model_2_3_weight = torch.randn(128, 64, 1, 1, device='cuda')
    p_model_2_4_weight = torch.randn(128, device='cuda')
    p_model_2_4_bias = torch.randn(128, device='cuda')

    p_model_3_0_weight = torch.randn(128, 1, 3, 3, device='cuda')
    p_model_3_1_weight = torch.randn(128, device='cuda')
    p_model_3_1_bias = torch.randn(128, device='cuda')
    p_model_3_3_weight = torch.randn(128, 128, 1, 1, device='cuda')
    p_model_3_4_weight = torch.randn(128, device='cuda')
    p_model_3_4_bias = torch.randn(128, device='cuda')

    p_model_4_0_weight = torch.randn(128, 1, 3, 3, device='cuda')
    p_model_4_1_weight = torch.randn(128, device='cuda')
    p_model_4_1_bias = torch.randn(128, device='cuda')
    p_model_4_3_weight = torch.randn(256, 128, 1, 1, device='cuda')
    p_model_4_4_weight = torch.randn(256, device='cuda')
    p_model_4_4_bias = torch.randn(256, device='cuda')

    p_model_5_0_weight = torch.randn(256, 1, 3, 3, device='cuda')
    p_model_5_1_weight = torch.randn(256, device='cuda')
    p_model_5_1_bias = torch.randn(256, device='cuda')
    p_model_5_3_weight = torch.randn(256, 256, 1, 1, device='cuda')
    p_model_5_4_weight = torch.randn(256, device='cuda')
    p_model_5_4_bias = torch.randn(256, device='cuda')

    p_model_6_0_weight = torch.randn(256, 1, 3, 3, device='cuda')
    p_model_6_1_weight = torch.randn(256, device='cuda')
    p_model_6_1_bias = torch.randn(256, device='cuda')
    p_model_6_3_weight = torch.randn(512, 256, 1, 1, device='cuda')
    p_model_6_4_weight = torch.randn(512, device='cuda')
    p_model_6_4_bias = torch.randn(512, device='cuda')

    p_model_7_0_weight = torch.randn(512, 1, 3, 3, device='cuda')
    p_model_7_1_weight = torch.randn(512, device='cuda')
    p_model_7_1_bias = torch.randn(512, device='cuda')
    p_model_7_3_weight = torch.randn(512, 512, 1, 1, device='cuda')
    p_model_7_4_weight = torch.randn(512, device='cuda')
    p_model_7_4_bias = torch.randn(512, device='cuda')

    p_model_8_0_weight = torch.randn(512, 1, 3, 3, device='cuda')
    p_model_8_1_weight = torch.randn(512, device='cuda')
    p_model_8_1_bias = torch.randn(512, device='cuda')
    p_model_8_3_weight = torch.randn(512, 512, 1, 1, device='cuda')
    p_model_8_4_weight = torch.randn(512, device='cuda')
    p_model_8_4_bias = torch.randn(512, device='cuda')

    p_model_9_0_weight = torch.randn(512, 1, 3, 3, device='cuda')
    p_model_9_1_weight = torch.randn(512, device='cuda')
    p_model_9_1_bias = torch.randn(512, device='cuda')
    p_model_9_3_weight = torch.randn(512, 512, 1, 1, device='cuda')
    p_model_9_4_weight = torch.randn(512, device='cuda')
    p_model_9_4_bias = torch.randn(512, device='cuda')

    p_model_10_0_weight = torch.randn(512, 1, 3, 3, device='cuda')
    p_model_10_1_weight = torch.randn(512, device='cuda')
    p_model_10_1_bias = torch.randn(512, device='cuda')
    p_model_10_3_weight = torch.randn(512, 512, 1, 1, device='cuda')
    p_model_10_4_weight = torch.randn(512, device='cuda')
    p_model_10_4_bias = torch.randn(512, device='cuda')

    p_model_11_0_weight = torch.randn(512, 1, 3, 3, device='cuda')
    p_model_11_1_weight = torch.randn(512, device='cuda')
    p_model_11_1_bias = torch.randn(512, device='cuda')
    p_model_11_3_weight = torch.randn(512, 512, 1, 1, device='cuda')
    p_model_11_4_weight = torch.randn(512, device='cuda')
    p_model_11_4_bias = torch.randn(512, device='cuda')

    p_model_12_0_weight = torch.randn(512, 1, 3, 3, device='cuda')
    p_model_12_1_weight = torch.randn(512, device='cuda')
    p_model_12_1_bias = torch.randn(512, device='cuda')
    p_model_12_3_weight = torch.randn(1024, 512, 1, 1, device='cuda')
    p_model_12_4_weight = torch.randn(1024, device='cuda')
    p_model_12_4_bias = torch.randn(1024, device='cuda')

    p_model_13_0_weight = torch.randn(1024, 1, 3, 3, device='cuda')
    p_model_13_1_weight = torch.randn(1024, device='cuda')
    p_model_13_1_bias = torch.randn(1024, device='cuda')
    p_model_13_3_weight = torch.randn(1024, 1024, 1, 1, device='cuda')
    p_model_13_4_weight = torch.randn(1024, device='cuda')
    p_model_13_4_bias = torch.randn(1024, device='cuda')

    p_fc_weight = torch.randn(1000, 1024, device='cuda')
    p_fc_bias = torch.randn(1000, device='cuda')

    b_model_0_1_running_mean = torch.zeros(32, device='cuda')
    b_model_0_1_running_var = torch.ones(32, device='cuda')
    b_model_1_1_running_mean = torch.zeros(32, device='cuda')
    b_model_1_1_running_var = torch.ones(32, device='cuda')
    b_model_1_4_running_mean = torch.zeros(64, device='cuda')
    b_model_1_4_running_var = torch.ones(64, device='cuda')
    b_model_2_1_running_mean = torch.zeros(64, device='cuda')
    b_model_2_1_running_var = torch.ones(64, device='cuda')
    b_model_2_4_running_mean = torch.zeros(128, device='cuda')
    b_model_2_4_running_var = torch.ones(128, device='cuda')
    b_model_3_1_running_mean = torch.zeros(128, device='cuda')
    b_model_3_1_running_var = torch.ones(128, device='cuda')
    b_model_3_4_running_mean = torch.zeros(128, device='cuda')
    b_model_3_4_running_var = torch.ones(128, device='cuda')
    b_model_4_1_running_mean = torch.zeros(128, device='cuda')
    b_model_4_1_running_var = torch.ones(128, device='cuda')
    b_model_4_4_running_mean = torch.zeros(256, device='cuda')
    b_model_4_4_running_var = torch.ones(256, device='cuda')
    b_model_5_1_running_mean = torch.zeros(256, device='cuda')
    b_model_5_1_running_var = torch.ones(256, device='cuda')
    b_model_5_4_running_mean = torch.zeros(256, device='cuda')
    b_model_5_4_running_var = torch.ones(256, device='cuda')
    b_model_6_1_running_mean = torch.zeros(256, device='cuda')
    b_model_6_1_running_var = torch.ones(256, device='cuda')
    b_model_6_4_running_mean = torch.zeros(512, device='cuda')
    b_model_6_4_running_var = torch.ones(512, device='cuda')