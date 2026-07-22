import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    # x is in NCHW layout: index = n*C*HW + c*HW + hw
    # bias index = c = (offset // HW) % C
    c_idx = (offsets // HW) % C
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    result = x + b
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def bias_relu_inplace_kernel(
    x_ptr,
    bias_ptr,
    numel,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    c_idx = (offsets // HW) % C
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    result = x + b
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(x_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr,
    out_ptr,
    numel,
    N, C, H_in, W_in, H_out, W_out,
    kernel_h, kernel_w,
    stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # output index: n, c, oh, ow
    ow = offsets % W_out
    oh = (offsets // W_out) % H_out
    c = (offsets // (W_out * H_out)) % C
    n = offsets // (W_out * H_out * C)

    n_valid = n < N
    combined_mask = mask & n_valid

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(kernel_h):
        for kw in range(kernel_w):
            ih = oh * stride_h + kh
            iw = ow * stride_w + kw
            valid = combined_mask & (ih < H_in) & (iw < W_in)
            in_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + ih * W_in + iw
            val = tl.load(x_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    x_ptr,
    out_ptr,
    numel,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Output is (N, C, 1, 1), so numel = N*C
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c = offsets % C
    n = offsets // C

    n_valid = n < N
    combined_mask = mask & n_valid

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    HW = H * W

    for h in range(H):
        for w in range(W):
            in_idx = n * (C * H * W) + c * (H * W) + h * W + w
            val = tl.load(x_ptr + in_idx, mask=combined_mask, other=0.0)
            acc = acc + val

    result = acc / (H * W).to(tl.float32)
    tl.store(out_ptr + offsets, result, mask=mask)


def apply_bias_relu(x, bias):
    """Apply bias add + ReLU in-place using Triton kernel."""
    N, C, H, W = x.shape
    numel = x.numel()
    HW = H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bias_relu_inplace_kernel[grid](x, bias, numel, C, HW)
    return x


def triton_max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True):
    """Max pooling with ceil_mode using Triton."""
    N, C, H_in, W_in = x.shape
    if ceil_mode:
        H_out = (H_in + stride - 1) // stride
        W_out = (W_in + stride - 1) // stride
        # Adjust for ceil_mode with kernel_size
        H_out = (H_in - kernel_size + stride - 1 + stride) // stride
        W_out = (W_in - kernel_size + stride - 1 + stride) // stride
        # Standard ceil_mode formula
        H_out = (H_in - kernel_size + stride + stride - 1) // stride
        W_out = (W_in - kernel_size + stride + stride - 1) // stride
        # Correct ceil_mode: ceil((H_in - kernel_size) / stride) + 1
        import math
        H_out = math.ceil((H_in - kernel_size) / stride) + 1
        W_out = math.ceil((W_in - kernel_size) / stride) + 1
    else:
        H_out = (H_in - kernel_size) // stride + 1
        W_out = (W_in - kernel_size) // stride + 1

    out = torch.empty(N, C, H_out, W_out, device=x.device, dtype=x.dtype)
    numel = out.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out, numel,
        N, C, H_in, W_in, H_out, W_out,
        kernel_size, kernel_size,
        stride, stride,
    )
    return out


def triton_adaptive_avg_pool2d_1x1(x):
    """Adaptive average pooling to (1,1) using Triton."""
    N, C, H, W = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    numel = N * C
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
    )
    return out


def conv2d_bias_relu(x, weight, bias, stride=1, padding=0):
    """Conv2d + bias + ReLU using cuDNN conv + Triton epilogue."""
    # Use PyTorch's cuDNN for the convolution (no bias in conv, add separately)
    out = F.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    apply_bias_relu(out, bias)
    return out


def kernel_function(
    p_features_0_weight, p_features_0_bias,
    p_features_3_squeeze_weight, p_features_3_squeeze_bias,
    p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,
    p_features_3_expand3x3_weight, p_features_3_expand3x3_bias,
    p_features_4_squeeze_weight, p_features_4_squeeze_bias,
    p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,
    p_features_4_expand3x3_weight, p_features_4_expand3x3_bias,
    p_features_5_squeeze_weight, p_features_5_squeeze_bias,
    p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,
    p_features_5_expand3x3_weight, p_features_5_expand3x3_bias,
    p_features_7_squeeze_weight, p_features_7_squeeze_bias,
    p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,
    p_features_7_expand3x3_weight, p_features_7_expand3x3_bias,
    p_features_8_squeeze_weight, p_features_8_squeeze_bias,
    p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,
    p_features_8_expand3x3_weight, p_features_8_expand3x3_bias,
    p_features_9_squeeze_weight, p_features_9_squeeze_bias,
    p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,
    p_features_9_expand3x3_weight, p_features_9_expand3x3_bias,
    p_features_10_squeeze_weight, p_features_10_squeeze_bias,
    p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,
    p_features_10_expand3x3_weight, p_features_10_expand3x3_bias,
    p_features_12_squeeze_weight, p_features_12_squeeze_bias,
    p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,
    p_features_12_expand3x3_weight, p_features_12_expand3x3_bias,
    p_classifier_1_weight, p_classifier_1_bias,
    x,
):
    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    x = conv2d_bias_relu(x, p_features_0_weight, p_features_0_bias, stride=2, padding=0)
    # features[2]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = triton_max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    # Fire module 3: squeeze + expand
    def fire_module(inp, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
        sq = conv2d_bias_relu(inp, sq_w, sq_b, stride=1, padding=0)
        ex1 = conv2d_bias_relu(sq, ex1_w, ex1_b, stride=1, padding=0)
        ex3 = conv2d_bias_relu(sq, ex3_w, ex3_b, stride=1, padding=1)
        return torch.cat([ex1, ex3], dim=1)

    x = fire_module(x,
                    p_features_3_squeeze_weight, p_features_3_squeeze_bias,
                    p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,

    x = fire_module(x,
                    p_features_4_squeeze_weight, p_features_4_squeeze_bias,
                    p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,

    x = fire_module(x,
                    p_features_5_squeeze_weight, p_features_5_squeeze_bias,
                    p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,

    # MaxPool2d
    x = triton_max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    x = fire_module(x,
                    p_features_7_squeeze_weight, p_features_7_squeeze_bias,
                    p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,

    x = fire_module(x,
                    p_features_8_squeeze_weight, p_features_8_squeeze_bias,
                    p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,

    x = fire_module(x,
                    p_features_9_squeeze_weight, p_features_9_squeeze_bias,
                    p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,

    x = fire_module(x,
                    p_features_10_squeeze_weight, p_features_10_squeeze_bias,
                    p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,

    # MaxPool2d
    x = triton_max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    x = fire_module(x,
                    p_features_12_squeeze_weight, p_features_12_squeeze_bias,
                    p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,

    # Dropout (p=0.0, so identity in inference)
    # x = x  # no-op

    # Classifier conv + ReLU
    x = conv2d_bias_relu(x, p_classifier_1_weight, p_classifier_1_bias, stride=1, padding=0)

    # AdaptiveAvgPool2d(1,1)
    x = triton_adaptive_avg_pool2d_1x1(x)

    # Flatten
    x = x.reshape(x.shape[0], -1)

    return x


def get_inputs():
    return [
        torch.randn(96, 3, 7, 7, device='cuda'),      # p_features_0_weight
        torch.randn(96, device='cuda'),                 # p_features_0_bias
        torch.randn(16, 96, 1, 1, device='cuda'),      # p_features_3_squeeze_weight
        torch.randn(16, device='cuda'),                 # p_features_3_squeeze_bias
        torch.randn(64, 16, 1, 1, device='cuda'),      # p_features_3_expand1x1_weight
        torch.randn(64, device='cuda'),                 # p_features_3_expand1x1_bias
        torch.randn(64, 16, 3, 3, device='cuda'),      # p_features_3_expand3x3_weight
        torch.randn(64, device='cuda'),                 # p_features_3_expand3x3_bias
        torch.randn(16, 128, 1, 1, device='cuda'),     # p_features_4_squeeze_weight
        torch.randn(16, device='cuda'),                 # p_features_4_squeeze_bias
        torch.randn(64, 16, 1, 1, device='cuda'),      # p_features_4_expand1x1_weight
        torch.randn(64, device='cuda'),                 # p_features_4_expand1x1_bias
        torch.randn(64, 16, 3, 3, device='cuda'),      # p_features_4_expand3x3_weight
        torch.randn(64, device='cuda'),                 # p_features_4_expand3x3_bias
        torch.randn(32, 128, 1, 1, device='cuda'),     # p_features_5_squeeze_weight
        torch.randn(32, device='cuda'),                 # p_features_5_squeeze_bias
        torch.randn(128, 32, 1, 1, device='cuda'),     # p_features_5_expand1x1_weight
        torch.randn(128, device='cuda'),                # p_features_5_expand1x1_bias
        torch.randn(128, 32, 3, 3, device='cuda'),     # p_features_5_expand3x3_weight
        torch.randn(128, device='cuda'),                # p_features_5_expand3x3_bias
        torch.randn(32, 256, 1, 1, device='cuda'),     # p_features_7_squeeze_weight
        torch.randn(32, device='cuda'),                 # p_features_7_squeeze_bias
        torch.randn(128, 32, 1, 1, device='cuda'),     # p_features_7_expand1x1_weight
        torch.randn(128, device='cuda'),                # p_features_7_expand1x1_bias
        torch.randn(128, 32, 3, 3, device='cuda'),     # p_features_7_expand3x3_weight
        torch.randn(128, device='cuda'),                # p_features_7_expand3x3_bias
        torch.randn(48, 256, 1, 1, device='cuda'),     # p_features_8_squeeze_weight
        torch.randn(48, device='cuda'),                 # p_features_8_squeeze_bias
        torch.randn(192, 48, 1, 1, device='cuda'),     # p_features_8_expand1x1_weight
        torch.randn(192, device='cuda'),                # p_features_8_expand1x1_bias
        torch.randn(192, 48, 3, 3, device='cuda'),     # p_features_8_expand3x3_weight
        torch.randn(192, device='cuda'),                # p_features_8_expand3x3_bias
        torch.randn(48, 384, 1, 1, device='cuda'),     # p_features_9_squeeze_weight
        torch.randn(48, device='cuda'),                 # p_features_9_squeeze_bias
        torch.randn(192, 48, 1, 1, device='cuda'),     # p_features_9_expand1x1_weight
        torch.randn(192, device='cuda'),                # p_features_9_expand1x1_bias
        torch.randn(192, 48, 3, 3, device='cuda'),     # p_features_9_expand3x3_weight
        torch.randn(192, device='cuda'),                # p_features_9_expand3x3_bias
        torch.randn(64, 384, 1, 1, device='cuda'),     # p_features_10_squeeze_weight
        torch.randn(64, device='cuda'),                 # p_features_10_squeeze_bias
        torch.randn(256, 64, 1, 1, device='cuda'),     # p_features_10_expand1x1_weight
        torch.randn(256, device='cuda'),                # p_features_10_expand1x1_bias
        torch.randn(256, 64, 3, 3, device='cuda'),     # p_features_10_expand3x3_weight
        torch.randn(256, device='cuda'),                # p_features_10_expand3x3_bias
        torch.randn(64, 512, 1, 1, device='cuda'),     # p_features_12_squeeze_weight
        torch.randn(64, device='cuda'),                 # p_features_12_squeeze_bias
        torch.randn(256, 64, 1, 1, device='cuda'),     # p_features_12_expand1x1_weight
        torch.randn(256, device='cuda'),                # p_features_12_expand1x1_bias
        torch.randn(256, 64, 3, 3, device='cuda'),     # p_features_12_expand3x3_weight
        torch.randn(256, device='cuda'),                # p_features_12_expand3x3_bias
        torch.randn(1000, 512, 1, 1, device='cuda'),   # p_classifier_1_weight
        torch.randn(1000, device='cuda'),               # p_classifier_1_bias
        torch.randn(64, 3, 512, 512, device='cuda'),   # x
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        class FireModule(nn.Module):
            def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
                super(FireModule, self).__init__()
                self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
                self.squeeze_activation = nn.ReLU(inplace=True)
                self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
                self.expand1x1_activation = nn.ReLU(inplace=True)
                self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
                self.expand3x3_activation = nn.ReLU(inplace=True)

            def forward(self, x):
                x = self.squeeze_activation(self.squeeze(x))
                return torch.cat([
                    self.expand1x1_activation(self.expand1x1(x)),
                    self.expand3x3_activation(self.expand3x3(x))
                ], 1)

        class Model(nn.Module):
            def __init__(self, num_classes=1000):
                super(Model, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=7, stride=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    FireModule(96, 16, 64, 64),
                    FireModule(128, 16, 64, 64),
                    FireModule(128, 32, 128, 128),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    FireModule(256, 32, 128, 128),
                    FireModule(256, 48, 192, 192),
                    FireModule(384, 48, 192, 192),
                    FireModule(384, 64, 256, 256),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    FireModule(512, 64, 256, 256),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(p=0.0),
                    nn.Conv2d(512, num_classes, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d((1, 1))
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return torch.flatten(x, 1)

        model = Model().cuda().eval()

        # Set weights from inputs
        inp = inputs
        model.features[0].weight.data = inp[0].clone()
        model.features[0].bias.data = inp[1].clone()
        # Fire 3
        model.features[3].squeeze.weight.data = inp[2].clone()
        model.features[3].squeeze.bias.data = inp[3].clone()
        model.features[3].expand1x1.weight.data = inp[4].clone()
        model.features[3].expand1x1.bias.data = inp[5].clone()
        model.features[3].expand3x3.weight.data = inp[6].clone()
        model.features[3].expand3x3.bias.data = inp[7].clone()
        # Fire 4
        model.features[4].squeeze.weight.data = inp[8].clone()
        model.features[4].squeeze.bias.data = inp[9].clone()
        model.features[4].expand1x1.weight.data = inp[10].clone()
        model.features[4].expand1x1.bias.data = inp[11].clone()
        model.features[4].expand3x3.weight.data = inp[12].clone()
        model.features[4].expand3x3.bias.data = inp[13].clone()
        # Fire 5
        model.features[5].squeeze.weight.data = inp[14].clone()
        model.features[5].squeeze.bias.data = inp[15].clone()
        model.features[5].expand1x1.weight.data = inp[16].clone()
        model.features[5].expand1x1.bias.data = inp[17].clone()
        model.features[5].expand3x3.weight.data = inp[18].clone()
        model.features[5].expand3x3.bias.data = inp[19].clone()
        # Fire 7
        model.features[7].squeeze.weight.data = inp[20].clone()
        model.features[7].squeeze.bias.data = inp[21].clone()
        model.features[7].expand1x1.weight.data = inp[22].clone()
        model.features[7].expand1x1.bias.data = inp[23].clone()
        model.features[7].expand3x3.weight.data = inp[24].clone()
        model.features[7].expand3x3.bias.data = inp[25].clone()
        # Fire 8
        model.features[8].squeeze.weight.data = inp[26].clone()
        model.features[8].squeeze.bias.data = inp[27].clone()
        model.features[8].expand1x1.weight.data = inp[28].clone()
        model.features[8].expand1x1.bias.data = inp[29].clone()
        model.features[8].expand3x3.weight.data = inp[30].clone()
        model.features[8].expand3x3.bias.data = inp[31].clone()
        # Fire 9
        model.features[9].squeeze.weight.data = inp[32].clone()
        model.features[9].squeeze.bias.data = inp[33].clone()
        model.features[9].expand1x1.weight.data = inp[34].clone()
        model.features[9].expand1x1.bias.data = inp[35].clone()
        model.features[9].expand3x3.weight.data = inp[36].clone()
        model.features[9].expand3x3.bias.data = inp[37].clone()
        # Fire 10
        model.features[10].squeeze.weight.data = inp[38].clone()
        model.features[10].squeeze.bias.data = inp[39].clone()
        model.features[10].expand1x1.weight.data = inp[40].clone()
        model.features[10].expand1x1.bias.data = inp[41].clone()
        model.features[10].expand3x3.weight.data = inp[42].clone()
        model.features[10].expand3x3.bias.data = inp[43].clone()
        # Fire 12
        model.features[12].squeeze.weight.data