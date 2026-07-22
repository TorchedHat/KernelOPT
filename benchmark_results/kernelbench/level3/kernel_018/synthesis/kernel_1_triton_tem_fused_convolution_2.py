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
    key=['numel'],
)
@triton.jit
def fused_bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    C,
    spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index for bias
    # Layout: (N, C, H, W) -> channel = (offset // spatial) % C
    c_idx = (offsets // spatial) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    result = x + b
    result = tl.where(result > 0.0, result, 0.0)

    tl.store(out_ptr + offsets, result, mask=mask)


def fused_bias_relu(x, bias):
    """Apply bias add and ReLU in a single Triton kernel."""
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bias_relu_kernel[grid](
        x, bias, out,
        numel, C, spatial,
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
def fused_bias_relu_inplace_kernel(
    x_ptr,
    bias_ptr,
    numel,
    C,
    spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c_idx = (offsets // spatial) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    result = x + b
    result = tl.where(result > 0.0, result, 0.0)

    tl.store(x_ptr + offsets, result, mask=mask)


def fused_bias_relu_inplace(x, bias):
    """Apply bias add and ReLU in-place."""
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial = H * W

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bias_relu_inplace_kernel[grid](
        x, bias,
        numel, C, spatial,
    )
    return x


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
def max_pool2d_kernel(
    x_ptr,
    out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    kernel_h, kernel_w,
    stride_h, stride_w,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Decode output index (N, C, H_out, W_out)
    w_out = offsets % W_out
    h_out = (offsets // W_out) % H_out
    c = (offsets // (W_out * H_out)) % C
    n = offsets // (W_out * H_out * C)

    h_start = h_out * stride_h
    w_start = w_out * stride_w

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(kernel_h):
        for kw in range(kernel_w):
            h_in = h_start + kh
            w_in = w_start + kw
            valid = mask & (h_in < H_in) & (w_in < W_in)
            in_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in
            val = tl.load(x_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid & (val > max_val), val, max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True):
    """Max pooling using Triton kernel."""
    N, C, H_in, W_in = x.shape
    if ceil_mode:
        H_out = (H_in + stride - 1) // stride
        W_out = (W_in + stride - 1) // stride
        # Adjust for ceil_mode with kernel_size
        H_out = (H_in - kernel_size + stride - 1 + stride) // stride
        W_out = (W_in - kernel_size + stride - 1 + stride) // stride
        # Standard ceil_mode formula
        import math
        H_out = math.ceil((H_in - kernel_size) / stride) + 1
        W_out = math.ceil((W_in - kernel_size) / stride) + 1
    else:
        H_out = (H_in - kernel_size) // stride + 1
        W_out = (W_in - kernel_size) // stride + 1

    out = torch.empty(N, C, H_out, W_out, device=x.device, dtype=x.dtype)
    numel = N * C * H_out * W_out

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out,
        N, C, H_in, W_in, H_out, W_out,
        kernel_size, kernel_size,
        stride, stride,
        numel,
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
def adaptive_avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    N, C, H_in, W_in,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Adaptive average pooling to (1, 1) output."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Output is (N, C, 1, 1), so offsets index into (n, c)
    c = offsets % C
    n = offsets // C

    spatial = H_in * W_in
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    base = n * C * spatial + c * spatial

    # Sum all spatial elements
    for h in range(H_in):
        for w in range(W_in):
            idx = base + h * W_in + w
            val = tl.load(x_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    avg = acc / (H_in * W_in)
    tl.store(out_ptr + offsets, avg, mask=mask)


def triton_adaptive_avg_pool2d(x):
    """Adaptive average pooling to (1, 1) using Triton."""
    N, C, H_in, W_in = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    numel = N * C

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        N, C, H_in, W_in,
        numel,
    )
    return out


def conv2d_bias_relu(x, weight, bias, stride=1, padding=0):
    """Conv2d followed by bias add and ReLU using Triton epilogue."""
    # Use PyTorch's conv2d (cuDNN) without bias, then fuse bias+relu in Triton
    out = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    return fused_bias_relu_inplace(out, bias)


def fire_module_forward(x,
                        sq_w, sq_b,
                        ex1_w, ex1_b,
    """Forward pass of a FireModule."""
    # Squeeze
    x = conv2d_bias_relu(x, sq_w, sq_b, stride=1, padding=0)
    # Expand 1x1
    e1 = conv2d_bias_relu(x, ex1_w, ex1_b, stride=1, padding=0)
    # Expand 3x3
    e3 = conv2d_bias_relu(x, ex3_w, ex3_b, stride=1, padding=1)
    return torch.cat([e1, e3], dim=1)


def kernel_function(
    p_features_0_weight,
    p_features_0_bias,
    p_features_3_squeeze_weight,
    p_features_3_squeeze_bias,
    p_features_3_expand1x1_weight,
    p_features_3_expand1x1_bias,
    p_features_3_expand3x3_weight,
    p_features_3_expand3x3_bias,
    p_features_4_squeeze_weight,
    p_features_4_squeeze_bias,
    p_features_4_expand1x1_weight,
    p_features_4_expand1x1_bias,
    p_features_4_expand3x3_weight,
    p_features_4_expand3x3_bias,
    p_features_5_squeeze_weight,
    p_features_5_squeeze_bias,
    p_features_5_expand1x1_weight,
    p_features_5_expand1x1_bias,
    p_features_5_expand3x3_weight,
    p_features_5_expand3x3_bias,
    p_features_7_squeeze_weight,
    p_features_7_squeeze_bias,
    p_features_7_expand1x1_weight,
    p_features_7_expand1x1_bias,
    p_features_7_expand3x3_weight,
    p_features_7_expand3x3_bias,
    p_features_8_squeeze_weight,
    p_features_8_squeeze_bias,
    p_features_8_expand1x1_weight,
    p_features_8_expand1x1_bias,
    p_features_8_expand3x3_weight,
    p_features_8_expand3x3_bias,
    p_features_9_squeeze_weight,
    p_features_9_squeeze_bias,
    p_features_9_expand1x1_weight,
    p_features_9_expand1x1_bias,
    p_features_9_expand3x3_weight,
    p_features_9_expand3x3_bias,
    p_features_10_squeeze_weight,
    p_features_10_squeeze_bias,
    p_features_10_expand1x1_weight,
    p_features_10_expand1x1_bias,
    p_features_10_expand3x3_weight,
    p_features_10_expand3x3_bias,
    p_features_12_squeeze_weight,
    p_features_12_squeeze_bias,
    p_features_12_expand1x1_weight,
    p_features_12_expand1x1_bias,
    p_features_12_expand3x3_weight,
    p_features_12_expand3x3_bias,
    p_classifier_1_weight,
    p_classifier_1_bias,
    x,
):
    import torch.nn.functional as F

    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    x = F.conv2d(x, p_features_0_weight, bias=None, stride=2, padding=0)
    x = fused_bias_relu_inplace(x, p_features_0_bias)

    # features[2]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = triton_max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    # features[3]: FireModule(96, 16, 64, 64)
    x = fire_module_forward(x,
                            p_features_3_squeeze_weight, p_features_3_squeeze_bias,
                            p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,

    # features[4]: FireModule(128, 16, 64, 64)
    x = fire_module_forward(x,
                            p_features_4_squeeze_weight, p_features_4_squeeze_bias,
                            p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,

    # features[5]: FireModule(128, 32, 128, 128)
    x = fire_module_forward(x,
                            p_features_5_squeeze_weight, p_features_5_squeeze_bias,
                            p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,

    # features[6]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = triton_max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    # features[7]: FireModule(256, 32, 128, 128)
    x = fire_module_forward(x,
                            p_features_7_squeeze_weight, p_features_7_squeeze_bias,
                            p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,

    # features[8]: FireModule(256, 48, 192, 192)
    x = fire_module_forward(x,
                            p_features_8_squeeze_weight, p_features_8_squeeze_bias,
                            p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,

    # features[9]: FireModule(384, 48, 192, 192)
    x = fire_module_forward(x,
                            p_features_9_squeeze_weight, p_features_9_squeeze_bias,
                            p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,

    # features[10]: FireModule(384, 64, 256, 256)
    x = fire_module_forward(x,
                            p_features_10_squeeze_weight, p_features_10_squeeze_bias,
                            p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,

    # features[11]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = triton_max_pool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    # features[12]: FireModule(512, 64, 256, 256)
    x = fire_module_forward(x,
                            p_features_12_squeeze_weight, p_features_12_squeeze_bias,
                            p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,

    # classifier[0]: Dropout (p=0.0, no-op in inference)
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    x = F.conv2d(x, p_classifier_1_weight, bias=None, stride=1, padding=0)
    x = fused_bias_relu_inplace(x, p_classifier_1_bias)

    # classifier[3]: AdaptiveAvgPool2d((1, 1))
    x = triton_adaptive_avg_pool2d(x)

    # flatten
    x = x.view(x.shape[0], -1)

    return x


def get_inputs():
    batch_size = 64
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
        torch.randn(batch_size, 3, 512, 512, device='cuda'),  # x
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
        import torch.nn.functional as F

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

        # Build reference model with the same weights
        model = Model(num_classes=1000).cuda().eval()

        # Assign weights from inputs
        (p_features_0_weight, p_features_0_bias,
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