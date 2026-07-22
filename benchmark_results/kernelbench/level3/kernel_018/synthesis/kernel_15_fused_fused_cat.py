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
    key=['n_elements'],
)
@triton.jit
def fused_bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    channel_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index for bias
    # For NCHW layout: channel = (offset / channel_stride) % n_channels
    chan_idx = (offsets // channel_stride) % n_channels
    bias = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0)

    result = x + bias
    result = tl.maximum(result, tl.zeros_like(result))

    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def fused_bias_relu_inplace_kernel(
    x_ptr,
    bias_ptr,
    n_elements,
    n_channels,
    channel_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    chan_idx = (offsets // channel_stride) % n_channels
    bias = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0)

    result = x + bias
    result = tl.maximum(result, tl.zeros_like(result))

    tl.store(x_ptr + offsets, result, mask=mask)


def apply_bias_relu(x, bias):
    """Apply bias and ReLU in-place using Triton kernel."""
    N, C, H, W = x.shape
    n_elements = N * C * H * W
    channel_stride = H * W

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bias_relu_inplace_kernel[grid](
    )
    return x


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr,
    out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Decode output index: N, C, H_out, W_out
    w_out = offsets % W_out
    h_out = (offsets // W_out) % H_out
    c_idx = (offsets // (W_out * H_out)) % C
    n_idx = offsets // (W_out * H_out * C)

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(kernel_h):
        for kw in range(kernel_w):
            h_in = h_out * stride_h + kh - pad_h
            w_in = w_out * stride_w + kw - pad_w
            valid = mask & (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
            in_idx = n_idx * C * H_in * W_in + c_idx * H_in * W_in + h_in * W_in + w_in
            val = tl.load(x_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    N, C, H_in, W_in,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Adaptive avg pool to (1,1) output."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # offsets index into output of shape (N, C, 1, 1)
    c_idx = offsets % C
    n_idx = offsets // C

    # Sum over all H_in * W_in
    total = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = H_in * W_in

    for h in range(H_in):
        for w in range(W_in):
            in_idx = n_idx * C * H_in * W_in + c_idx * H_in * W_in + h * W_in + w
            val = tl.load(x_ptr + in_idx, mask=mask, other=0.0)
            total = total + val

    avg = total / count
    tl.store(out_ptr + offsets, avg, mask=mask)


def fire_module(x, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """Fire module: squeeze + expand (1x1 and 3x3) with ReLU."""
    # Squeeze
    sq = torch.nn.functional.conv2d(x, sq_w, None)
    apply_bias_relu(sq, sq_b)

    # Expand 1x1
    ex1 = torch.nn.functional.conv2d(sq, ex1_w, None)
    apply_bias_relu(ex1, ex1_b)

    # Expand 3x3
    ex3 = torch.nn.functional.conv2d(sq, ex3_w, None, padding=1)
    apply_bias_relu(ex3, ex3_b)

    return torch.cat([ex1, ex3], dim=1)


def do_max_pool(x, kernel_size=3, stride=2, ceil_mode=True):
    """Max pooling using PyTorch (complex boundary logic)."""
    return torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, ceil_mode=ceil_mode)


def do_adaptive_avg_pool(x):
    """Adaptive average pooling to (1,1) using Triton."""
    N, C, H, W = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    n_elements = N * C
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
    )
    return out


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
    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    out = torch.nn.functional.conv2d(x, p_features_0_weight, None, stride=2)
    apply_bias_relu(out, p_features_0_bias)

    # features[2]: MaxPool2d(3, stride=2, ceil_mode=True)
    out = do_max_pool(out, kernel_size=3, stride=2, ceil_mode=True)

    # features[3]: Fire(96, 16, 64, 64)
    out = fire_module(out,
                      p_features_3_squeeze_weight, p_features_3_squeeze_bias,
                      p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,

    # features[4]: Fire(128, 16, 64, 64)
    out = fire_module(out,
                      p_features_4_squeeze_weight, p_features_4_squeeze_bias,
                      p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,

    # features[5]: Fire(128, 32, 128, 128)
    out = fire_module(out,
                      p_features_5_squeeze_weight, p_features_5_squeeze_bias,
                      p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,

    # features[6]: MaxPool2d(3, stride=2, ceil_mode=True)
    out = do_max_pool(out, kernel_size=3, stride=2, ceil_mode=True)

    # features[7]: Fire(256, 32, 128, 128)
    out = fire_module(out,
                      p_features_7_squeeze_weight, p_features_7_squeeze_bias,
                      p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,

    # features[8]: Fire(256, 48, 192, 192)
    out = fire_module(out,
                      p_features_8_squeeze_weight, p_features_8_squeeze_bias,
                      p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,

    # features[9]: Fire(384, 48, 192, 192)
    out = fire_module(out,
                      p_features_9_squeeze_weight, p_features_9_squeeze_bias,
                      p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,

    # features[10]: Fire(384, 64, 256, 256)
    out = fire_module(out,
                      p_features_10_squeeze_weight, p_features_10_squeeze_bias,
                      p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,

    # features[11]: MaxPool2d(3, stride=2, ceil_mode=True)
    out = do_max_pool(out, kernel_size=3, stride=2, ceil_mode=True)

    # features[12]: Fire(512, 64, 256, 256)
    out = fire_module(out,
                      p_features_12_squeeze_weight, p_features_12_squeeze_bias,
                      p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,

    # classifier[0]: Dropout (p=0.0, no-op in inference)
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    out = torch.nn.functional.conv2d(out, p_classifier_1_weight, None)
    apply_bias_relu(out, p_classifier_1_bias)

    # classifier[3]: AdaptiveAvgPool2d((1,1))
    out = do_adaptive_avg_pool(out)

    # flatten
    out = out.view(out.shape[0], -1)

    return out


def get_inputs():
    batch_size = 64
    return [
        torch.randn(96, 3, 7, 7, dtype=torch.float32, device='cuda'),      # p_features_0_weight
        torch.randn(96, dtype=torch.float32, device='cuda'),                 # p_features_0_bias
        torch.randn(16, 96, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_3_squeeze_weight
        torch.randn(16, dtype=torch.float32, device='cuda'),                 # p_features_3_squeeze_bias
        torch.randn(64, 16, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_3_expand1x1_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_3_expand1x1_bias
        torch.randn(64, 16, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_3_expand3x3_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_3_expand3x3_bias
        torch.randn(16, 128, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_4_squeeze_weight
        torch.randn(16, dtype=torch.float32, device='cuda'),                 # p_features_4_squeeze_bias
        torch.randn(64, 16, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_4_expand1x1_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_4_expand1x1_bias
        torch.randn(64, 16, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_4_expand3x3_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_4_expand3x3_bias
        torch.randn(32, 128, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_5_squeeze_weight
        torch.randn(32, dtype=torch.float32, device='cuda'),                 # p_features_5_squeeze_bias
        torch.randn(128, 32, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_5_expand1x1_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                # p_features_5_expand1x1_bias
        torch.randn(128, 32, 3, 3, dtype=torch.float32, device='cuda'),     # p_features_5_expand3x3_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                # p_features_5_expand3x3_bias
        torch.randn(32, 256, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_7_squeeze_weight
        torch.randn(32, dtype=torch.float32, device='cuda'),                 # p_features_7_squeeze_bias
        torch.randn(128, 32, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_7_expand1x1_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                # p_features_7_expand1x1_bias
        torch.randn(128, 32, 3, 3, dtype=torch.float32, device='cuda'),     # p_features_7_expand3x3_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                # p_features_7_expand3x3_bias
        torch.randn(48, 256, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_8_squeeze_weight
        torch.randn(48, dtype=torch.float32, device='cuda'),                 # p_features_8_squeeze_bias
        torch.randn(192, 48, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_8_expand1x1_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                # p_features_8_expand1x1_bias
        torch.randn(192, 48, 3, 3, dtype=torch.float32, device='cuda'),     # p_features_8_expand3x3_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                # p_features_8_expand3x3_bias
        torch.randn(48, 384, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_9_squeeze_weight
        torch.randn(48, dtype=torch.float32, device='cuda'),                 # p_features_9_squeeze_bias
        torch.randn(192, 48, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_9_expand1x1_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                # p_features_9_expand1x1_bias
        torch.randn(192, 48, 3, 3, dtype=torch.float32, device='cuda'),     # p_features_9_expand3x3_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                # p_features_9_expand3x3_bias
        torch.randn(64, 384, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_10_squeeze_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_10_squeeze_bias
        torch.randn(256, 64, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_10_expand1x1_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),                # p_features_10_expand1x1_bias
        torch.randn(256, 64, 3, 3, dtype=torch.float32, device='cuda'),     # p_features_10_expand3x3_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),                # p_features_10_expand3x3_bias
        torch.randn(64, 512, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_12_squeeze_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_12_squeeze_bias
        torch.randn(256, 64, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_12_expand1x1_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),                # p_features_12_expand1x1_bias
        torch.randn(256, 64, 3, 3, dtype=torch.float32, device='cuda'),     # p_features_12_expand3x3_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),                # p_features_12_expand3x3_bias
        torch.randn(1000, 512, 1, 1, dtype=torch.float32, device='cuda'),   # p_classifier_1_weight
        torch.randn(1000, dtype=torch.float32, device='cuda'),               # p_classifier_1_bias
        torch.randn(batch_size, 3, 512, 512, dtype=torch.float32, device='cuda'),  # x
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

        # Build reference model with same weights
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
         p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,
         p_features_12_expand3x3_weight, p_features_12_expand3x3_bias,
         p_classifier_1_weight, p_classifier_1_bias,
         x) = inputs

        model = Model(num_classes=1000).cuda()
        model.eval()

        with torch.no_grad():
            # Set weights
            model.features[0].weight.data.copy_(p_features_0_weight)
            model.features[0].bias.data.copy_(p_features_0_bias)

            model.features[3].squeeze.weight.data.copy_(p_features_3_squeeze_weight)
            model.features[3].squeeze.bias.data.copy_(p_features_3_squeeze_bias)
            model.features[3].expand1x1.weight.data.copy_(p_features_3_expand1x1_weight)
            model.features[3].expand1x1.bias.data.copy_(p_