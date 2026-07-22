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
    key=['n_elements'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # bias index is channel dimension
    bias_idx = offsets % n_channels
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def bias_relu_nhwc_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    """For NCHW tensors where bias is along channel dim (dim 1)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias_idx = offsets % n_channels
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def bias_add_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias_idx = offsets % n_channels
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    result = x + bias
    tl.store(out_ptr + offsets, result, mask=mask)


def conv2d_bias_relu(x, weight, bias, stride=1, padding=0):
    """Conv2d followed by bias add and ReLU, fused in Triton."""
    import torch.nn.functional as F
    # Use cuDNN for conv (no bias in conv call)
    out = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    # out shape: (N, C, H, W) - contiguous in NCHW
    # For NCHW layout, channel is dim 1, so bias index = (flat_idx // (H*W)) % C
    # But if we flatten, we need to know H*W
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    
    # For NCHW, flat index i -> channel = (i // (H*W)) % C
    # This is complex for a simple kernel. Instead, reshape to (N*C, H*W) and use C as stride
    # Actually for contiguous NCHW: flat_idx = n*C*H*W + c*H*W + h*W + w
    # So channel = (flat_idx // (H*W)) % C
    # We need a different approach - use spatial_size = H*W
    
    # Use a kernel that knows spatial_size
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    # We'll use a specialized kernel
    _conv_bias_relu_nchw_kernel(out, bias, result, N, C, H, W)
    return result


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def conv_bias_relu_nchw_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    """Add bias and apply ReLU for NCHW tensor. bias shape: [C]"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = N * C * HW
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channel index for NCHW: (offset // HW) % C
    c_idx = (offsets // HW) % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


def _conv_bias_relu_nchw_kernel(x, bias, out, N, C, H, W):
    HW = H * W
    n_elements = N * C * HW
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    conv_bias_relu_nchw_kernel[grid](x, bias, out, N, C, HW)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def conv_bias_nchw_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    N,
    C,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    """Add bias for NCHW tensor. bias shape: [C]"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    n_elements = N * C * HW
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = (offsets // HW) % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    result = x + bias
    tl.store(out_ptr + offsets, result, mask=mask)


def fused_conv_bias_relu(x, weight, bias, stride=1, padding=0):
    """Fused conv2d + bias + relu."""
    import torch.nn.functional as F
    out = F.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    N, C, H, W = out.shape
    HW = H * W
    n_elements = N * C * HW
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    conv_bias_relu_nchw_kernel[grid](out, bias, result, N, C, HW)
    return result


def maxpool2d_ceil(x, kernel_size=3, stride=2):
    """Max pool with ceil_mode=True."""
    import torch.nn.functional as F
    return F.max_pool2d(x, kernel_size=kernel_size, stride=stride, ceil_mode=True)


def fire_module(x, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """Fire module: squeeze -> expand1x1 + expand3x3 -> cat."""
    # Squeeze
    sq = fused_conv_bias_relu(x, sq_w, sq_b, stride=1, padding=0)
    # Expand 1x1
    ex1 = fused_conv_bias_relu(sq, ex1_w, ex1_b, stride=1, padding=0)
    # Expand 3x3
    ex3 = fused_conv_bias_relu(sq, ex3_w, ex3_b, stride=1, padding=1)
    # Cat
    return torch.cat([ex1, ex3], dim=1)


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
    x = fused_conv_bias_relu(x, p_features_0_weight, p_features_0_bias, stride=2, padding=0)
    # features[1]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = maxpool2d_ceil(x, kernel_size=3, stride=2)
    # features[3]: FireModule(96, 16, 64, 64)
    x = fire_module(x,
                    p_features_3_squeeze_weight, p_features_3_squeeze_bias,
                    p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,
    # features[4]: FireModule(128, 16, 64, 64)
    x = fire_module(x,
                    p_features_4_squeeze_weight, p_features_4_squeeze_bias,
                    p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,
    # features[5]: FireModule(128, 32, 128, 128)
    x = fire_module(x,
                    p_features_5_squeeze_weight, p_features_5_squeeze_bias,
                    p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,
    # features[6]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = maxpool2d_ceil(x, kernel_size=3, stride=2)
    # features[7]: FireModule(256, 32, 128, 128)
    x = fire_module(x,
                    p_features_7_squeeze_weight, p_features_7_squeeze_bias,
                    p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,
    # features[8]: FireModule(256, 48, 192, 192)
    x = fire_module(x,
                    p_features_8_squeeze_weight, p_features_8_squeeze_bias,
                    p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,
    # features[9]: FireModule(384, 48, 192, 192)
    x = fire_module(x,
                    p_features_9_squeeze_weight, p_features_9_squeeze_bias,
                    p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,
    # features[10]: FireModule(384, 64, 256, 256)
    x = fire_module(x,
                    p_features_10_squeeze_weight, p_features_10_squeeze_bias,
                    p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,
    # features[11]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = maxpool2d_ceil(x, kernel_size=3, stride=2)
    # features[12]: FireModule(512, 64, 256, 256)
    x = fire_module(x,
                    p_features_12_squeeze_weight, p_features_12_squeeze_bias,
                    p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,
    # classifier[0]: Dropout(p=0.0) - identity at inference
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    x = fused_conv_bias_relu(x, p_classifier_1_weight, p_classifier_1_bias, stride=1, padding=0)
    # classifier[3]: AdaptiveAvgPool2d((1, 1))
    x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    # flatten
    x = torch.flatten(x, 1)
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

        # Load weights from inputs
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
         x_input) = inputs

        with torch.no_grad():
            model.features[0].weight.copy_(p_features_0_weight)
            model.features[0].bias.copy_(p_features_0_bias)
            model.features[3].squeeze.weight.copy_(p_features_3_squeeze_weight)
            model.features[3].squeeze.bias.copy_(p_features_3_squeeze_bias)
            model.features[3].expand1x1.weight.copy_(p_features_3_expand1x1_weight)
            model.features[3].expand1x1.bias.copy_(p_features_3