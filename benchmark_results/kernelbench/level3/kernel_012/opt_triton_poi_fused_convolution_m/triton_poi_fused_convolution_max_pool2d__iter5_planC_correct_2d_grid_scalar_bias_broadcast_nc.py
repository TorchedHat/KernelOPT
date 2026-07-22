import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=3),
    ],
    key=['channel_stride'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    channel_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid_s = tl.program_id(0)
    pid_c = tl.program_id(1)
    # Load bias as a scalar broadcast: one bias value per channel plane
    # pid_c % n_channels keeps index within [C]-shaped bias tensor
    bias_idx = pid_c % n_channels
    bias_val = tl.load(bias_ptr + bias_idx, mask=bias_idx < n_channels, other=0.0)
    spatial_offsets = pid_s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = spatial_offsets < channel_stride
    flat_offsets = pid_c * channel_stride + spatial_offsets
    x = tl.load(x_ptr + flat_offsets, mask=mask, other=0.0, eviction_policy='evict_first')
    result = tl.maximum(x + bias_val, 0.0)
    tl.store(out_ptr + flat_offsets, result, mask=mask, eviction_policy='evict_first')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['spatial_size'],
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
    """For NHWC layout where channel is the last dimension."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    chan = offsets % n_channels
    bias = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    result = x + bias
    result = tl.maximum(result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def bias_relu_linear_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    """For linear layer output (batch x features), bias per feature."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    chan = offsets % n_channels
    bias = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    result = x + bias
    result = tl.maximum(result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
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
    """Bias add without ReLU (for final linear layer)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    chan = offsets % n_channels
    bias = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    result = x + bias
    tl.store(out_ptr + offsets, result, mask=mask)


def conv_bias_relu(x, weight, bias, padding=1):
    """Perform conv2d then fuse bias+relu with Triton."""
    # Use PyTorch's conv2d (cuDNN) without bias
    out = torch.nn.functional.conv2d(x, weight, bias=None, padding=padding)
    # out shape: (N, C, H, W)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    # NCHW: channel stride = H*W
    channel_stride = H * W
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_relu_kernel[grid](
        out, bias, result,
        n_elements, C, channel_stride,
    )
    return result


def max_pool2d(x, kernel_size=2, stride=2):
    return torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride)


def linear_bias_relu(x, weight, bias):
    """Linear layer with bias+relu fused."""
    # x: (batch, in_features), weight: (out_features, in_features)
    out = torch.mm(x, weight.t())
    n_elements = out.numel()
    n_channels = out.shape[1]
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_relu_linear_kernel[grid](
        out, bias, result,
        n_elements, n_channels,
    )
    return result


def linear_bias(x, weight, bias):
    """Linear layer with bias (no relu)."""
    out = torch.mm(x, weight.t())
    n_elements = out.numel()
    n_channels = out.shape[1]
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_add_kernel[grid](
        out, bias, result,
        n_elements, n_channels,
    )
    return result


def kernel_function(
    p_features_0_weight, p_features_0_bias,
    p_features_2_weight, p_features_2_bias,
    p_features_5_weight, p_features_5_bias,
    p_features_7_weight, p_features_7_bias,
    p_features_10_weight, p_features_10_bias,
    p_features_12_weight, p_features_12_bias,
    p_features_14_weight, p_features_14_bias,
    p_features_16_weight, p_features_16_bias,
    p_features_19_weight, p_features_19_bias,
    p_features_21_weight, p_features_21_bias,
    p_features_23_weight, p_features_23_bias,
    p_features_25_weight, p_features_25_bias,
    p_features_28_weight, p_features_28_bias,
    p_features_30_weight, p_features_30_bias,
    p_features_32_weight, p_features_32_bias,
    p_features_34_weight, p_features_34_bias,
    p_classifier_0_weight, p_classifier_0_bias,
    p_classifier_3_weight, p_classifier_3_bias,
    p_classifier_6_weight, p_classifier_6_bias,
    x,
):
    # Block 1
    x = conv_bias_relu(x, p_features_0_weight, p_features_0_bias, padding=1)
    x = conv_bias_relu(x, p_features_2_weight, p_features_2_bias, padding=1)
    x = max_pool2d(x)

    # Block 2
    x = conv_bias_relu(x, p_features_5_weight, p_features_5_bias, padding=1)
    x = conv_bias_relu(x, p_features_7_weight, p_features_7_bias, padding=1)
    x = max_pool2d(x)

    # Block 3
    x = conv_bias_relu(x, p_features_10_weight, p_features_10_bias, padding=1)
    x = conv_bias_relu(x, p_features_12_weight, p_features_12_bias, padding=1)
    x = conv_bias_relu(x, p_features_14_weight, p_features_14_bias, padding=1)
    x = conv_bias_relu(x, p_features_16_weight, p_features_16_bias, padding=1)
    x = max_pool2d(x)

    # Block 4
    x = conv_bias_relu(x, p_features_19_weight, p_features_19_bias, padding=1)
    x = conv_bias_relu(x, p_features_21_weight, p_features_21_bias, padding=1)
    x = conv_bias_relu(x, p_features_23_weight, p_features_23_bias, padding=1)
    x = conv_bias_relu(x, p_features_25_weight, p_features_25_bias, padding=1)
    x = max_pool2d(x)

    # Block 5
    x = conv_bias_relu(x, p_features_28_weight, p_features_28_bias, padding=1)
    x = conv_bias_relu(x, p_features_30_weight, p_features_30_bias, padding=1)
    x = conv_bias_relu(x, p_features_32_weight, p_features_32_bias, padding=1)
    x = conv_bias_relu(x, p_features_34_weight, p_features_34_bias, padding=1)
    x = max_pool2d(x)

    # Flatten
    x = x.flatten(1)

    # Classifier
    x = linear_bias_relu(x, p_classifier_0_weight, p_classifier_0_bias)
    # dropout with p=0.0 is identity
    x = linear_bias_relu(x, p_classifier_3_weight, p_classifier_3_bias)
    # dropout with p=0.0 is identity
    x = linear_bias(x, p_classifier_6_weight, p_classifier_6_bias)

    return x


def get_inputs():
    device = 'cuda'
    dtype = torch.float32

    p_features_0_weight = torch.randn(64, 3, 3, 3, device=device, dtype=dtype)
    p_features_0_bias = torch.randn(64, device=device, dtype=dtype)
    p_features_2_weight = torch.randn(64, 64, 3, 3, device=device, dtype=dtype)
    p_features_2_bias = torch.randn(64, device=device, dtype=dtype)
    p_features_5_weight = torch.randn(128, 64, 3, 3, device=device, dtype=dtype)
    p_features_5_bias = torch.randn(128, device=device, dtype=dtype)
    p_features_7_weight = torch.randn(128, 128, 3, 3, device=device, dtype=dtype)
    p_features_7_bias = torch.randn(128, device=device, dtype=dtype)
    p_features_10_weight = torch.randn(256, 128, 3, 3, device=device, dtype=dtype)
    p_features_10_bias = torch.randn(256, device=device, dtype=dtype)
    p_features_12_weight = torch.randn(256, 256, 3, 3, device=device, dtype=dtype)
    p_features_12_bias = torch.randn(256, device=device, dtype=dtype)
    p_features_14_weight = torch.randn(256, 256, 3, 3, device=device, dtype=dtype)
    p_features_14_bias = torch.randn(256, device=device, dtype=dtype)
    p_features_16_weight = torch.randn(256, 256, 3, 3, device=device, dtype=dtype)
    p_features_16_bias = torch.randn(256, device=device, dtype=dtype)
    p_features_19_weight = torch.randn(512, 256, 3, 3, device=device, dtype=dtype)
    p_features_19_bias = torch.randn(512, device=device, dtype=dtype)
    p_features_21_weight = torch.randn(512, 512, 3, 3, device=device, dtype=dtype)
    p_features_21_bias = torch.randn(512, device=device, dtype=dtype)
    p_features_23_weight = torch.randn(512, 512, 3, 3, device=device, dtype=dtype)
    p_features_23_bias = torch.randn(512, device=device, dtype=dtype)
    p_features_25_weight = torch.randn(512, 512, 3, 3, device=device, dtype=dtype)
    p_features_25_bias = torch.randn(512, device=device, dtype=dtype)
    p_features_28_weight = torch.randn(512, 512, 3, 3, device=device, dtype=dtype)
    p_features_28_bias = torch.randn(512, device=device, dtype=dtype)
    p_features_30_weight = torch.randn(512, 512, 3, 3, device=device, dtype=dtype)
    p_features_30_bias = torch.randn(512, device=device, dtype=dtype)
    p_features_32_weight = torch.randn(512, 512, 3, 3, device=device, dtype=dtype)
    p_features_32_bias = torch.randn(512, device=device, dtype=dtype)
    p_features_34_weight = torch.randn(512, 512, 3, 3, device=device, dtype=dtype)
    p_features_34_bias = torch.randn(512, device=device, dtype=dtype)
    p_classifier_0_weight = torch.randn(4096, 25088, device=device, dtype=dtype)
    p_classifier_0_bias = torch.randn(4096, device=device, dtype=dtype)
    p_classifier_3_weight = torch.randn(4096, 4096, device=device, dtype=dtype)
    p_classifier_3_bias = torch.randn(4096, device=device, dtype=dtype)
    p_classifier_6_weight = torch.randn(1000, 4096, device=device, dtype=dtype)
    p_classifier_6_bias = torch.randn(1000, device=device, dtype=dtype)
    x = torch.randn(10, 3, 224, 224, device=device, dtype=dtype)

    return [
        p_features_0_weight, p_features_0_bias,
        p_features_2_weight, p_features_2_bias,
        p_features_5_weight, p_features_5_bias,
        p_features_7_weight, p_features_7_bias,
        p_features_10_weight, p_features_10_bias,
        p_features_12_weight, p_features_12_bias,
        p_features_14_weight, p_features_14_bias,
        p_features_16_weight, p_features_16_bias,
        p_features_19_weight, p_features_19_bias,
        p_features_21_weight, p_features_21_bias,
        p_features_23_weight, p_features_23_bias,
        p_features_25_weight, p_features_25_bias,
        p_features_28_weight, p_features_28_bias,
        p_features_30_weight, p_features_30_bias,
        p_features_32_weight, p_features_32_bias,
        p_features_34_weight, p_features_34_bias,
        p_classifier_0_weight, p_classifier_0_bias,
        p_classifier_3_weight, p_classifier_3_bias,
        p_classifier_6_weight, p_classifier_6_bias,
        x,
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

        # Build reference VGG19 model
        class VGG19(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.0),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.0),
                    nn.Linear(4096, 1000),
                )

            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        (
            p_features_0_weight, p_features_0_bias,
            p_features_2_weight, p_features_2_bias,
            p_features_5_weight, p_features_5_bias,
            p_features_7_weight, p_features_7_bias,
            p_features_10_weight, p_features_10_bias,
            p_features_12_weight, p_features_12_bias,
            p_features_14_weight, p_features_14_bias,
            p_features_16_weight, p_features_16_bias,
            p_features_19_weight, p_features_19_bias,
            p_features_21_weight, p_features_21_bias,
            p_features_23_weight, p_features_23_bias,
            p_features_25_weight, p_features_25_bias,
            p_features_28_weight, p_features_28_bias,
            p_features_30_weight, p_features_30_bias,
            p_features_32_weight, p_features_32_bias,
            p_features_34_weight, p_features_34_bias,
            p_classifier_0_weight, p_classifier_0_bias,
            p_classifier_3_weight, p_classifier_3_bias,
            p_classifier_6_weight, p_classifier_6_bias,
            x_in,
        ) = inputs

        model = VGG19().cuda().eval()
        # Load weights
        model.features[0].weight.data = p_features_0_weight
        model.features[0].bias.data = p_features_0_bias
        model.features[2].weight.data = p_features_2_weight
        model.features[2].bias.data = p_features_2_bias
        model.features[5].weight.data = p_features_5_weight
        model.features[5].bias.data = p_features_5_bias
        model.features[7].weight.data = p_features_7_weight
        model.features[7].bias.data = p_features_7_bias
        model.features[10].weight.data = p_features_10_weight
        model.features[10].bias.data = p_features_10_bias
        model.features[12].weight.data = p_features_12_weight
        model.features[12].bias.data = p_features_12_bias
        model.features[14].weight.data = p_features_14_weight
        model.features[14].bias.data = p_features_14_bias
        model.features[16].weight.data = p_features_16_weight
        model.features[16].bias.data = p_features_16_bias
        model.features[19].weight.data = p_features_19_weight
        model.features[19].bias.data = p_features_19_bias
        model.features[21].weight.data = p_features_21_weight
        model.features[21].bias.data = p_features_21_bias
        model.features[23].weight.data = p_features_23_weight
        model.features[23].bias.data = p_features_23_bias
        model.features[25].weight.data = p_features_25_weight
        model.features[25].bias.data = p_features_25_bias
        model.features[28].weight.data = p_features_28_weight
        model.features[28].bias.data = p_features_28_bias
        model.features[30].weight.data = p_features_30_weight
        model.features[30].bias.data = p_features_30_bias
        model.features[32].weight.data = p_features_32_weight
        model.features[32].bias.data = p_features_32_bias
        model.features[34].weight.data = p_features_34_weight
        model.features[34].bias.data = p_features_34_bias
        model.classifier[0].weight.data = p_classifier_0_weight
        model.classifier[0].bias.data = p_classifier_0_bias
        model.classifier[3].weight.data = p_classifier_3_weight
        model.classifier[3].bias.data = p_classifier_3_bias
        model.classifier[6].weight.data = p_classifier_6_weight
        model.classifier[6].bias.data = p_classifier_6_bias

        with torch.no_grad():
            ref = model(x_in)

        torch.cuda.synchronize()
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")