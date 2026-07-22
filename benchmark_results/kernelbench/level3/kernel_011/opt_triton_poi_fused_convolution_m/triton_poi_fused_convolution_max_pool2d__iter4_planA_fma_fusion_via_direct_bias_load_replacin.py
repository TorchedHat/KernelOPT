import torch
import triton
import triton.language as tl
import torch.nn.functional as F



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
    ],
    key=['numel'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    channels,
    spatial,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Hint: offsets are contiguous and aligned to BLOCK_SIZE -> enables LDG.128
    offsets = tl.max_contiguous(tl.multiple_of(offsets, BLOCK_SIZE), BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index for bias
    # Layout: [N, C, H, W] -> channel = (offset / spatial) % channels
    # But we store in NHWC-like after conv template, so need to handle both
    # For standard NCHW: channel = (offset // spatial) % channels
    chan_idx = (offsets // spatial) % channels

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')

    # Direct indexed load: eliminates tl.gather and enables FFMA fusion
    bias = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0, eviction_policy='evict_last')

    result = tl.maximum(x + bias, 0.0)

    tl.store(out_ptr + offsets, result, mask=mask, eviction_policy='evict_first')


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
    ],
    key=['numel'],
)
@triton.jit

def bias_relu_nhwc_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    """For NHWC layout where channel is the last dimension."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # In NHWC layout, channel = offset % channels
    chan_idx = offsets % channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0)
    
    result = x + bias
    result = tl.maximum(result, 0.0)
    
    tl.store(out_ptr + offsets, result, mask=mask)


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
def bias_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    channels,
    spatial,
    BLOCK_SIZE: tl.constexpr,
):
    """Add bias without ReLU (for final linear layer)."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    chan_idx = (offsets // spatial) % channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0)
    
    result = x + bias
    
    tl.store(out_ptr + offsets, result, mask=mask)


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
def bias_relu_linear_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    """For linear layer output: add bias + relu. channels = output features."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # For linear output [N, C], channel = offset % channels
    chan_idx = offsets % channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0)
    
    result = x + bias
    result = tl.maximum(result, 0.0)
    
    tl.store(out_ptr + offsets, result, mask=mask)


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
def bias_linear_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    """For final linear layer output: add bias only. channels = output features."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    chan_idx = offsets % channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + chan_idx, mask=mask, other=0.0)
    
    result = x + bias
    
    tl.store(out_ptr + offsets, result, mask=mask)


def conv_bias_relu(inp, weight, bias, padding=1, stride=1):
    """Perform conv2d (no bias) then fuse bias+relu in Triton."""
    # Use F.conv2d without bias
    out = F.conv2d(inp, weight, bias=None, padding=padding, stride=stride)
    N, C, H, W = out.shape
    numel = N * C * H * W
    spatial = H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bias_relu_kernel[grid](out, bias, result, numel, C, spatial)
    return result


def conv_bias_no_relu(inp, weight, bias, padding=1, stride=1):
    """Perform conv2d (no bias) then fuse bias in Triton."""
    out = F.conv2d(inp, weight, bias=None, padding=padding, stride=stride)
    N, C, H, W = out.shape
    numel = N * C * H * W
    spatial = H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bias_kernel[grid](out, bias, result, numel, C, spatial)
    return result


def linear_bias_relu(inp, weight, bias):
    """Perform linear (no bias) then fuse bias+relu in Triton."""
    out = torch._C._nn.linear(inp, weight)
    N, C = out.shape
    numel = N * C
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bias_relu_linear_kernel[grid](out, bias, result, numel, C)
    return result


def linear_bias(inp, weight, bias):
    """Perform linear (no bias) then fuse bias in Triton."""
    out = torch._C._nn.linear(inp, weight)
    N, C = out.shape
    numel = N * C
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bias_linear_kernel[grid](out, bias, result, numel, C)
    return result


def kernel_function(
    p_features_0_weight, p_features_0_bias,
    p_features_2_weight, p_features_2_bias,
    p_features_5_weight, p_features_5_bias,
    p_features_7_weight, p_features_7_bias,
    p_features_10_weight, p_features_10_bias,
    p_features_12_weight, p_features_12_bias,
    p_features_14_weight, p_features_14_bias,
    p_features_17_weight, p_features_17_bias,
    p_features_19_weight, p_features_19_bias,
    p_features_21_weight, p_features_21_bias,
    p_features_24_weight, p_features_24_bias,
    p_features_26_weight, p_features_26_bias,
    p_features_28_weight, p_features_28_bias,
    p_classifier_0_weight, p_classifier_0_bias,
    p_classifier_3_weight, p_classifier_3_bias,
    p_classifier_6_weight, p_classifier_6_bias,
    inp,
):
    # Block 1
    feat = conv_bias_relu(inp, p_features_0_weight, p_features_0_bias, padding=1)
    feat = conv_bias_relu(feat, p_features_2_weight, p_features_2_bias, padding=1)
    feat = F.max_pool2d(feat, kernel_size=2, stride=2)
    
    # Block 2
    feat = conv_bias_relu(feat, p_features_5_weight, p_features_5_bias, padding=1)
    feat = conv_bias_relu(feat, p_features_7_weight, p_features_7_bias, padding=1)
    feat = F.max_pool2d(feat, kernel_size=2, stride=2)
    
    # Block 3
    feat = conv_bias_relu(feat, p_features_10_weight, p_features_10_bias, padding=1)
    feat = conv_bias_relu(feat, p_features_12_weight, p_features_12_bias, padding=1)
    feat = conv_bias_relu(feat, p_features_14_weight, p_features_14_bias, padding=1)
    feat = F.max_pool2d(feat, kernel_size=2, stride=2)
    
    # Block 4
    feat = conv_bias_relu(feat, p_features_17_weight, p_features_17_bias, padding=1)
    feat = conv_bias_relu(feat, p_features_19_weight, p_features_19_bias, padding=1)
    feat = conv_bias_relu(feat, p_features_21_weight, p_features_21_bias, padding=1)
    feat = F.max_pool2d(feat, kernel_size=2, stride=2)
    
    # Block 5
    feat = conv_bias_relu(feat, p_features_24_weight, p_features_24_bias, padding=1)
    feat = conv_bias_relu(feat, p_features_26_weight, p_features_26_bias, padding=1)
    feat = conv_bias_relu(feat, p_features_28_weight, p_features_28_bias, padding=1)
    feat = F.max_pool2d(feat, kernel_size=2, stride=2)
    
    # Flatten
    feat = feat.flatten(1)
    
    # Classifier
    feat = linear_bias_relu(feat, p_classifier_0_weight, p_classifier_0_bias)
    # dropout p=0.0 is identity
    feat = linear_bias_relu(feat, p_classifier_3_weight, p_classifier_3_bias)
    # dropout p=0.0 is identity
    feat = linear_bias(feat, p_classifier_6_weight, p_classifier_6_bias)
    
    return feat


def get_inputs():
    p_features_0_weight = torch.randn(64, 3, 3, 3, device='cuda')
    p_features_0_bias = torch.randn(64, device='cuda')
    p_features_2_weight = torch.randn(64, 64, 3, 3, device='cuda')
    p_features_2_bias = torch.randn(64, device='cuda')
    p_features_5_weight = torch.randn(128, 64, 3, 3, device='cuda')
    p_features_5_bias = torch.randn(128, device='cuda')
    p_features_7_weight = torch.randn(128, 128, 3, 3, device='cuda')
    p_features_7_bias = torch.randn(128, device='cuda')
    p_features_10_weight = torch.randn(256, 128, 3, 3, device='cuda')
    p_features_10_bias = torch.randn(256, device='cuda')
    p_features_12_weight = torch.randn(256, 256, 3, 3, device='cuda')
    p_features_12_bias = torch.randn(256, device='cuda')
    p_features_14_weight = torch.randn(256, 256, 3, 3, device='cuda')
    p_features_14_bias = torch.randn(256, device='cuda')
    p_features_17_weight = torch.randn(512, 256, 3, 3, device='cuda')
    p_features_17_bias = torch.randn(512, device='cuda')
    p_features_19_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_19_bias = torch.randn(512, device='cuda')
    p_features_21_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_21_bias = torch.randn(512, device='cuda')
    p_features_24_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_24_bias = torch.randn(512, device='cuda')
    p_features_26_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_26_bias = torch.randn(512, device='cuda')
    p_features_28_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_28_bias = torch.randn(512, device='cuda')
    p_classifier_0_weight = torch.randn(4096, 25088, device='cuda')
    p_classifier_0_bias = torch.randn(4096, device='cuda')
    p_classifier_3_weight = torch.randn(4096, 4096, device='cuda')
    p_classifier_3_bias = torch.randn(4096, device='cuda')
    p_classifier_6_weight = torch.randn(1000, 4096, device='cuda')
    p_classifier_6_bias = torch.randn(1000, device='cuda')
    x = torch.randn(10, 3, 224, 224, device='cuda')
    return [
        p_features_0_weight, p_features_0_bias,
        p_features_2_weight, p_features_2_bias,
        p_features_5_weight, p_features_5_bias,
        p_features_7_weight, p_features_7_bias,
        p_features_10_weight, p_features_10_bias,
        p_features_12_weight, p_features_12_bias,
        p_features_14_weight, p_features_14_bias,
        p_features_17_weight, p_features_17_bias,
        p_features_19_weight, p_features_19_bias,
        p_features_21_weight, p_features_21_bias,
        p_features_24_weight, p_features_24_bias,
        p_features_26_weight, p_features_26_bias,
        p_features_28_weight, p_features_28_bias,
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
        
        # Build reference model
        class VGG16(nn.Module):
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
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
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
        
        model = VGG16().cuda().eval()
        
        # Load weights from inputs
        (p_features_0_weight, p_features_0_bias,
         p_features_2_weight, p_features_2_bias,
         p_features_5_weight, p_features_5_bias,
         p_features_7_weight, p_features_7_bias,
         p_features_10_weight, p_features_10_bias,
         p_features_12_weight, p_features_12_bias,
         p_features_14_weight, p_features_14_bias,
         p_features_17_weight, p_features_17_bias,
         p_features_19_weight, p_features_19_bias,
         p_features_21_weight, p_features_21_bias,
         p_features_24_weight, p_features_24_bias,
         p_features_26_weight, p_features_26_bias,
         p_features_28_weight, p_features_28_bias,
         p_classifier_0_weight, p_classifier_0_bias,
         p_classifier_3_weight, p_classifier_3_bias,
         p_classifier_6_weight, p_classifier_6_bias,
         x_inp) = inputs
        
        with torch.no_grad():
            model.features[0].weight.copy_(p_features_0_weight)
            model.features[0].bias.copy_(p_features_0_bias)
            model.features[2].weight.copy_(p_features_2_weight)
            model.features[2].bias.copy_(p_features_2_bias)
            model.features[5].weight.copy_(p_features_5_weight)
            model.features[5].bias.copy_(p_features_5_bias)
            model.features[7].weight.copy_(p_features_7_weight)
            model.features[7].bias.copy_(p_features_7_bias)
            model.features[10].weight.copy_(p_features_10_weight)
            model.features[10].bias.copy_(p_features_10_bias)
            model.features[12].weight.copy_(p_features_12_weight)
            model.features[12].bias.copy_(p_features_12_bias)
            model.features[14].weight.copy_(p_features_14_weight)
            model.features[14].bias.copy_(p_features_14_bias)
            model.features[17].weight.copy_(p_features_17_weight)
            model.features[17].bias.copy_(p_features_17_bias)
            model.features[19].weight.copy_(p_features_19_weight)
            model.features[19].bias.copy_(p_features_19_bias)
            model.features[21].weight.copy_(p_features_21_weight)
            model.features[21].bias.copy_(p_features_21_bias)
            model.features[24].weight.copy_(p_features_24_weight)
            model.features[24].bias.copy_(p_features_24_bias)
            model.features[26].weight.copy_(p_features_26_weight)
            model.features[26].bias.copy_(p_features_26_bias)
            model.features[28].weight.copy_(p_features_28_weight)
            model.features[28].bias.copy_(p_features_28_bias)
            model.classifier[0].weight.copy_(p_classifier_0_weight)
            model.classifier[0].bias.copy_(p_classifier_0_bias)
            model.classifier[3].weight.copy_(p_classifier_3_weight)
            model.classifier[3].bias.copy_(p_classifier_3_bias)
            model.classifier[6].weight.copy_(p_classifier_6_weight)
            model.classifier[6].bias.copy_(p_classifier_6_bias)
            
            ref = model(x_inp)
        
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")