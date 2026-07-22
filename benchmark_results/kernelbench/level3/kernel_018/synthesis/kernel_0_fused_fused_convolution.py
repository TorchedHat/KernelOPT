import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def fused_bias_relu_kernel(
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
    # bias index = offset % n_channels
    bias_idx = offsets % n_channels
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


def conv_bias_relu(x, weight, bias, stride=1, padding=0):
    """Perform conv2d then fuse bias+relu in Triton."""
    # Use PyTorch conv2d without bias
    out = F.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    # Fuse bias + relu in Triton
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bias_relu_kernel[grid](out, bias, result, n_elements, C)
    return result


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def fused_bias_relu_inplace_kernel(
    x_ptr,
    bias_ptr,
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
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(x_ptr + offsets, result, mask=mask)


def conv_bias_relu_inplace(x, weight, bias, stride=1, padding=0):
    """Perform conv2d then fuse bias+relu in Triton (inplace on conv output)."""
    out = F.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bias_relu_inplace_kernel[grid](out, bias, n_elements, C)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['n_elements', 'kernel_h', 'kernel_w', 'stride_h', 'stride_w', 'in_h', 'in_w', 'out_h', 'out_w', 'channels'],
)
@triton.jit
def max_pool2d_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    channels,
    in_h,
    in_w,
    out_h,
    out_w,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    pad_h,
    pad_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # offsets index into output: [N, C, out_h, out_w]
    # layout: N * C * out_h * out_w
    ow = offsets % out_w
    tmp = offsets // out_w
    oh = tmp % out_h
    tmp2 = tmp // out_h
    c = tmp2 % channels
    n = tmp2 // channels
    
    # Compute max over kernel window
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            ih = oh * stride_h + kh - pad_h
            iw = ow * stride_w + kw - pad_w
            valid = mask & (ih >= 0) & (ih < in_h) & (iw >= 0) & (iw < in_w)
            in_idx = n * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw
            val = tl.load(in_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid & (val > max_val), val, max_val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True):
    """Max pool2d implemented in Triton."""
    N, C, H, W = x.shape
    if ceil_mode:
        out_h = (H + 2 * padding - kernel_size + stride - 1) // stride + 1
        out_w = (W + 2 * padding - kernel_size + stride - 1) // stride + 1
    else:
        out_h = (H + 2 * padding - kernel_size) // stride + 1
        out_w = (W + 2 * padding - kernel_size) // stride + 1
    
    out = torch.empty(N, C, out_h, out_w, device=x.device, dtype=x.dtype)
    n_elements = N * C * out_h * out_w
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out, n_elements, C, H, W, out_h, out_w,
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['n_elements', 'channels'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    channels,
    in_h,
    in_w,
    BLOCK_SIZE: tl.constexpr,
):
    """Adaptive average pool to (1,1) - just global average pool."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # offsets index into output: [N, C, 1, 1]
    c = offsets % channels
    n = offsets // channels
    
    # Sum over all spatial positions
    total = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    hw = in_h * in_w
    
    for i in range(in_h):
        for j in range(in_w):
            in_idx = n * channels * hw + c * hw + i * in_w + j
            val = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
            total = total + val.to(tl.float32)
    
    avg = total / (in_h * in_w).to(tl.float32)
    tl.store(out_ptr + offsets, avg.to(tl.float32), mask=mask)


def triton_adaptive_avg_pool2d(x):
    """Adaptive average pool to (1,1)."""
    N, C, H, W = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    n_elements = N * C
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](x, out, n_elements, C, H, W)
    return out


def fire_module(x, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """Fire module: squeeze + expand (1x1 and 3x3) with relu, then cat."""
    # Squeeze
    x = conv_bias_relu_inplace(x, sq_w, sq_b, stride=1, padding=0)
    # Expand 1x1
    e1 = conv_bias_relu_inplace(x, ex1_w, ex1_b, stride=1, padding=0)
    # Expand 3x3
    e3 = conv_bias_relu_inplace(x, ex3_w, ex3_b, stride=1, padding=1)
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
    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    x = conv_bias_relu_inplace(x, p_features_0_weight, p_features_0_bias, stride=2, padding=0)
    
    # features[1]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True)
    
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
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True)
    
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
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True)
    
    # features[12]: FireModule(512, 64, 256, 256)
    x = fire_module(x,
                    p_features_12_squeeze_weight, p_features_12_squeeze_bias,
                    p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,
    
    # classifier[0]: Dropout(p=0.0) - identity at inference
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    x = conv_bias_relu_inplace(x, p_classifier_1_weight, p_classifier_1_bias, stride=1, padding=0)
    
    # classifier[3]: AdaptiveAvgPool2d((1,1))
    x = triton_adaptive_avg_pool2d(x)
    
    # flatten
    x = x.reshape(x.shape[0], -1)
    
    return x


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