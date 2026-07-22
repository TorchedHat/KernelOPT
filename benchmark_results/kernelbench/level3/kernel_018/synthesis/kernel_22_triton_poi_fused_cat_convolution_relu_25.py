import torch
import triton
import triton.language as tl

# Fused bias add + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
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
    
    # Compute channel index for bias
    c_idx = (offsets // 1) % n_channels  # This needs to be computed based on layout
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def fused_bias_relu_nchw_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    n_elements,
    n_channels,
    spatial_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    channel index = (flat_idx // spatial_size) % n_channels
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute channel index: in NCHW, channel = (flat_idx // (H*W)) % C
    c_idx = (offsets // spatial_size) % n_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    
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
def relu_inplace_kernel(
    x_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.where(x > 0.0, x, 0.0)
    tl.store(x_ptr + offsets, result, mask=mask)


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
def maxpool2d_ceil_kernel(
    in_ptr,
    out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Decode output index
    w_out = offsets % W_out
    h_out = (offsets // W_out) % H_out
    c = (offsets // (W_out * H_out)) % C
    n = offsets // (W_out * H_out * C)
    
    # Compute max over kernel window
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            h_in = h_out * stride_h + kh - pad_h
            w_in = w_out * stride_w + kw - pad_w
            
            valid = mask & (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in) & (n < N)
            
            in_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in
            val = tl.load(in_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


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
def adaptive_avg_pool2d_kernel(
    in_ptr,
    out_ptr,
    N, C, H_in, W_in,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    """
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # offsets indexes [N, C] output
    c = offsets % C
    n = offsets // C
    
    valid_nc = mask & (n < N) & (c < C)
    
    # Sum over all H_in * W_in
    total = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = H_in * W_in
    
    for h in range(H_in):
        for w in range(W_in):
            in_idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h * W_in + w
            val = tl.load(in_ptr + in_idx, mask=valid_nc, other=0.0)
            total = total + val
    
    avg = total / count
    tl.store(out_ptr + offsets, avg, mask=valid_nc)


def conv2d_bias_relu(x, weight, bias, stride=1, padding=0):
    """Perform conv2d then fused bias+relu."""
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    # Use PyTorch's cuDNN conv (no bias in conv, add separately)
    # Actually, pass bias=None and fuse manually
    out = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    spatial_size = H * W
    
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    
    fused_bias_relu_nchw_kernel[grid](
        out, bias, result,
        n_elements, C, spatial_size,
    )
    return result


def maxpool2d_ceil(x, kernel_size=3, stride=2, padding=0):
    """Max pool 2D with ceil mode."""
    N, C, H_in, W_in = x.shape
    import math
    H_out = math.ceil((H_in + 2 * padding - kernel_size) / stride + 1)
    W_out = math.ceil((W_in + 2 * padding - kernel_size) / stride + 1)
    
    out = torch.empty(N, C, H_out, W_out, device=x.device, dtype=x.dtype)
    n_elements = N * C * H_out * W_out
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    maxpool2d_ceil_kernel[grid](
        x, out,
        N, C, H_in, W_in, H_out, W_out,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        n_elements,
    )
    return out


def fire_module(x, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """Fire module: squeeze + expand (1x1 and 3x3)."""
    # Squeeze
    sq = conv2d_bias_relu(x, sq_w, sq_b, stride=1, padding=0)
    # Expand 1x1
    ex1 = conv2d_bias_relu(sq, ex1_w, ex1_b, stride=1, padding=0)
    # Expand 3x3
    ex3 = conv2d_bias_relu(sq, ex3_w, ex3_b, stride=1, padding=1)
    # Cat
    return torch.cat([ex1, ex3], dim=1)


def adaptive_avg_pool2d_1x1(x):
    """Adaptive average pool to (1, 1)."""
    N, C, H_in, W_in = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    n_elements = N * C
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        N, C, H_in, W_in,
        n_elements,
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
    x = conv2d_bias_relu(x, p_features_0_weight, p_features_0_bias, stride=2, padding=0)
    
    # features[1]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = maxpool2d_ceil(x, kernel_size=3, stride=2, padding=0)
    
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
    x = maxpool2d_ceil(x, kernel_size=3, stride=2, padding=0)
    
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
    x = maxpool2d_ceil(x, kernel_size=3, stride=2, padding=0)
    
    # features[12]: FireModule(512, 64, 256, 256)
    x = fire_module(x,
                    p_features_12_squeeze_weight, p_features_12_squeeze_bias,
                    p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,
    
    # classifier[0]: Dropout (p=0.0, no-op in inference)
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    x = conv2d_bias_relu(x, p_classifier_1_weight, p_classifier_1_bias, stride=1, padding=0)
    
    # classifier[3]: AdaptiveAvgPool2d((1, 1))
    x = adaptive_avg_pool2d_1x1(x)
    
    # flatten
    x = x.reshape(x.shape[0], -1)
    
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
    import math
    import torch.nn as nn
    import torch.nn.functional as F
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        # Reference implementation
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