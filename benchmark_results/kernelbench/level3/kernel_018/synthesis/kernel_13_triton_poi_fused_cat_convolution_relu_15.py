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
def bias_relu_kernel(
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
    
    # Compute channel index for bias lookup
    # For NCHW layout: channel = (offset / spatial_size) % n_channels
    # channel_stride = H * W (spatial size per channel)
    channel_idx = (offsets // channel_stride) % n_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    
    result = x + bias
    result = tl.maximum(result, 0.0)
    
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
def bias_add_kernel(
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
    
    channel_idx = (offsets // channel_stride) % n_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    
    result = x + bias
    
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
def maxpool2d_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Decode output index: N, C, OUT_H, OUT_W
    ow = offsets % OUT_W
    oh = (offsets // OUT_W) % OUT_H
    c = (offsets // (OUT_W * OUT_H)) % C
    n = offsets // (OUT_W * OUT_H * C)
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (n < N)
            in_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(in_ptr + in_idx, mask=valid, other=float('-inf'))
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
def adaptive_avgpool_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    ow = offsets % OUT_W
    oh = (offsets // OUT_W) % OUT_H
    c = (offsets // (OUT_W * OUT_H)) % C
    n = offsets // (OUT_W * OUT_H * C)
    
    # Compute input range for this output position
    ih_start = oh * IN_H // OUT_H
    ih_end = (oh + 1) * IN_H // OUT_H
    iw_start = ow * IN_W // OUT_W
    iw_end = (ow + 1) * IN_W // OUT_W
    
    pool_h = ih_end - ih_start
    pool_w = iw_end - iw_start
    pool_size = pool_h * pool_w
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for ih in range(IN_H):
        for iw in range(IN_W):
            in_range = (ih >= ih_start) & (ih < ih_end) & (iw >= iw_start) & (iw < iw_end)
            in_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(in_ptr + in_idx, mask=mask & in_range, other=0.0)
            acc = acc + tl.where(in_range, val, tl.zeros([BLOCK_SIZE], dtype=tl.float32))
    
    result = acc / pool_size.to(tl.float32)
    tl.store(out_ptr + offsets, result, mask=mask)


def conv2d_bias_relu(x, weight, bias, stride=1, padding=0):
    """Perform conv2d then fuse bias+relu in Triton"""
    # Use PyTorch for conv (cuDNN is faster)
    # We do conv without bias, then fuse bias+relu in Triton
    out = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    channel_stride = H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_relu_kernel[grid](out, bias, result, n_elements, C, channel_stride)
    return result


def conv2d_bias(x, weight, bias, stride=1, padding=0):
    """Perform conv2d then fuse bias in Triton"""
    out = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    channel_stride = H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_add_kernel[grid](out, bias, result, n_elements, C, channel_stride)
    return result


def triton_maxpool2d(x, kernel_size=3, stride=2, ceil_mode=True):
    """Max pooling implemented in Triton"""
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    STRIDE_H = STRIDE_W = stride
    PAD_H = PAD_W = 0
    
    if ceil_mode:
        OUT_H = (IN_H + STRIDE_H - 1) // STRIDE_H
        OUT_W = (IN_W + STRIDE_W - 1) // STRIDE_W
    else:
        OUT_H = (IN_H - KH) // STRIDE_H + 1
        OUT_W = (IN_W - KW) // STRIDE_W + 1
    
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    n_elements = N * C * OUT_H * OUT_W
    
    # Make input contiguous
    x_cont = x.contiguous()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x_cont, out, n_elements,
        N, C, IN_H, IN_W,
        OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
    )
    return out


def triton_adaptive_avgpool(x, output_size=(1, 1)):
    """Adaptive average pooling implemented in Triton"""
    N, C, IN_H, IN_W = x.shape
    OUT_H, OUT_W = output_size
    
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    n_elements = N * C * OUT_H * OUT_W
    
    x_cont = x.contiguous()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    adaptive_avgpool_kernel[grid](
        x_cont, out, n_elements,
        N, C, IN_H, IN_W,
        OUT_H, OUT_W,
    )
    return out


def fire_module(x, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """Fire module: squeeze + expand"""
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
    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    x = conv2d_bias_relu(x, p_features_0_weight, p_features_0_bias, stride=2, padding=0)
    
    # features[1]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = triton_maxpool2d(x, kernel_size=3, stride=2, ceil_mode=True)
    
    # features[3]: FireModule(96, 16, 64, 64)
    x = fire_module(
        x,
        p_features_3_squeeze_weight, p_features_3_squeeze_bias,
        p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,
        p_features_3_expand3x3_weight, p_features_3_expand3x3_bias,
    )
    
    # features[4]: FireModule(128, 16, 64, 64)
    x = fire_module(
        x,
        p_features_4_squeeze_weight, p_features_4_squeeze_bias,
        p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,
        p_features_4_expand3x3_weight, p_features_4_expand3x3_bias,
    )
    
    # features[5]: FireModule(128, 32, 128, 128)
    x = fire_module(
        x,
        p_features_5_squeeze_weight, p_features_5_squeeze_bias,
        p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,
        p_features_5_expand3x3_weight, p_features_5_expand3x3_bias,
    )
    
    # features[6]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = triton_maxpool2d(x, kernel_size=3, stride=2, ceil_mode=True)
    
    # features[7]: FireModule(256, 32, 128, 128)
    x = fire_module(
        x,
        p_features_7_squeeze_weight, p_features_7_squeeze_bias,
        p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,
        p_features_7_expand3x3_weight, p_features_7_expand3x3_bias,
    )
    
    # features[8]: FireModule(256, 48, 192, 192)
    x = fire_module(
        x,
        p_features_8_squeeze_weight, p_features_8_squeeze_bias,
        p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,
        p_features_8_expand3x3_weight, p_features_8_expand3x3_bias,
    )
    
    # features[9]: FireModule(384, 48, 192, 192)
    x = fire_module(
        x,
        p_features_9_squeeze_weight, p_features_9_squeeze_bias,
        p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,
        p_features_9_expand3x3_weight, p_features_9_expand3x3_bias,
    )
    
    # features[10]: FireModule(384, 64, 256, 256)
    x = fire_module(
        x,
        p_features_10_squeeze_weight, p_features_10_squeeze_bias,
        p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,
        p_features_10_expand3x3_weight, p_features_10_expand3x3_bias,
    )
    
    # features[11]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = triton_maxpool2d(x, kernel_size=3, stride=2, ceil_mode=True)
    
    # features[12]: FireModule(512, 64, 256, 256)
    x = fire_module(
        x,
        p_features_12_squeeze_weight, p_features_12_squeeze_bias,
        p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,
        p_features_12_expand3x3_weight, p_features_12_expand3x3_bias,
    )
    
    # classifier[0]: Dropout (p=0.0, no-op in inference)
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    x = conv2d_bias_relu(x, p_classifier_1_weight, p_classifier_1_bias, stride=1, padding=0)
    
    # classifier[3]: AdaptiveAvgPool2d((1, 1))
    x = triton_adaptive_avgpool(x, output_size=(1, 1))
    
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