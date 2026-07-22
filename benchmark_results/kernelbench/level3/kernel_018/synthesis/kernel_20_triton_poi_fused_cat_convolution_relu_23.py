import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    num_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index for bias
    c_idx = (offsets % num_channels)
    
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
    key=['numel'],
)
@triton.jit
def relu_kernel(
    x_ptr,
    out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'C', 'H', 'W', 'kernel_h', 'kernel_w', 'stride_h', 'stride_w', 'out_h', 'out_w'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr,
    out_ptr,
    N, C, H, W,
    out_h, out_w,
    kernel_h, kernel_w,
    stride_h, stride_w,
    pad_h, pad_w,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Decode output indices: output is (N, C, out_h, out_w) in NCHW
    ow = offsets % out_w
    oh = (offsets // out_w) % out_h
    oc = (offsets // (out_w * out_h)) % C
    on = offsets // (out_w * out_h * C)
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(kernel_h):
        for kw in range(kernel_w):
            ih = oh * stride_h + kh - pad_h
            iw = ow * stride_w + kw - pad_w
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & (on < N)
            in_idx = on * (C * H * W) + oc * (H * W) + ih * W + iw
            val = tl.load(x_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid & (val > max_val), val, max_val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'C', 'out_h', 'out_w'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    x_ptr,
    out_ptr,
    N, C, H, W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Adaptive avg pool to (1,1) output"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # output is (N, C, 1, 1), so numel = N*C
    oc = offsets % C
    on = offsets // C
    
    # Sum over all H*W
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    total = H * W
    
    for h in range(H):
        for w in range(W):
            in_idx = on * (C * H * W) + oc * (H * W) + h * W + w
            val = tl.load(x_ptr + in_idx, mask=mask & (on < N), other=0.0)
            acc = acc + val
    
    result = acc / total.to(tl.float32)
    tl.store(out_ptr + offsets, result, mask=mask)


def conv2d_relu(x, weight, bias):
    """Conv2d followed by ReLU - use PyTorch conv then Triton relu"""
    # Use F.conv2d equivalent via torch operations
    # Determine padding from weight shape
    out = torch.nn.functional.conv2d(x, weight, bias=None, 
                                      stride=1, padding=0)
    # Add bias and relu in Triton
    N, C, H, W = out.shape
    numel = N * C * H * W
    result = torch.empty_like(out)
    
    # Reshape to (N, H, W, C) for channel-last bias access? No, keep NCHW
    # bias is indexed by channel
    # For NCHW layout: element i has channel = (i // (H*W)) % C
    # But that's expensive to compute. Let's use a different approach.
    # Actually for contiguous NCHW: index = n*C*H*W + c*H*W + h*W + w
    # channel = (index // (H*W)) % C
    # This is what the kernel does with num_channels = C and spatial = H*W
    # Actually we need: c_idx = (offsets // (H*W)) % C
    # But our kernel uses offsets % num_channels which is wrong for NCHW
    
    # Let's just do relu separately after adding bias via broadcasting
    out = out + bias.view(1, C, 1, 1)
    numel = out.numel()
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](out, result, numel)
    return result


def triton_relu(x):
    numel = x.numel()
    result = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, result, numel)
    return result


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True):
    N, C, H, W = x.shape
    if ceil_mode:
        out_h = (H + 2*padding - kernel_size + stride - 1) // stride + 1
        out_w = (W + 2*padding - kernel_size + stride - 1) // stride + 1
    else:
        out_h = (H + 2*padding - kernel_size) // stride + 1
        out_w = (W + 2*padding - kernel_size) // stride + 1
    
    out = torch.empty(N, C, out_h, out_w, device=x.device, dtype=x.dtype)
    numel = N * C * out_h * out_w
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        out_h, out_w,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        numel,
    )
    return out


def triton_adaptive_avg_pool2d_1x1(x):
    N, C, H, W = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    numel = N * C
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x, out,
        N, C, H, W,
        numel,
    )
    return out


def fire_module(x, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """Fire module: squeeze -> relu, then expand 1x1 and 3x3 -> relu -> cat"""
    import torch.nn.functional as F
    
    # Squeeze
    sq = F.conv2d(x, sq_w, bias=sq_b, stride=1, padding=0)
    sq = triton_relu(sq)
    
    # Expand 1x1
    ex1 = F.conv2d(sq, ex1_w, bias=ex1_b, stride=1, padding=0)
    ex1 = triton_relu(ex1)
    
    # Expand 3x3
    ex3 = F.conv2d(sq, ex3_w, bias=ex3_b, stride=1, padding=1)
    ex3 = triton_relu(ex3)
    
    return torch.cat([ex1, ex3], dim=1)


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
    import torch.nn.functional as F
    
    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    out = F.conv2d(x, p_features_0_weight, bias=p_features_0_bias, stride=2, padding=0)
    out = triton_relu(out)
    
    # features[2]: MaxPool2d(3, stride=2, ceil_mode=True)
    out = triton_max_pool2d(out, kernel_size=3, stride=2, padding=0, ceil_mode=True)
    
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
    out = triton_max_pool2d(out, kernel_size=3, stride=2, padding=0, ceil_mode=True)
    
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
    out = triton_max_pool2d(out, kernel_size=3, stride=2, padding=0, ceil_mode=True)
    
    # features[12]: Fire(512, 64, 256, 256)
    out = fire_module(out,
                      p_features_12_squeeze_weight, p_features_12_squeeze_bias,
                      p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,
    
    # classifier[0]: Dropout(p=0.0) - identity at inference
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    out = F.conv2d(out, p_classifier_1_weight, bias=p_classifier_1_bias, stride=1, padding=0)
    out = triton_relu(out)
    
    # classifier[3]: AdaptiveAvgPool2d((1,1))
    out = triton_adaptive_avg_pool2d_1x1(out)
    
    # flatten
    out = out.view(out.shape[0], -1)
    
    return out


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
        model = Model().cuda().eval()
        
        # Assign weights from inputs
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
        model.features[