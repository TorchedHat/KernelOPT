import torch
import triton
import triton.language as tl
import torch.nn.functional as F

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
    # bias index = offset % n_channels
    bias_idx = offsets % n_channels
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


def conv_bias_relu(x, weight, bias, stride=1, padding=0):
    """Conv2d + bias + ReLU using cuDNN conv + Triton epilogue"""
    # Use PyTorch conv (cuDNN) without bias, then fuse bias+relu in Triton
    out = F.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    n_elements = out.numel()
    n_channels = out.shape[1]
    # out is (N, C, H, W) - need to apply bias per channel
    # Reshape to apply bias efficiently
    out_flat = out.reshape(-1)
    # We need bias indexed by channel
    # out layout: (N, C, H, W) -> element i has channel = (i // (H*W)) % C
    # But flat indexing: element i -> channel = (i // (H*W)) % C
    # This is complex for arbitrary shapes, use a different approach
    # Apply bias first using broadcasting, then ReLU in Triton
    # Actually let's just use PyTorch conv with bias and then Triton ReLU
    out = F.conv2d(x, weight, bias=bias, stride=stride, padding=padding)
    n_elements = out.numel()
    out_flat = out.contiguous().view(-1)
    # Simple ReLU in Triton
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # Use a simpler pointwise ReLU
    relu_inplace_kernel[grid](out_flat, n_elements)
    return out_flat.view(out.shape)


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
def relu_inplace_kernel(x_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
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
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 8}, num_warps=8),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def max_pool2d_ceil_kernel(
    inp_ptr,
    out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    HW,
    KERNEL: tl.constexpr,
    STRIDE: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Max pool 2D with ceil mode, kernel=3, stride=2"""
    pid_n = tl.program_id(2)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(0)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    c_mask = c_offs < C
    hw_mask = hw_offs < HW

    out_h = hw_offs // OUT_W
    out_w = hw_offs % OUT_W

    # Initialize accumulator
    acc = tl.full((BLOCK_HW, BLOCK_C), float('-inf'), dtype=tl.float32)

    for ki in range(KERNEL):
        for kj in range(KERNEL):
            in_h = out_h * STRIDE + ki
            in_w = out_w * STRIDE + kj
            valid_h = (in_h >= 0) & (in_h < IN_H)
            valid_w = (in_w >= 0) & (in_w < IN_W)
            valid = valid_h[:, None] & valid_w[:, None] & hw_mask[:, None] & c_mask[None, :]

            in_idx = (pid_n * C * IN_H * IN_W +
                      c_offs[None, :] * IN_H * IN_W +
                      in_h[:, None] * IN_W +
                      in_w[:, None])
            vals = tl.load(inp_ptr + in_idx, mask=valid, other=float('-inf'))
            acc = tl.maximum(acc, vals)

    out_idx = (pid_n * C * OUT_H * OUT_W +
               c_offs[None, :] * OUT_H * OUT_W +
               out_h[:, None] * OUT_W +
               out_w[:, None])
    out_mask = hw_mask[:, None] & c_mask[None, :]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 8}, num_warps=8),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    inp_ptr,
    out_ptr,
    N, C, IN_H, IN_W,
    HW,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Adaptive avg pool 2D to (1,1)"""
    pid_n = tl.program_id(2)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(0)

    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    hw_offs = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    c_mask = c_offs < C
    hw_mask = hw_offs < HW  # HW=1 here

    total = IN_H * IN_W
    acc = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32)

    for i in range(IN_H):
        for j in range(IN_W):
            in_idx = (pid_n * C * IN_H * IN_W +
                      c_offs[None, :] * IN_H * IN_W +
            valid = hw_mask[:, None] & c_mask[None, :]
            vals = tl.load(inp_ptr + in_idx, mask=c_mask[None, :] & hw_mask[:, None], other=0.0)
            acc = acc + vals

    acc = acc / total

    out_idx = pid_n * C + c_offs[None, :]
    tl.store(out_ptr + out_idx, acc, mask=c_mask[None, :] & hw_mask[:, None])


def triton_relu(x):
    """Apply ReLU in-place using Triton"""
    n_elements = x.numel()
    x_flat = x.contiguous().view(-1)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_inplace_kernel[grid](x_flat, n_elements)
    return x_flat.view(x.shape)


def triton_max_pool2d_ceil(x, kernel_size=3, stride=2):
    """Max pool 2D with ceil mode"""
    N, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + stride - 1) // stride  # ceil mode approximation
    # Proper ceil mode: ceil((IN_H - kernel_size) / stride) + 1
    import math
    OUT_H = math.ceil((IN_H - kernel_size) / stride) + 1
    OUT_W = math.ceil((IN_W - kernel_size) / stride) + 1
    HW = OUT_H * OUT_W

    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    x_cont = x.contiguous()

    BLOCK_C = 32
    BLOCK_HW = 32
    grid = (
        triton.cdiv(HW, BLOCK_HW),
        triton.cdiv(C, BLOCK_C),
        N,
    )
    max_pool2d_ceil_kernel[grid](
        x_cont, out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        HW,
        kernel_size, stride,
    )
    return out


def triton_adaptive_avg_pool2d(x):
    """Adaptive avg pool 2D to (1,1)"""
    N, C, IN_H, IN_W = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    x_cont = x.contiguous()

    BLOCK_C = 32
    BLOCK_HW = 1
    grid = (1, triton.cdiv(C, BLOCK_C), N)
    adaptive_avg_pool2d_kernel[grid](
        x_cont, out.view(N, C),
        N, C, IN_H, IN_W,
        1,
    )
    return out


def fire_module(x, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """Fire module: squeeze + expand (1x1 and 3x3) + cat"""
    # Squeeze
    x = F.conv2d(x, sq_w, bias=sq_b, stride=1, padding=0)
    x = triton_relu(x)
    # Expand 1x1
    e1 = F.conv2d(x, ex1_w, bias=ex1_b, stride=1, padding=0)
    e1 = triton_relu(e1)
    # Expand 3x3
    e3 = F.conv2d(x, ex3_w, bias=ex3_b, stride=1, padding=1)
    e3 = triton_relu(e3)
    return torch.cat([e1, e3], dim=1)


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
    x = F.conv2d(x, p_features_0_weight, bias=p_features_0_bias, stride=2, padding=0)
    x = triton_relu(x)

    # features[2]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = triton_max_pool2d_ceil(x, kernel_size=3, stride=2)

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
    x = triton_max_pool2d_ceil(x, kernel_size=3, stride=2)

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
    x = triton_max_pool2d_ceil(x, kernel_size=3, stride=2)

    # features[12]: FireModule(512, 64, 256, 256)
    x = fire_module(x,
                    p_features_12_squeeze_weight, p_features_12_squeeze_bias,
                    p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,

    # classifier[0]: Dropout(p=0.0) - identity at inference
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    x = F.conv2d(x, p_classifier_1_weight, bias=p_classifier_1_bias, stride=1, padding=0)
    x = triton_relu(x)

    # classifier[3]: AdaptiveAvgPool2d((1,1))
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

        # Build reference model
        class FireModule(nn.Module):
            def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
                super().__init__()
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
                super().__init__()
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