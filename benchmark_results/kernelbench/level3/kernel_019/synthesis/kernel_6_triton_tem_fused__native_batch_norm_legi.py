import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'num_channels'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, num_channels, spatial_size,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index: layout is (N, C, H, W) -> channel = (offset / spatial_size) % num_channels
    channel_idx = (offsets // spatial_size) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, tl.zeros_like(out))
    
    tl.store(out_ptr + offsets, out, mask=mask)


def apply_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial_size = H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial_size,
        eps,
    )
    return out


# Fused Linear kernel (GEMM with bias)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def apply_linear(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


def kernel_function(
    p_model_0_0_weight, p_model_0_1_weight, p_model_0_1_bias,
    p_model_1_0_weight, p_model_1_1_weight, p_model_1_1_bias,
    p_model_1_3_weight, p_model_1_4_weight, p_model_1_4_bias,
    p_model_2_0_weight, p_model_2_1_weight, p_model_2_1_bias,
    p_model_2_3_weight, p_model_2_4_weight, p_model_2_4_bias,
    p_model_3_0_weight, p_model_3_1_weight, p_model_3_1_bias,
    p_model_3_3_weight, p_model_3_4_weight, p_model_3_4_bias,
    p_model_4_0_weight, p_model_4_1_weight, p_model_4_1_bias,
    p_model_4_3_weight, p_model_4_4_weight, p_model_4_4_bias,
    p_model_5_0_weight, p_model_5_1_weight, p_model_5_1_bias,
    p_model_5_3_weight, p_model_5_4_weight, p_model_5_4_bias,
    p_model_6_0_weight, p_model_6_1_weight, p_model_6_1_bias,
    p_model_6_3_weight, p_model_6_4_weight, p_model_6_4_bias,
    p_model_7_0_weight, p_model_7_1_weight, p_model_7_1_bias,
    p_model_7_3_weight, p_model_7_4_weight, p_model_7_4_bias,
    p_model_8_0_weight, p_model_8_1_weight, p_model_8_1_bias,
    p_model_8_3_weight, p_model_8_4_weight, p_model_8_4_bias,
    p_model_9_0_weight, p_model_9_1_weight, p_model_9_1_bias,
    p_model_9_3_weight, p_model_9_4_weight, p_model_9_4_bias,
    p_model_10_0_weight, p_model_10_1_weight, p_model_10_1_bias,
    p_model_10_3_weight, p_model_10_4_weight, p_model_10_4_bias,
    p_model_11_0_weight, p_model_11_1_weight, p_model_11_1_bias,
    p_model_11_3_weight, p_model_11_4_weight, p_model_11_4_bias,
    p_model_12_0_weight, p_model_12_1_weight, p_model_12_1_bias,
    p_model_12_3_weight, p_model_12_4_weight, p_model_12_4_bias,
    p_model_13_0_weight, p_model_13_1_weight, p_model_13_1_bias,
    p_model_13_3_weight, p_model_13_4_weight, p_model_13_4_bias,
    p_fc_weight, p_fc_bias,
    b_model_0_1_running_mean, b_model_0_1_running_var,
    b_model_1_1_running_mean, b_model_1_1_running_var,
    b_model_1_4_running_mean, b_model_1_4_running_var,
    b_model_2_1_running_mean, b_model_2_1_running_var,
    b_model_2_4_running_mean, b_model_2_4_running_var,
    b_model_3_1_running_mean, b_model_3_1_running_var,
    b_model_3_4_running_mean, b_model_3_4_running_var,
    b_model_4_1_running_mean, b_model_4_1_running_var,
    b_model_4_4_running_mean, b_model_4_4_running_var,
    b_model_5_1_running_mean, b_model_5_1_running_var,
    b_model_5_4_running_mean, b_model_5_4_running_var,
    b_model_6_1_running_mean, b_model_6_1_running_var,
    b_model_6_4_running_mean, b_model_6_4_running_var,
    b_model_7_1_running_mean, b_model_7_1_running_var,
    b_model_7_4_running_mean, b_model_7_4_running_var,
    b_model_8_1_running_mean, b_model_8_1_running_var,
    b_model_8_4_running_mean, b_model_8_4_running_var,
    b_model_9_1_running_mean, b_model_9_1_running_var,
    b_model_9_4_running_mean, b_model_9_4_running_var,
    b_model_10_1_running_mean, b_model_10_1_running_var,
    b_model_10_4_running_mean, b_model_10_4_running_var,
    b_model_11_1_running_mean, b_model_11_1_running_var,
    b_model_11_4_running_mean, b_model_11_4_running_var,
    b_model_12_1_running_mean, b_model_12_1_running_var,
    b_model_12_4_running_mean, b_model_12_4_running_var,
    b_model_13_1_running_mean, b_model_13_1_running_var,
    b_model_13_4_running_mean, b_model_13_4_running_var,
    x,
):
    import torch.nn.functional as F
    
    # Layer 0: conv_bn (3->32, stride=2)
    x = F.conv2d(x, p_model_0_0_weight, None, stride=2, padding=1)
    x = apply_bn_relu(x, p_model_0_1_weight, p_model_0_1_bias,
    
    # Layer 1: conv_dw (32->64, stride=1)
    x = F.conv2d(x, p_model_1_0_weight, None, stride=1, padding=1, groups=32)
    x = apply_bn_relu(x, p_model_1_1_weight, p_model_1_1_bias,
    x = F.conv2d(x, p_model_1_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_1_4_weight, p_model_1_4_bias,
    
    # Layer 2: conv_dw (64->128, stride=2)
    x = F.conv2d(x, p_model_2_0_weight, None, stride=2, padding=1, groups=64)
    x = apply_bn_relu(x, p_model_2_1_weight, p_model_2_1_bias,
    x = F.conv2d(x, p_model_2_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_2_4_weight, p_model_2_4_bias,
    
    # Layer 3: conv_dw (128->128, stride=1)
    x = F.conv2d(x, p_model_3_0_weight, None, stride=1, padding=1, groups=128)
    x = apply_bn_relu(x, p_model_3_1_weight, p_model_3_1_bias,
    x = F.conv2d(x, p_model_3_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_3_4_weight, p_model_3_4_bias,
    
    # Layer 4: conv_dw (128->256, stride=2)
    x = F.conv2d(x, p_model_4_0_weight, None, stride=2, padding=1, groups=128)
    x = apply_bn_relu(x, p_model_4_1_weight, p_model_4_1_bias,
    x = F.conv2d(x, p_model_4_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_4_4_weight, p_model_4_4_bias,
    
    # Layer 5: conv_dw (256->256, stride=1)
    x = F.conv2d(x, p_model_5_0_weight, None, stride=1, padding=1, groups=256)
    x = apply_bn_relu(x, p_model_5_1_weight, p_model_5_1_bias,
    x = F.conv2d(x, p_model_5_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_5_4_weight, p_model_5_4_bias,
    
    # Layer 6: conv_dw (256->512, stride=2)
    x = F.conv2d(x, p_model_6_0_weight, None, stride=2, padding=1, groups=256)
    x = apply_bn_relu(x, p_model_6_1_weight, p_model_6_1_bias,
    x = F.conv2d(x, p_model_6_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_6_4_weight, p_model_6_4_bias,
    
    # Layer 7: conv_dw (512->512, stride=1)
    x = F.conv2d(x, p_model_7_0_weight, None, stride=1, padding=1, groups=512)
    x = apply_bn_relu(x, p_model_7_1_weight, p_model_7_1_bias,
    x = F.conv2d(x, p_model_7_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_7_4_weight, p_model_7_4_bias,
    
    # Layer 8: conv_dw (512->512, stride=1)
    x = F.conv2d(x, p_model_8_0_weight, None, stride=1, padding=1, groups=512)
    x = apply_bn_relu(x, p_model_8_1_weight, p_model_8_1_bias,
    x = F.conv2d(x, p_model_8_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_8_4_weight, p_model_8_4_bias,
    
    # Layer 9: conv_dw (512->512, stride=1)
    x = F.conv2d(x, p_model_9_0_weight, None, stride=1, padding=1, groups=512)
    x = apply_bn_relu(x, p_model_9_1_weight, p_model_9_1_bias,
    x = F.conv2d(x, p_model_9_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_9_4_weight, p_model_9_4_bias,
    
    # Layer 10: conv_dw (512->512, stride=1)
    x = F.conv2d(x, p_model_10_0_weight, None, stride=1, padding=1, groups=512)
    x = apply_bn_relu(x, p_model_10_1_weight, p_model_10_1_bias,
    x = F.conv2d(x, p_model_10_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_10_4_weight, p_model_10_4_bias,
    
    # Layer 11: conv_dw (512->512, stride=1)
    x = F.conv2d(x, p_model_11_0_weight, None, stride=1, padding=1, groups=512)
    x = apply_bn_relu(x, p_model_11_1_weight, p_model_11_1_bias,
    x = F.conv2d(x, p_model_11_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_11_4_weight, p_model_11_4_bias,
    
    # Layer 12: conv_dw (512->1024, stride=2)
    x = F.conv2d(x, p_model_12_0_weight, None, stride=2, padding=1, groups=512)
    x = apply_bn_relu(x, p_model_12_1_weight, p_model_12_1_bias,
    x = F.conv2d(x, p_model_12_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_12_4_weight, p_model_12_4_bias,
    
    # Layer 13: conv_dw (1024->1024, stride=1)
    x = F.conv2d(x, p_model_13_0_weight, None, stride=1, padding=1, groups=1024)
    x = apply_bn_relu(x, p_model_13_1_weight, p_model_13_1_bias,
    x = F.conv2d(x, p_model_13_3_weight, None, stride=1, padding=0)
    x = apply_bn_relu(x, p_model_13_4_weight, p_model_13_4_bias,
    
    # AvgPool2d(7)
    x = torch.nn.functional.avg_pool2d(x, 7)
    
    # View
    x = x.view(x.size(0), -1)
    
    # Linear
    x = apply_linear(x, p_fc_weight, p_fc_bias)
    
    return x


def get_inputs():
    # Create model to get proper weights
    import torch.nn as nn
    
    class Model(nn.Module):
        def __init__(self, num_classes=1000, input_channels=3, alpha=1.0):
            super(Model, self).__init__()
            
            def conv_bn(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
                )
            
            def conv_dw(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )
            
            self.model = nn.Sequential(
                conv_bn(input_channels, int(32 * alpha), 2),
                conv_dw(int(32 * alpha), int(64 * alpha), 1),
                conv_dw(int(64 * alpha), int(128 * alpha), 2),
                conv_dw(int(128 * alpha), int(128 * alpha), 1),
                conv_dw(int(128 * alpha), int(256 * alpha), 2),
                conv_dw(int(256 * alpha), int(256 * alpha), 1),
                conv_dw(int(256 * alpha), int(512 * alpha), 2),
                conv_dw(int(512 * alpha), int(512 * alpha), 1),
                conv_dw(int(512 * alpha), int(512 * alpha), 1),
                conv_dw(int(512 * alpha), int(512 * alpha), 1),
                conv_dw(int(512 * alpha), int(512 * alpha), 1),
                conv_dw(int(512 * alpha), int(512 * alpha), 1),
                conv_dw(int(512 * alpha), int(1024 * alpha), 2),
                conv_dw(int(1024 * alpha), int(1024 * alpha), 1),
                nn.AvgPool2d(7),
            )
            self.fc = nn.Linear(int(1024 * alpha), num_classes)
        
        def forward(self, x):
            x = self.model(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    model = Model().cuda().eval()
    
    inputs = [
        model.model[0][0].weight.data,   # p_model_0_0_weight
        model.model[0][1].weight.data,   # p_model_0_1_weight
        model.model[0][1].bias.data,     # p_model_0_1_bias
        model.model[1][0].weight.data,   # p_model_1_0_weight
        model.model[1][1].weight.data,   # p_model_1_1_weight
        model.model[1][1].bias.data,     # p_model_1_1_bias
        model.model[1][3].weight.data,   # p_model_1_3_weight
        model.model[1][4].weight.data,   # p_model_1_4_weight
        model.model[1][4].bias.data,     # p_model_1_4_bias
        model.model[2][0].weight.data,   # p_model_2_0_weight
        model.model[2][1].weight.data,   # p_model_2_1_weight
        model.model[2][1].bias.data,     # p_model_2_1_bias
        model.model[2][3].weight.data,   # p_model_2_3_weight
        model.model[2][4].weight.data,   # p_model_2_4_weight
        model.model[2][4].bias.data,     # p_model_2_4_bias
        model.model[3][0].weight.data,   # p_model_3_0_weight
        model.model[3][1].weight.data,   # p_model_3_1_weight
        model.model[3][1].bias.data,     # p_model_3_1_bias
        model.model[3][3].weight.data,   # p_model_3_3_weight
        model.model[3][4].weight.data,   # p_model_3_4_weight
        model.model[3][4].bias.data,     # p_model_3_4_bias
        model.model[4][0].weight.data,   # p_model_4_0_weight
        model.model[4][1].weight.data,   # p_model_4_1_weight
        model.model[4][1].bias.data,     # p_model_4_1_bias
        model.model[4][3].weight.data,   # p_model_4_3_weight
        model.model[4][4].weight.data,   # p_model_4_4_weight
        model.model[4][4].bias.data,     # p_model_4_4_bias
        model.model[5][0].weight.data,   # p_model_5_0_weight
        model.model[5][1].weight.data,   # p_model_5_1_weight
        model.model[5][1].bias.data,     # p_model_5_1_bias
        model.model[5][3].weight.data,   # p_model_5_3_weight
        model.model[5][4].weight.data,   # p_model_5_4_weight
        model.model[5][4].bias.data,     # p_model_5_4_bias
        model.model[6][0].weight.data,   # p_model_6_0_weight
        model.model[6][1].weight.data,   # p_model_6_1_weight
        model.model[6][1].bias.data,     # p_model_6_1_bias
        model.model[6][3].weight.data,   # p_model_6_3_weight
        model.model[6][4].weight.data,   # p_model_6_4_weight
        model.model[6][4].bias.data,     # p_model_6_4_bias
        model.model[7][0].weight.data,