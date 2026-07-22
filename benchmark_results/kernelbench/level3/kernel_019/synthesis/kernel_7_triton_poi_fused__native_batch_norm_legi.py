import torch
import triton
import triton.language as tl


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
def bn_relu_kernel(
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
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0).to(tl.float32)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    x_scaled = x_norm * w + b
    # ReLU
    result = tl.maximum(x_scaled, 0.0)
    
    tl.store(out_ptr + offsets, result.to(tl.float32), mask=mask)


# Linear kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def apply_bn_relu(x, running_mean, running_var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    numel = N * C * H * W
    spatial_size = H * W
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, spatial_size,
        eps,
    )
    return out


def apply_linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        out.stride(0), out.stride(1),
    )
    return out


def conv2d_op(x, weight, stride=1, padding=0, groups=1):
    return torch.ops.aten.convolution.default(
        x, weight, None, 
        [stride, stride] if isinstance(stride, int) else stride,
        [padding, padding] if isinstance(padding, int) else padding,
    )


def avg_pool2d_op(x, kernel_size):
    return torch.ops.aten.avg_pool2d.default(x, [kernel_size, kernel_size])


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
    # Layer 0: conv_bn (3->32, stride=2, pad=1)
    x = conv2d_op(x, p_model_0_0_weight, stride=2, padding=1)
    x = apply_bn_relu(x, b_model_0_1_running_mean, b_model_0_1_running_var, p_model_0_1_weight, p_model_0_1_bias)
    
    # Layer 1: conv_dw (32->64, stride=1)
    # DW conv: groups=32
    x = conv2d_op(x, p_model_1_0_weight, stride=1, padding=1, groups=32)
    x = apply_bn_relu(x, b_model_1_1_running_mean, b_model_1_1_running_var, p_model_1_1_weight, p_model_1_1_bias)
    # PW conv: 32->64
    x = conv2d_op(x, p_model_1_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_1_4_running_mean, b_model_1_4_running_var, p_model_1_4_weight, p_model_1_4_bias)
    
    # Layer 2: conv_dw (64->128, stride=2)
    x = conv2d_op(x, p_model_2_0_weight, stride=2, padding=1, groups=64)
    x = apply_bn_relu(x, b_model_2_1_running_mean, b_model_2_1_running_var, p_model_2_1_weight, p_model_2_1_bias)
    x = conv2d_op(x, p_model_2_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_2_4_running_mean, b_model_2_4_running_var, p_model_2_4_weight, p_model_2_4_bias)
    
    # Layer 3: conv_dw (128->128, stride=1)
    x = conv2d_op(x, p_model_3_0_weight, stride=1, padding=1, groups=128)
    x = apply_bn_relu(x, b_model_3_1_running_mean, b_model_3_1_running_var, p_model_3_1_weight, p_model_3_1_bias)
    x = conv2d_op(x, p_model_3_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_3_4_running_mean, b_model_3_4_running_var, p_model_3_4_weight, p_model_3_4_bias)
    
    # Layer 4: conv_dw (128->256, stride=2)
    x = conv2d_op(x, p_model_4_0_weight, stride=2, padding=1, groups=128)
    x = apply_bn_relu(x, b_model_4_1_running_mean, b_model_4_1_running_var, p_model_4_1_weight, p_model_4_1_bias)
    x = conv2d_op(x, p_model_4_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_4_4_running_mean, b_model_4_4_running_var, p_model_4_4_weight, p_model_4_4_bias)
    
    # Layer 5: conv_dw (256->256, stride=1)
    x = conv2d_op(x, p_model_5_0_weight, stride=1, padding=1, groups=256)
    x = apply_bn_relu(x, b_model_5_1_running_mean, b_model_5_1_running_var, p_model_5_1_weight, p_model_5_1_bias)
    x = conv2d_op(x, p_model_5_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_5_4_running_mean, b_model_5_4_running_var, p_model_5_4_weight, p_model_5_4_bias)
    
    # Layer 6: conv_dw (256->512, stride=2)
    x = conv2d_op(x, p_model_6_0_weight, stride=2, padding=1, groups=256)
    x = apply_bn_relu(x, b_model_6_1_running_mean, b_model_6_1_running_var, p_model_6_1_weight, p_model_6_1_bias)
    x = conv2d_op(x, p_model_6_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_6_4_running_mean, b_model_6_4_running_var, p_model_6_4_weight, p_model_6_4_bias)
    
    # Layer 7: conv_dw (512->512, stride=1)
    x = conv2d_op(x, p_model_7_0_weight, stride=1, padding=1, groups=512)
    x = apply_bn_relu(x, b_model_7_1_running_mean, b_model_7_1_running_var, p_model_7_1_weight, p_model_7_1_bias)
    x = conv2d_op(x, p_model_7_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_7_4_running_mean, b_model_7_4_running_var, p_model_7_4_weight, p_model_7_4_bias)
    
    # Layer 8: conv_dw (512->512, stride=1)
    x = conv2d_op(x, p_model_8_0_weight, stride=1, padding=1, groups=512)
    x = apply_bn_relu(x, b_model_8_1_running_mean, b_model_8_1_running_var, p_model_8_1_weight, p_model_8_1_bias)
    x = conv2d_op(x, p_model_8_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_8_4_running_mean, b_model_8_4_running_var, p_model_8_4_weight, p_model_8_4_bias)
    
    # Layer 9: conv_dw (512->512, stride=1)
    x = conv2d_op(x, p_model_9_0_weight, stride=1, padding=1, groups=512)
    x = apply_bn_relu(x, b_model_9_1_running_mean, b_model_9_1_running_var, p_model_9_1_weight, p_model_9_1_bias)
    x = conv2d_op(x, p_model_9_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_9_4_running_mean, b_model_9_4_running_var, p_model_9_4_weight, p_model_9_4_bias)
    
    # Layer 10: conv_dw (512->512, stride=1)
    x = conv2d_op(x, p_model_10_0_weight, stride=1, padding=1, groups=512)
    x = apply_bn_relu(x, b_model_10_1_running_mean, b_model_10_1_running_var, p_model_10_1_weight, p_model_10_1_bias)
    x = conv2d_op(x, p_model_10_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_10_4_running_mean, b_model_10_4_running_var, p_model_10_4_weight, p_model_10_4_bias)
    
    # Layer 11: conv_dw (512->512, stride=1)
    x = conv2d_op(x, p_model_11_0_weight, stride=1, padding=1, groups=512)
    x = apply_bn_relu(x, b_model_11_1_running_mean, b_model_11_1_running_var, p_model_11_1_weight, p_model_11_1_bias)
    x = conv2d_op(x, p_model_11_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_11_4_running_mean, b_model_11_4_running_var, p_model_11_4_weight, p_model_11_4_bias)
    
    # Layer 12: conv_dw (512->1024, stride=2)
    x = conv2d_op(x, p_model_12_0_weight, stride=2, padding=1, groups=512)
    x = apply_bn_relu(x, b_model_12_1_running_mean, b_model_12_1_running_var, p_model_12_1_weight, p_model_12_1_bias)
    x = conv2d_op(x, p_model_12_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_12_4_running_mean, b_model_12_4_running_var, p_model_12_4_weight, p_model_12_4_bias)
    
    # Layer 13: conv_dw (1024->1024, stride=1)
    x = conv2d_op(x, p_model_13_0_weight, stride=1, padding=1, groups=1024)
    x = apply_bn_relu(x, b_model_13_1_running_mean, b_model_13_1_running_var, p_model_13_1_weight, p_model_13_1_bias)
    x = conv2d_op(x, p_model_13_3_weight, stride=1, padding=0)
    x = apply_bn_relu(x, b_model_13_4_running_mean, b_model_13_4_running_var, p_model_13_4_weight, p_model_13_4_bias)
    
    # AvgPool2d(7)
    x = avg_pool2d_op(x, 7)
    
    # View
    x = x.reshape(x.shape[0], -1)
    
    # Linear
    x = apply_linear(x, p_fc_weight, p_fc_bias)
    
    return x


def get_inputs():
    import torch
    device = 'cuda'
    
    def make_param(*shape):
        return torch.randn(*shape, device=device)
    
    def make_bn_params(c):
        weight = torch.ones(c, device=device)
        bias = torch.zeros(c, device=device)
        running_mean = torch.zeros(c, device=device)
        running_var = torch.ones(c, device=device)
        return weight, bias, running_mean, running_var
    
    p_model_0_0_weight = make_param(32, 3, 3, 3)
    w0, b0, rm0, rv0 = make_bn_params(32)
    
    p_model_1_0_weight = make_param(32, 1, 3, 3)
    w1, b1, rm1, rv1 = make_bn_params(32)
    p_model_1_3_weight = make_param(64, 32, 1, 1)
    w1p, b1p, rm1p, rv1p = make_bn_params(64)
    
    p_model_2_0_weight = make_param(64, 1, 3, 3)
    w2, b2, rm2, rv2 = make_bn_params(64)
    p_model_2_3_weight = make_param(128, 64, 1, 1)
    w2p, b2p, rm2p, rv2p = make_bn_params(128)
    
    p_model_3_0_weight = make_param(128, 1, 3, 3)
    w3, b3, rm3, rv3 = make_bn_params(128)
    p_model_3_3_weight = make_param(128, 128, 1, 1)
    w3p, b3p, rm3p, rv3p = make_bn_params(128)
    
    p_model_4_0_weight = make_param(128, 1, 3, 3)
    w4, b4, rm4, rv4 = make_bn_params(128)
    p_model_4_3_weight = make_param(256, 128, 1, 1)
    w4p, b4p, rm4p, rv4p = make_bn_params(256)
    
    p_model_5_0_weight = make_param(256, 1, 3, 3)
    w5, b5, rm5, rv5 = make_bn_params(256)
    p_model_5_3_weight = make_param(256, 256, 1, 1)
    w5p, b5p, rm5p, rv5p = make_bn_params(256)
    
    p_model_6_0_weight = make_param(256, 1, 3, 3)
    w6, b6, rm6, rv6 = make_bn_params(256)
    p_model_6_3_weight = make_param(512, 256, 1, 1)
    w6p, b6p, rm6p, rv6p = make_bn_params(512)
    
    p_model_7_0_weight = make_param(512, 1, 3, 3)
    w7, b7, rm7, rv7 = make_bn_params(512)
    p_model_7_3_weight = make_param(512, 512, 1, 1)
    w7p, b7p, rm7p, rv7p = make_bn_params(512)
    
    p_model_8_0_weight = make_param(512, 1, 3, 3)
    w8, b8, rm8, rv8 = make_bn_params(512)
    p_model_8_3_weight = make_param(512, 512, 1, 1)
    w8p, b8p, rm8p, rv8p = make_bn_params(512)
    
    p_model_9_0_weight = make_param(512, 1, 3, 3)
    w9, b9, rm9, rv9 = make_bn_params(512)
    p_model_9_3_weight = make_param(512, 512, 1, 1)
    w9p, b9p, rm9p, rv9p = make_bn_params(512)
    
    p_model_10_0_weight = make_param(512, 1, 3, 3)
    w10, b10, rm10, rv10 = make_bn_params(512)
    p_model_10_3_weight = make_param(512, 512, 1, 1)