import torch
import triton
import triton.language as tl
import math

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
    
    # Determine channel index for each element
    # Layout: (N, C, H, W) -> channel = (offset / spatial_size) % num_channels
    spatial_idx = offsets % spatial_size
    channel_idx = (offsets // spatial_size) % num_channels
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + channel_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + channel_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + channel_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + channel_idx, mask=mask, other=0.0)
    
    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    # ReLU
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
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
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    spatial_size = H * W
    out = torch.empty_like(x)
    
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, C, spatial_size,
        eps,
    )
    return out


def avg_pool2d_triton(x, kernel_size):
    # Use PyTorch for avg_pool2d - it's a simple reduction
    import torch.nn.functional as F
    return F.avg_pool2d(x, kernel_size)


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
    x = bn_relu(x, b_model_0_1_running_mean, b_model_0_1_running_var, p_model_0_1_weight, p_model_0_1_bias)
    
    # Layer 1: conv_dw (32->32 dw, then 32->64 pw)
    x = F.conv2d(x, p_model_1_0_weight, None, stride=1, padding=1, groups=32)
    x = bn_relu(x, b_model_1_1_running_mean, b_model_1_1_running_var, p_model_1_1_weight, p_model_1_1_bias)
    x = F.conv2d(x, p_model_1_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_1_4_running_mean, b_model_1_4_running_var, p_model_1_4_weight, p_model_1_4_bias)
    
    # Layer 2: conv_dw (64->64 dw stride=2, then 64->128 pw)
    x = F.conv2d(x, p_model_2_0_weight, None, stride=2, padding=1, groups=64)
    x = bn_relu(x, b_model_2_1_running_mean, b_model_2_1_running_var, p_model_2_1_weight, p_model_2_1_bias)
    x = F.conv2d(x, p_model_2_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_2_4_running_mean, b_model_2_4_running_var, p_model_2_4_weight, p_model_2_4_bias)
    
    # Layer 3: conv_dw (128->128 dw, then 128->128 pw)
    x = F.conv2d(x, p_model_3_0_weight, None, stride=1, padding=1, groups=128)
    x = bn_relu(x, b_model_3_1_running_mean, b_model_3_1_running_var, p_model_3_1_weight, p_model_3_1_bias)
    x = F.conv2d(x, p_model_3_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_3_4_running_mean, b_model_3_4_running_var, p_model_3_4_weight, p_model_3_4_bias)
    
    # Layer 4: conv_dw (128->128 dw stride=2, then 128->256 pw)
    x = F.conv2d(x, p_model_4_0_weight, None, stride=2, padding=1, groups=128)
    x = bn_relu(x, b_model_4_1_running_mean, b_model_4_1_running_var, p_model_4_1_weight, p_model_4_1_bias)
    x = F.conv2d(x, p_model_4_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_4_4_running_mean, b_model_4_4_running_var, p_model_4_4_weight, p_model_4_4_bias)
    
    # Layer 5: conv_dw (256->256 dw, then 256->256 pw)
    x = F.conv2d(x, p_model_5_0_weight, None, stride=1, padding=1, groups=256)
    x = bn_relu(x, b_model_5_1_running_mean, b_model_5_1_running_var, p_model_5_1_weight, p_model_5_1_bias)
    x = F.conv2d(x, p_model_5_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_5_4_running_mean, b_model_5_4_running_var, p_model_5_4_weight, p_model_5_4_bias)
    
    # Layer 6: conv_dw (256->256 dw stride=2, then 256->512 pw)
    x = F.conv2d(x, p_model_6_0_weight, None, stride=2, padding=1, groups=256)
    x = bn_relu(x, b_model_6_1_running_mean, b_model_6_1_running_var, p_model_6_1_weight, p_model_6_1_bias)
    x = F.conv2d(x, p_model_6_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_6_4_running_mean, b_model_6_4_running_var, p_model_6_4_weight, p_model_6_4_bias)
    
    # Layer 7: conv_dw (512->512 dw, then 512->512 pw)
    x = F.conv2d(x, p_model_7_0_weight, None, stride=1, padding=1, groups=512)
    x = bn_relu(x, b_model_7_1_running_mean, b_model_7_1_running_var, p_model_7_1_weight, p_model_7_1_bias)
    x = F.conv2d(x, p_model_7_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_7_4_running_mean, b_model_7_4_running_var, p_model_7_4_weight, p_model_7_4_bias)
    
    # Layer 8: conv_dw (512->512 dw, then 512->512 pw)
    x = F.conv2d(x, p_model_8_0_weight, None, stride=1, padding=1, groups=512)
    x = bn_relu(x, b_model_8_1_running_mean, b_model_8_1_running_var, p_model_8_1_weight, p_model_8_1_bias)
    x = F.conv2d(x, p_model_8_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_8_4_running_mean, b_model_8_4_running_var, p_model_8_4_weight, p_model_8_4_bias)
    
    # Layer 9: conv_dw (512->512 dw, then 512->512 pw)
    x = F.conv2d(x, p_model_9_0_weight, None, stride=1, padding=1, groups=512)
    x = bn_relu(x, b_model_9_1_running_mean, b_model_9_1_running_var, p_model_9_1_weight, p_model_9_1_bias)
    x = F.conv2d(x, p_model_9_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_9_4_running_mean, b_model_9_4_running_var, p_model_9_4_weight, p_model_9_4_bias)
    
    # Layer 10: conv_dw (512->512 dw, then 512->512 pw)
    x = F.conv2d(x, p_model_10_0_weight, None, stride=1, padding=1, groups=512)
    x = bn_relu(x, b_model_10_1_running_mean, b_model_10_1_running_var, p_model_10_1_weight, p_model_10_1_bias)
    x = F.conv2d(x, p_model_10_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_10_4_running_mean, b_model_10_4_running_var, p_model_10_4_weight, p_model_10_4_bias)
    
    # Layer 11: conv_dw (512->512 dw, then 512->512 pw)
    x = F.conv2d(x, p_model_11_0_weight, None, stride=1, padding=1, groups=512)
    x = bn_relu(x, b_model_11_1_running_mean, b_model_11_1_running_var, p_model_11_1_weight, p_model_11_1_bias)
    x = F.conv2d(x, p_model_11_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_11_4_running_mean, b_model_11_4_running_var, p_model_11_4_weight, p_model_11_4_bias)
    
    # Layer 12: conv_dw (512->512 dw stride=2, then 512->1024 pw)
    x = F.conv2d(x, p_model_12_0_weight, None, stride=2, padding=1, groups=512)
    x = bn_relu(x, b_model_12_1_running_mean, b_model_12_1_running_var, p_model_12_1_weight, p_model_12_1_bias)
    x = F.conv2d(x, p_model_12_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_12_4_running_mean, b_model_12_4_running_var, p_model_12_4_weight, p_model_12_4_bias)
    
    # Layer 13: conv_dw (1024->1024 dw, then 1024->1024 pw)
    x = F.conv2d(x, p_model_13_0_weight, None, stride=1, padding=1, groups=1024)
    x = bn_relu(x, b_model_13_1_running_mean, b_model_13_1_running_var, p_model_13_1_weight, p_model_13_1_bias)
    x = F.conv2d(x, p_model_13_3_weight, None, stride=1, padding=0)
    x = bn_relu(x, b_model_13_4_running_mean, b_model_13_4_running_var, p_model_13_4_weight, p_model_13_4_bias)
    
    # AvgPool2d(7)
    x = avg_pool2d_triton(x, 7)
    
    # View
    x = x.view(x.size(0), -1)
    
    # Linear
    M, K = x.shape
    N = p_fc_weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, p_fc_weight, p_fc_bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_fc_weight.stride(0), p_fc_weight.stride(1),
    )
    
    return out


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'
    
    inputs = [
        torch.randn(32, 3, 3, 3, device=device),      # p_model_0_0_weight
        torch.ones(32, device=device),                  # p_model_0_1_weight
        torch.zeros(32, device=device),                 # p_model_0_1_bias
        torch.randn(32, 1, 3, 3, device=device),       # p_model_1_0_weight
        torch.ones(32, device=device),                  # p_model_1_1_weight
        torch.zeros(32, device=device),                 # p_model_1_1_bias
        torch.randn(64, 32, 1, 1, device=device),      # p_model_1_3_weight
        torch.ones(64, device=device),                  # p_model_1_4_weight
        torch.zeros(64, device=device),                 # p_model_1_4_bias
        torch.randn(64, 1, 3, 3, device=device),       # p_model_2_0_weight
        torch.ones(64, device=device),                  # p_model_2_1_weight
        torch.zeros(64, device=device),                 # p_model_2_1_bias
        torch.randn(128, 64, 1, 1, device=device),     # p_model_2_3_weight
        torch.ones(128, device=device),                 # p_model_2_4_weight
        torch.zeros(128, device=device),                # p_model_2_4_bias
        torch.randn(128, 1, 3, 3, device=device),      # p_model_3_0_weight
        torch.ones(128, device=device),                 # p_model_3_1_weight
        torch.zeros(128, device=device),                # p_model_3_1_bias
        torch.randn(128, 128, 1, 1, device=device),    # p_model_3_3_weight
        torch.ones(128, device=device),                 # p_model_3_4_weight
        torch.zeros(128, device=device),                # p_model_3_4_bias
        torch.randn(128, 1, 3, 3, device=device),      # p_model_4_0_weight
        torch.ones(128, device=device),                 # p_model_4_1_weight
        torch.zeros(128, device=device),                # p_model_4_1_bias
        torch.randn(256, 128, 1, 1, device=device),    # p_model_4_3_weight
        torch.ones(256, device=device),                 # p_model_4_4_weight
        torch.zeros(256, device=device),                # p_model_4_4_bias
        torch.randn(256, 1, 3, 3, device=device),      # p_model_5_0_weight
        torch.ones(256, device=device),                 # p_model_5_1_weight
        torch.zeros(256, device=device),                # p_model_5_1_bias
        torch.randn(256, 256, 1, 1, device=device),    # p_model_5_3_weight
        torch.ones(256, device=device),                 # p_model_5_4_weight
        torch.zeros(256, device=device),                # p_model_5_4_bias
        torch.randn(256, 1, 3, 3, device=device),      # p_model_6_0_weight
        torch.ones(256, device=device),                 # p_model_6_1_weight
        torch.zeros(256, device=device),                # p_model_6_1_bias
        torch.randn(512, 256, 1, 1, device=device),    # p_model_6_3_weight
        torch.ones(512, device=device),                 # p_model_6_4_weight
        torch.zeros(512, device=device),                # p_model_6_4_bias
        torch.randn(512, 1, 3, 3, device=device),      # p_model_7_0_weight
        torch.ones(512, device=device),                 # p_model_7_1_weight
        torch.zeros(512, device=device),                # p_model_7_1_bias
        torch.randn(512, 512, 1, 1, device=device),    # p_model_7_3_weight
        torch.ones(512, device=device),                 # p_model_7_4_weight
        torch.zeros(512, device=device),                # p_model_7_4_bias
        torch.randn(512, 1, 3, 3, device=device),      # p_model_8_0_weight
        torch.ones(512, device=device),                 # p_model_8_1_weight
        torch.zeros(512, device=device),                # p_model_8_1_bias
        torch.randn(512, 512, 1, 1, device=device),    # p_model_8_3_weight
        torch.ones(512, device=device),                 # p_model_8_4_weight
        torch.zeros(512, device=device),                # p_model_8_4_bias
        torch.randn(512, 1, 3, 3, device=device),      # p_model_9_0_weight
        torch.ones(512, device=device),                 # p_model_9_1_weight
        torch.zeros(512, device=device),                # p_model_9_1_bias
        torch.randn(512, 512, 1, 1, device=device),    # p_model_9_3_weight
        torch.ones(512, device=device),                 # p_model_9_4_weight
        torch.zeros(512, device=device),                # p_model_9_4_bias
        torch.randn(512, 1, 3, 3, device=device),      # p_model_10_0_weight
        torch.ones(512, device=device),                 # p_model_10_1_weight
        torch.zeros(512, device=device),                # p_model_10_1_bias
        torch.randn(512, 512, 1, 1, device=device),    # p_model_10_3_weight
        torch.ones(512, device=device),                 # p_model_10_4_weight
        torch.zeros(512, device=device),                # p_model_10_4_bias
        torch.randn(512, 1, 3, 3, device=device),      # p_model_11_0_weight
        torch.ones(512, device=device),                 # p_model_11_1_weight
        torch.zeros(512, device=device),                # p_model_11_1_bias
        torch.randn(512, 512, 1, 1, device=device),    # p_model_