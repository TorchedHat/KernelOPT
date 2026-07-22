import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused BatchNorm + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index: layout is (N, C, H, W) -> flat index
    # channel = (offset // HW) % C
    c_idx = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Fused BatchNorm + Add + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_add_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    residual_ptr,
    out_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c_idx = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    res = tl.load(residual_ptr + offsets, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b + res
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Fused BatchNorm only (no relu, used before residual add)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c_idx = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Linear kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
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
    
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # w is (N, K), we want (K, N) for the dot
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc, allow_tf32=False)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def fused_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        numel, C, HW, eps,
    )
    return out


def fused_bn_add_relu(x, mean, var, weight, bias, residual, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_relu_kernel[grid](
        x, mean, var, weight, bias,
        residual,
        out,
        numel, C, HW, eps,
    )
    return out


def fused_bn(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, mean, var, weight, bias,
        out,
        numel, C, HW, eps,
    )
    return out


def triton_linear(x, weight, bias):
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


def conv2d(x, weight, stride=1, padding=0, groups=1):
    return F.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_layer1_0_conv1_weight, p_layer1_0_bn1_weight, p_layer1_0_bn1_bias,
    p_layer1_0_conv2_weight, p_layer1_0_bn2_weight, p_layer1_0_bn2_bias,
    p_layer1_1_conv1_weight, p_layer1_1_bn1_weight, p_layer1_1_bn1_bias,
    p_layer1_1_conv2_weight, p_layer1_1_bn2_weight, p_layer1_1_bn2_bias,
    p_layer2_0_conv1_weight, p_layer2_0_bn1_weight, p_layer2_0_bn1_bias,
    p_layer2_0_conv2_weight, p_layer2_0_bn2_weight, p_layer2_0_bn2_bias,
    p_layer2_0_downsample_0_weight, p_layer2_0_downsample_1_weight, p_layer2_0_downsample_1_bias,
    p_layer2_1_conv1_weight, p_layer2_1_bn1_weight, p_layer2_1_bn1_bias,
    p_layer2_1_conv2_weight, p_layer2_1_bn2_weight, p_layer2_1_bn2_bias,
    p_layer3_0_conv1_weight, p_layer3_0_bn1_weight, p_layer3_0_bn1_bias,
    p_layer3_0_conv2_weight, p_layer3_0_bn2_weight, p_layer3_0_bn2_bias,
    p_layer3_0_downsample_0_weight, p_layer3_0_downsample_1_weight, p_layer3_0_downsample_1_bias,
    p_layer3_1_conv1_weight, p_layer3_1_bn1_weight, p_layer3_1_bn1_bias,
    p_layer3_1_conv2_weight, p_layer3_1_bn2_weight, p_layer3_1_bn2_bias,
    p_layer4_0_conv1_weight, p_layer4_0_bn1_weight, p_layer4_0_bn1_bias,
    p_layer4_0_conv2_weight, p_layer4_0_bn2_weight, p_layer4_0_bn2_bias,
    p_layer4_0_downsample_0_weight, p_layer4_0_downsample_1_weight, p_layer4_0_downsample_1_bias,
    p_layer4_1_conv1_weight, p_layer4_1_bn1_weight, p_layer4_1_bn1_bias,
    p_layer4_1_conv2_weight, p_layer4_1_bn2_weight, p_layer4_1_bn2_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_layer1_0_bn1_running_mean, b_layer1_0_bn1_running_var,
    b_layer1_0_bn2_running_mean, b_layer1_0_bn2_running_var,
    b_layer1_1_bn1_running_mean, b_layer1_1_bn1_running_var,
    b_layer1_1_bn2_running_mean, b_layer1_1_bn2_running_var,
    b_layer2_0_bn1_running_mean, b_layer2_0_bn1_running_var,
    b_layer2_0_bn2_running_mean, b_layer2_0_bn2_running_var,
    b_layer2_0_downsample_1_running_mean, b_layer2_0_downsample_1_running_var,
    b_layer2_1_bn1_running_mean, b_layer2_1_bn1_running_var,
    b_layer2_1_bn2_running_mean, b_layer2_1_bn2_running_var,
    b_layer3_0_bn1_running_mean, b_layer3_0_bn1_running_var,
    b_layer3_0_bn2_running_mean, b_layer3_0_bn2_running_var,
    b_layer3_0_downsample_1_running_mean, b_layer3_0_downsample_1_running_var,
    b_layer3_1_bn1_running_mean, b_layer3_1_bn1_running_var,
    b_layer3_1_bn2_running_mean, b_layer3_1_bn2_running_var,
    b_layer4_0_bn1_running_mean, b_layer4_0_bn1_running_var,
    b_layer4_0_bn2_running_mean, b_layer4_0_bn2_running_var,
    b_layer4_0_downsample_1_running_mean, b_layer4_0_downsample_1_running_var,
    b_layer4_1_bn1_running_mean, b_layer4_1_bn1_running_var,
    b_layer4_1_bn2_running_mean, b_layer4_1_bn2_running_var,
    x,
):
    # Initial conv + bn + relu + maxpool
    x = conv2d(x, p_conv1_weight, stride=2, padding=3)
    x = fused_bn_relu(x, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias)
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # Layer 1, block 0 (no downsample)
    identity = x
    out = conv2d(x, p_layer1_0_conv1_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_layer1_0_bn1_running_mean, b_layer1_0_bn1_running_var, p_layer1_0_bn1_weight, p_layer1_0_bn1_bias)
    out = conv2d(out, p_layer1_0_conv2_weight, stride=1, padding=1)
    x = fused_bn_add_relu(out, b_layer1_0_bn2_running_mean, b_layer1_0_bn2_running_var, p_layer1_0_bn2_weight, p_layer1_0_bn2_bias, identity)
    
    # Layer 1, block 1 (no downsample)
    identity = x
    out = conv2d(x, p_layer1_1_conv1_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_layer1_1_bn1_running_mean, b_layer1_1_bn1_running_var, p_layer1_1_bn1_weight, p_layer1_1_bn1_bias)
    out = conv2d(out, p_layer1_1_conv2_weight, stride=1, padding=1)
    x = fused_bn_add_relu(out, b_layer1_1_bn2_running_mean, b_layer1_1_bn2_running_var, p_layer1_1_bn2_weight, p_layer1_1_bn2_bias, identity)
    
    # Layer 2, block 0 (with downsample)
    identity = conv2d(x, p_layer2_0_downsample_0_weight, stride=2, padding=0)
    identity = fused_bn(identity, b_layer2_0_downsample_1_running_mean, b_layer2_0_downsample_1_running_var, p_layer2_0_downsample_1_weight, p_layer2_0_downsample_1_bias)
    out = conv2d(x, p_layer2_0_conv1_weight, stride=2, padding=1)
    out = fused_bn_relu(out, b_layer2_0_bn1_running_mean, b_layer2_0_bn1_running_var, p_layer2_0_bn1_weight, p_layer2_0_bn1_bias)
    out = conv2d(out, p_layer2_0_conv2_weight, stride=1, padding=1)
    x = fused_bn_add_relu(out, b_layer2_0_bn2_running_mean, b_layer2_0_bn2_running_var, p_layer2_0_bn2_weight, p_layer2_0_bn2_bias, identity)
    
    # Layer 2, block 1 (no downsample)
    identity = x
    out = conv2d(x, p_layer2_1_conv1_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_layer2_1_bn1_running_mean, b_layer2_1_bn1_running_var, p_layer2_1_bn1_weight, p_layer2_1_bn1_bias)
    out = conv2d(out, p_layer2_1_conv2_weight, stride=1, padding=1)
    x = fused_bn_add_relu(out, b_layer2_1_bn2_running_mean, b_layer2_1_bn2_running_var, p_layer2_1_bn2_weight, p_layer2_1_bn2_bias, identity)
    
    # Layer 3, block 0 (with downsample)
    identity = conv2d(x, p_layer3_0_downsample_0_weight, stride=2, padding=0)
    identity = fused_bn(identity, b_layer3_0_downsample_1_running_mean, b_layer3_0_downsample_1_running_var, p_layer3_0_downsample_1_weight, p_layer3_0_downsample_1_bias)
    out = conv2d(x, p_layer3_0_conv1_weight, stride=2, padding=1)
    out = fused_bn_relu(out, b_layer3_0_bn1_running_mean, b_layer3_0_bn1_running_var, p_layer3_0_bn1_weight, p_layer3_0_bn1_bias)
    out = conv2d(out, p_layer3_0_conv2_weight, stride=1, padding=1)
    x = fused_bn_add_relu(out, b_layer3_0_bn2_running_mean, b_layer3_0_bn2_running_var, p_layer3_0_bn2_weight, p_layer3_0_bn2_bias, identity)
    
    # Layer 3, block 1 (no downsample)
    identity = x
    out = conv2d(x, p_layer3_1_conv1_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_layer3_1_bn1_running_mean, b_layer3_1_bn1_running_var, p_layer3_1_bn1_weight, p_layer3_1_bn1_bias)
    out = conv2d(out, p_layer3_1_conv2_weight, stride=1, padding=1)
    x = fused_bn_add_relu(out, b_layer3_1_bn2_running_mean, b_layer3_1_bn2_running_var, p_layer3_1_bn2_weight, p_layer3_1_bn2_bias, identity)
    
    # Layer 4, block 0 (with downsample)
    identity = conv2d(x, p_layer4_0_downsample_0_weight, stride=2, padding=0)
    identity = fused_bn(identity, b_layer4_0_downsample_1_running_mean, b_layer4_0_downsample_1_running_var, p_layer4_0_downsample_1_weight, p_layer4_0_downsample_1_bias)
    out = conv2d(x, p_layer4_0_conv1_weight, stride=2, padding=1)
    out = fused_bn_relu(out, b_layer4_0_bn1_running_mean, b_layer4_0_bn1_running_var, p_layer4_0_bn1_weight, p_layer4_0_bn1_bias)
    out = conv2d(out, p_layer4_0_conv2_weight, stride=1, padding=1)
    x = fused_bn_add_relu(out, b_layer4_0_bn2_running_mean, b_layer4_0_bn2_running_var, p_layer4_0_bn2_weight, p_layer4_0_bn2_bias, identity)
    
    # Layer 4, block 1 (no downsample)
    identity = x
    out = conv2d(x, p_layer4_1_conv1_weight, stride=1, padding=1)
    out = fused_bn_relu(out, b_layer4_1_bn1_running_mean, b_layer4_1_bn1_running_var, p_layer4_1_bn1_weight, p_layer4_1_bn1_bias)
    out = conv2d(out, p_layer4_1_conv2_weight, stride=1, padding=1)
    x = fused_bn_add_relu(out, b_layer4_1_bn2_running_mean, b_layer4_1_bn2_running_var, p_layer4_1_bn2_weight, p_layer4_1_bn2_bias, identity)
    
    # Adaptive avg pool + flatten + linear
    x = F.adaptive_avg_pool2d(x, (1, 1))
    x = x.flatten(1)
    x = triton_linear(x, p_fc_weight, p_fc_bias)
    
    return x


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'
    
    def make(*shape):
        return torch.randn(*shape, device=device)
    
    def make_pos(*shape):
        return torch.rand(*shape, device=device) + 0.1
    
    inputs = [
        make(64, 3, 7, 7),    # p_conv1_weight
        make(64),              # p_bn1_weight
        make(64),              # p_bn1_bias
        make(64, 64, 3, 3),   # p_layer1_0_conv1_weight
        make(64),              # p_layer1_0_bn1_weight
        make(64),              # p_layer1_0_bn1_bias
        make(64, 64, 3, 3),   # p_layer1_0_conv2_weight
        make(64),              # p_layer1_0_bn2_weight
        make(64),              # p_layer1_0_bn2_bias
        make(64, 64, 3, 3),   # p_layer1_1_conv1_weight
        make(64),              # p_layer1_1_bn1_weight
        make(64),              # p_layer1_1_bn1_bias
        make(64, 64, 3, 3),   # p_layer1_1_conv2_weight
        make(64),              # p_layer1_1_bn2_weight
        make(64),              # p_layer1_1_bn2_bias
        make(128, 64, 3, 3),  # p_layer2_0_conv1_weight
        make(128),             # p_layer2_0_bn1_weight
        make(128),             # p_layer2_0_bn1_bias
        make(128, 128, 3, 3), # p_layer2_0_conv2_weight
        make(128),             # p_layer2_0_bn2_weight
        make(128),             # p_layer2_0_bn2_bias
        make(128, 64, 1, 1),  # p_layer2_0_downsample_0_weight
        make(128),             # p_layer2_0_downsample_1_weight
        make(128),             # p_layer2_0_downsample_1_bias
        make(128, 128, 3, 3), # p_layer2_1_conv1_weight
        make(128),             # p_layer2_1_bn1_weight
        make(128),             # p_layer2_1_bn1_bias
        make(128, 128, 3, 3), # p_layer2_1_conv2_weight
        make(128),             # p_layer2_1_bn2_weight
        make(128),             # p_layer2_1_bn2_bias
        make(256, 128, 3, 3), # p_layer3_0_conv1_weight
        make(256),             # p_layer3_0_bn1_weight
        make(256),             # p_layer3_0_bn1_bias
        make(256, 256, 3, 3), # p_layer3_0_conv2_weight
        make(256),             # p_layer3_0_bn2_weight
        make(256),             # p_layer3_0_bn2_bias
        make(256, 128, 1, 1), # p_layer3_0_downsample_0_weight
        make(256),             # p_layer3_0_downsample_1_weight
        make(256),             # p_layer3_0_downsample_1_bias
        make(256, 256, 3, 3), # p_layer3_1_conv1_weight
        make(256),             # p_layer3_1_bn1_weight
        make(256),             # p_layer3_1_bn1_bias
        make(256, 256, 3, 3), # p_layer3_1_conv2_weight
        make(256),             # p_layer3_1_bn2_weight
        make(256),             # p_layer3_1_bn2_bias
        make(512, 256, 3, 3), # p_layer4_0_conv1_weight
        make(512),             # p_layer4_0_bn1_weight
        make(512),             # p_layer4_0_bn1_bias
        make(512, 512, 3, 3), # p_layer4_0_conv2_weight
        make(512),             # p_layer4_0_bn2_weight
        make(512),             # p_