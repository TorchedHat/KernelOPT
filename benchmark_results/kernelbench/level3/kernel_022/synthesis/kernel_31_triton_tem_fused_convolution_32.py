import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# Fused BN + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Fused BN + HardTanh kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_hardtanh_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    # HardTanh with min=-6, max=6 (ReLU6)
    out = tl.minimum(tl.maximum(out, -6.0), 6.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Fused BN only (no activation) kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    
    tl.store(out_ptr + offsets, out, mask=mask)


# Triton GEMM kernel for the final linear layer
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
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
        
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def apply_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
    )
    return out


def apply_bn_hardtanh(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_hardtanh_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
    )
    return out


def apply_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
    )
    return out


def conv2d(x, weight, stride=1, padding=0, groups=1):
    return torch.ops.aten.convolution(x, weight, None, 
                                       [stride, stride] if isinstance(stride, int) else stride,
                                       [padding, padding] if isinstance(padding, int) else padding,


def mbconv(x, 
           expand_w, expand_bn_w, expand_bn_b, expand_rm, expand_rv,
           dw_w, dw_bn_w, dw_bn_b, dw_rm, dw_rv,
           proj_w, proj_bn_w, proj_bn_b, proj_rm, proj_rv,
    identity = x
    
    if expand_ratio != 1:
        x = conv2d(x, expand_w, stride=1, padding=0)
        x = apply_bn_hardtanh(x, expand_bn_w, expand_bn_b, expand_rm, expand_rv)
    
    # Depthwise conv
    ksize = dw_w.shape[2]
    pad = ksize // 2
    C = dw_w.shape[0]
    x = conv2d(x, dw_w, stride=stride, padding=pad, groups=C)
    x = apply_bn_hardtanh(x, dw_bn_w, dw_bn_b, dw_rm, dw_rv)
    
    # Project conv
    x = conv2d(x, proj_w, stride=1, padding=0)
    x = apply_bn(x, proj_bn_w, proj_bn_b, proj_rm, proj_rv)
    
    if has_skip:
        x = x + identity
    
    return x


def kernel_function(
    p_conv1_weight,
    p_bn1_weight, p_bn1_bias,
    p_blocks_0_depthwise_conv_0_weight,
    p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
    p_blocks_0_project_conv_0_weight,
    p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
    p_blocks_1_expand_conv_0_weight,
    p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
    p_blocks_1_depthwise_conv_0_weight,
    p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
    p_blocks_1_project_conv_0_weight,
    p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
    p_blocks_2_expand_conv_0_weight,
    p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
    p_blocks_2_depthwise_conv_0_weight,
    p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
    p_blocks_2_project_conv_0_weight,
    p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
    p_blocks_3_expand_conv_0_weight,
    p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
    p_blocks_3_depthwise_conv_0_weight,
    p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
    p_blocks_3_project_conv_0_weight,
    p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
    p_blocks_4_expand_conv_0_weight,
    p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
    p_blocks_4_depthwise_conv_0_weight,
    p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
    p_blocks_4_project_conv_0_weight,
    p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,
    p_blocks_5_expand_conv_0_weight,
    p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias,
    p_blocks_5_depthwise_conv_0_weight,
    p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias,
    p_blocks_5_project_conv_0_weight,
    p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_1_bias,
    p_blocks_6_expand_conv_0_weight,
    p_blocks_6_expand_conv_1_weight, p_blocks_6_expand_conv_1_bias,
    p_blocks_6_depthwise_conv_0_weight,
    p_blocks_6_depthwise_conv_1_weight, p_blocks_6_depthwise_conv_1_bias,
    p_blocks_6_project_conv_0_weight,
    p_blocks_6_project_conv_1_weight, p_blocks_6_project_conv_1_bias,
    p_blocks_7_expand_conv_0_weight,
    p_blocks_7_expand_conv_1_weight, p_blocks_7_expand_conv_1_bias,
    p_blocks_7_depthwise_conv_0_weight,
    p_blocks_7_depthwise_conv_1_weight, p_blocks_7_depthwise_conv_1_bias,
    p_blocks_7_project_conv_0_weight,
    p_blocks_7_project_conv_1_weight, p_blocks_7_project_conv_1_bias,
    p_blocks_8_expand_conv_0_weight,
    p_blocks_8_expand_conv_1_weight, p_blocks_8_expand_conv_1_bias,
    p_blocks_8_depthwise_conv_0_weight,
    p_blocks_8_depthwise_conv_1_weight, p_blocks_8_depthwise_conv_1_bias,
    p_blocks_8_project_conv_0_weight,
    p_blocks_8_project_conv_1_weight, p_blocks_8_project_conv_1_bias,
    p_blocks_9_expand_conv_0_weight,
    p_blocks_9_expand_conv_1_weight, p_blocks_9_expand_conv_1_bias,
    p_blocks_9_depthwise_conv_0_weight,
    p_blocks_9_depthwise_conv_1_weight, p_blocks_9_depthwise_conv_1_bias,
    p_blocks_9_project_conv_0_weight,
    p_blocks_9_project_conv_1_weight, p_blocks_9_project_conv_1_bias,
    p_blocks_10_expand_conv_0_weight,
    p_blocks_10_expand_conv_1_weight, p_blocks_10_expand_conv_1_bias,
    p_blocks_10_depthwise_conv_0_weight,
    p_blocks_10_depthwise_conv_1_weight, p_blocks_10_depthwise_conv_1_bias,
    p_blocks_10_project_conv_0_weight,
    p_blocks_10_project_conv_1_weight, p_blocks_10_project_conv_1_bias,
    p_blocks_11_expand_conv_0_weight,
    p_blocks_11_expand_conv_1_weight, p_blocks_11_expand_conv_1_bias,
    p_blocks_11_depthwise_conv_0_weight,
    p_blocks_11_depthwise_conv_1_weight, p_blocks_11_depthwise_conv_1_bias,
    p_blocks_11_project_conv_0_weight,
    p_blocks_11_project_conv_1_weight, p_blocks_11_project_conv_1_bias,
    p_blocks_12_expand_conv_0_weight,
    p_blocks_12_expand_conv_1_weight, p_blocks_12_expand_conv_1_bias,
    p_blocks_12_depthwise_conv_0_weight,
    p_blocks_12_depthwise_conv_1_weight, p_blocks_12_depthwise_conv_1_bias,
    p_blocks_12_project_conv_0_weight,
    p_blocks_12_project_conv_1_weight, p_blocks_12_project_conv_1_bias,
    p_conv2_weight,
    p_bn2_weight, p_bn2_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
    b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
    b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
    b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
    b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
    b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
    b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
    b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
    b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
    b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
    b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
    b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
    b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var,
    b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var,
    b_blocks_5_expand_conv_1_running_mean, b_blocks_5_expand_conv_1_running_var,
    b_blocks_5_depthwise_conv_1_running_mean, b_blocks_5_depthwise_conv_1_running_var,
    b_blocks_5_project_conv_1_running_mean, b_blocks_5_project_conv_1_running_var,
    b_blocks_6_expand_conv_1_running_mean, b_blocks_6_expand_conv_1_running_var,
    b_blocks_6_depthwise_conv_1_running_mean, b_blocks_6_depthwise_conv_1_running_var,
    b_blocks_6_project_conv_1_running_mean, b_blocks_6_project_conv_1_running_var,
    b_blocks_7_expand_conv_1_running_mean, b_blocks_7_expand_conv_1_running_var,
    b_blocks_7_depthwise_conv_1_running_mean, b_blocks_7_depthwise_conv_1_running_var,
    b_blocks_7_project_conv_1_running_mean, b_blocks_7_project_conv_1_running_var,
    b_blocks_8_expand_conv_1_running_mean, b_blocks_8_expand_conv_1_running_var,
    b_blocks_8_depthwise_conv_1_running_mean, b_blocks_8_depthwise_conv_1_running_var,
    b_blocks_8_project_conv_1_running_mean, b_blocks_8_project_conv_1_running_var,
    b_blocks_9_expand_conv_1_running_mean, b_blocks_9_expand_conv_1_running_var,
    b_blocks_9_depthwise_conv_1_running_mean, b_blocks_9_depthwise_conv_1_running_var,
    b_blocks_9_project_conv_1_running_mean, b_blocks_9_project_conv_1_running_var,
    b_blocks_10_expand_conv_1_running_mean, b_blocks_10_expand_conv_1_running_var,
    b_blocks_10_depthwise_conv_1_running_mean, b_blocks_10_depthwise_conv_1_running_var,
    b_blocks_10_project_conv_1_running_mean, b_blocks_10_project_conv_1_running_var,
    b_blocks_11_expand_conv_1_running_mean, b_blocks_11_expand_conv_1_running_var,
    b_blocks_11_depthwise_conv_1_running_mean, b_blocks_11_depthwise_conv_1_running_var,
    b_blocks_11_project_conv_1_running_mean, b_blocks_11_project_conv_1_running_var,
    b_blocks_12_expand_conv_1_running_mean, b_blocks_12_expand_conv_1_running_var,
    b_blocks_12_depthwise_conv_1_running_mean, b_blocks_12_depthwise_conv_1_running_var,
    b_blocks_12_project_conv_1_running_mean, b_blocks_12_project_conv_1_running_var,
    b_bn2_running_mean, b_bn2_running_var,
    x,
):
    # Initial conv + BN + ReLU
    x = conv2d(x, p_conv1_weight, stride=2, padding=1)
    x = apply_bn_relu(x, p_bn1_weight, p_bn1_bias, b_bn1_running_mean, b_bn1_running_var)
    
    # Block 0: MBConv1 (32->16, stride=1, expand=1, no skip since in!=out... wait)
    # Actually block 0 has expand_ratio=1, in=32, out=16, stride=1
    # No skip connection since in_channels != out_channels
    # expand_ratio=1 means no expand conv
    x = mbconv(x,
               p_blocks_0_depthwise_conv_0_weight,
               p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
               b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
               p_blocks_0_project_conv_0_weight,
               p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
               b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
               stride=1, expand_ratio=1, has_skip=False)
    
    # Block 1: MBConv6 (16->24, stride=2, expand=6)
    x = mbconv(x,
               p_blocks_1_expand_conv_0_weight,
               p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
               b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
               p_blocks_1_depthwise_conv_0_weight,
               p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
               b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
               p_blocks_1_project_conv_0_weight,
               p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
               b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
               stride=2, expand_ratio=6, has_skip=False)
    
    # Block 2: MBConv6 (24->24, stride=1, expand=6) - has skip
    x = mbconv(x,
               p_blocks_2_expand_conv_0_weight,
               p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
               b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
               p_blocks_2_depthwise_conv_0_weight,
               p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
               b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
               p_blocks_2_project_conv_0_weight,
               p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
               b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
               stride=1, expand_ratio=6, has_skip=True)
    
    # Block 3: MBConv6 (24->40, stride=2, expand=6, kernel=5)
    x = mbconv(x,
               p_blocks_3_expand_conv_0_weight,
               p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
               b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
               p_blocks_3_depthwise_conv_0_weight,