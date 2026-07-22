import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
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
    
    # Load input
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute channel index for bias
    c_idx = offsets % n_channels
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    # Add bias and apply ReLU
    out = x + bias
    out = tl.where(out > 0.0, out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def bias_kernel(
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
    c_idx = offsets % n_channels
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    out = x + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)


def conv2d_bias_relu(x, weight, bias, stride=1, padding=0):
    """Conv2d + bias + ReLU using PyTorch conv then fused Triton bias+relu"""
    # Use PyTorch for the heavy convolution
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    # Perform convolution without bias first
    out = torch.nn.functional.conv2d(x, weight, None, stride, padding)
    
    # Fuse bias + relu in Triton
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    
    # Reshape to (N, H, W, C) for channel-last access pattern
    # Actually keep NCHW but handle bias indexing
    result = torch.empty_like(out)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_relu_kernel[grid](
        out, bias, result,
        n_elements, C,
    )
    return result


def conv2d_bias(x, weight, bias, stride=1, padding=0):
    """Conv2d + bias using PyTorch conv then fused Triton bias"""
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    out = torch.nn.functional.conv2d(x, weight, None, stride, padding)
    
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    
    result = torch.empty_like(out)
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_kernel[grid](
        out, bias, result,
        n_elements, C,
    )
    return result


def inception_module(x, 
                     w_1x1, b_1x1,
                     w_3x3_reduce, b_3x3_reduce, w_3x3, b_3x3,
                     w_5x5_reduce, b_5x5_reduce, w_5x5, b_5x5,
    """Inception module forward pass"""
    # Branch 1x1
    branch1x1 = conv2d_bias_relu(x, w_1x1, b_1x1, stride=1, padding=0)
    
    # Branch 3x3
    branch3x3 = conv2d_bias_relu(x, w_3x3_reduce, b_3x3_reduce, stride=1, padding=0)
    branch3x3 = conv2d_bias_relu(branch3x3, w_3x3, b_3x3, stride=1, padding=1)
    
    # Branch 5x5
    branch5x5 = conv2d_bias_relu(x, w_5x5_reduce, b_5x5_reduce, stride=1, padding=0)
    branch5x5 = conv2d_bias_relu(branch5x5, w_5x5, b_5x5, stride=1, padding=2)
    
    # Branch pool
    branch_pool = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = conv2d_bias_relu(branch_pool, w_pool_proj, b_pool_proj, stride=1, padding=0)
    
    return torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_conv3_weight, p_conv3_bias,
    p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
    p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
    p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
    p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
    p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
    p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
    p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
    p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
    p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
    p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
    p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
    p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
    p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
    p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
    p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
    p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
    p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
    p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
    p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
    p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
    p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
    p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
    p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
    p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
    p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
    p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias,
    p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
    p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
    p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
    p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
    p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
    p_inception4d_branch_pool_1_weight, p_inception4d_branch_pool_1_bias,
    p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
    p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
    p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
    p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
    p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
    p_inception4e_branch_pool_1_weight, p_inception4e_branch_pool_1_bias,
    p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
    p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
    p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
    p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
    p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
    p_inception5a_branch_pool_1_weight, p_inception5a_branch_pool_1_bias,
    p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
    p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
    p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
    p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
    p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
    p_inception5b_branch_pool_1_weight, p_inception5b_branch_pool_1_bias,
    p_fc_weight, p_fc_bias,
):
    # Initial convolutions
    x = conv2d_bias_relu(x, p_conv1_weight, p_conv1_bias, stride=2, padding=3)
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    x = conv2d_bias_relu(x, p_conv2_weight, p_conv2_bias, stride=1, padding=0)
    x = conv2d_bias_relu(x, p_conv3_weight, p_conv3_bias, stride=1, padding=1)
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # Inception 3a
    x = inception_module(x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
    
    # Inception 3b
    x = inception_module(x,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
        p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
        p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
        p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
    
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # Inception 4a
    x = inception_module(x,
        p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
        p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
        p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
        p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
        p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
    
    # Inception 4b
    x = inception_module(x,
        p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
        p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
        p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
        p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
        p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
    
    # Inception 4c
    x = inception_module(x,
        p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
        p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
        p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
        p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
        p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
    
    # Inception 4d
    x = inception_module(x,
        p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
        p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
        p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
        p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
        p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
    
    # Inception 4e
    x = inception_module(x,
        p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
        p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
        p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
        p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
        p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
    
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # Inception 5a
    x = inception_module(x,
        p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
        p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
        p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
        p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
        p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
    
    # Inception 5b
    x = inception_module(x,
        p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
        p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
        p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
        p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
        p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
    
    # Final layers
    x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    x = torch.flatten(x, 1)
    # dropout is identity at inference
    x = torch.nn.functional.linear(x, p_fc_weight, p_fc_bias)
    
    return x


def get_inputs():
    device = 'cuda'
    dtype = torch.float32
    
    inputs = [
        torch.randn(64, 3, 7, 7, device=device, dtype=dtype),      # p_conv1_weight
        torch.randn(64, device=device, dtype=dtype),                 # p_conv1_bias
        torch.randn(64, 64, 1, 1, device=device, dtype=dtype),      # p_conv2_weight
        torch.randn(64, device=device, dtype=dtype),                 # p_conv2_bias
        torch.randn(192, 64, 3, 3, device=device, dtype=dtype),     # p_conv3_weight
        torch.randn(192, device=device, dtype=dtype),                # p_conv3_bias
        # inception3a
        torch.randn(64, 192, 1, 1, device=device, dtype=dtype),
        torch.randn(64, device=device, dtype=dtype),
        torch.randn(96, 192, 1, 1, device=device, dtype=dtype),
        torch.randn(96, device=device, dtype=dtype),
        torch.randn(128, 96, 3, 3, device=device, dtype=dtype),
        torch.randn(128, device=device, dtype=dtype),
        torch.randn(16, 192, 1, 1, device=device, dtype=dtype),
        torch.randn(16, device=device, dtype=dtype),
        torch.randn(32, 16, 5, 5, device=device, dtype=dtype),
        torch.randn(32, device=device, dtype=dtype),
        torch.randn(32, 192, 1, 1, device=device, dtype=dtype),
        torch.randn(32, device=device, dtype=dtype),
        # inception3b
        torch.randn(128, 256, 1, 1, device=device, dtype=dtype),
        torch.randn(128, device=device, dtype=dtype),
        torch.randn(128, 256, 1, 1, device=device, dtype=dtype),
        torch.randn(128, device=device, dtype=dtype),
        torch.randn(192, 128, 3, 3, device=device, dtype=dtype),
        torch.randn(192, device=device, dtype=dtype),
        torch.randn(32, 256, 1, 1, device=device, dtype=dtype),
        torch.randn(32, device=device, dtype=dtype),
        torch.randn(96, 32, 5, 5, device=device, dtype=dtype),
        torch.randn(96, device=device, dtype=dtype),
        torch.randn(64, 256, 1, 1, device=device, dtype=dtype),
        torch.randn(64, device=device, dtype=dtype),
        # inception4a
        torch.randn(192, 480, 1, 1, device=device, dtype=dtype),
        torch.randn(192, device=device, dtype=dtype),
        torch.randn(96, 480, 1, 1, device=device, dtype=dtype),
        torch.randn(96, device=device, dtype=dtype),
        torch.randn(208, 96, 3, 3, device=device, dtype=dtype),
        torch.randn(208, device=device, dtype=dtype),
        torch.randn(16, 480, 1, 1, device=device, dtype=dtype),
        torch.randn(16, device=device, dtype=dtype),
        torch.randn(48, 16, 5, 5, device=device, dtype=dtype),
        torch.randn(48, device=device, dtype=dtype),
        torch.randn(64, 480, 1, 1, device=device, dtype=dtype),
        torch.randn(64, device=device, dtype=dtype),
        # inception4b
        torch.randn(160, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(160, device=device, dtype=dtype),
        torch.randn(112, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(112, device=device, dtype=dtype),
        torch.randn(224, 112, 3, 3, device=device, dtype=dtype),
        torch.randn(224, device=device, dtype=dtype),
        torch.randn(24, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(24, device=device, dtype=dtype),
        torch.randn(64, 24, 5, 5, device=device, dtype=dtype),
        torch.randn(64, device=device, dtype=dtype),
        torch.randn(64, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(64, device=device, dtype=dtype),
        # inception4c
        torch.randn(128, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(128, device=device, dtype=dtype),
        torch.randn(128, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(128, device=device, dtype=dtype),
        torch.randn(256, 128, 3, 3, device=device, dtype=dtype),
        torch.randn(256, device=device, dtype=dtype),
        torch.randn(24, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(24, device=device, dtype=dtype),
        torch.randn(64, 24, 5, 5, device=device, dtype=dtype),
        torch.randn(64, device=device, dtype=dtype),
        torch.randn(64, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(64, device=device, dtype=dtype),
        # inception4d
        torch.randn(112, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(112, device=device, dtype=dtype),
        torch.randn(144, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(144, device=device, dtype=dtype),
        torch.randn(288, 144, 3, 3, device=device, dtype=dtype),
        torch.randn(288, device=device, dtype=dtype),
        torch.randn(32, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(32, device=device, dtype=dtype),
        torch.randn(64, 32, 5, 5, device=device, dtype=dtype),
        torch.randn(64, device=device, dtype=dtype),
        torch.randn(64, 512, 1, 1, device=device, dtype=dtype),
        torch.randn(64, device=device, dtype=dtype),
        # inception4e
        torch.randn(256, 528, 1, 1, device=device, dtype=dtype),
        torch.randn(256, device=device, dtype=dtype),
        torch.randn(160, 528, 1, 1, device=device, dtype=dtype),
        torch.randn(160, device=device, dtype=dtype),
        torch.randn(320, 160, 3, 3, device=device, dtype=dtype),
        torch.randn(320, device=device, dtype=dtype),
        torch.randn(32, 528, 1, 1, device=device, dtype=dtype),
        torch.randn(32, device=device, dtype=dtype),
        torch.randn(128, 32, 5, 5, device=device, dtype=dtype),
        torch.randn(128, device=device, dtype=dtype),
        torch.randn(128, 528, 1, 1, device=device, dtype=dtype),
        torch.randn(128, device=device, dtype=dtype),
        # inception5a
        torch.randn(256, 832, 1, 1, device=device, dtype=dtype),
        torch.randn(256, device=device, dtype=dtype),
        torch.randn(160, 832, 1, 1, device=device, dtype=dtype),
        torch.randn(160, device=device, dtype=dtype),
        torch.randn(320, 160, 3, 3, device=device, dtype=dtype),
        torch.randn(320, device=device, dtype=dtype),
        torch.randn(32, 832, 1, 1, device=device, dtype=dtype),
        torch.randn(32, device=device, dtype=dtype),
        torch.randn(128, 32, 5, 5, device=device, dtype=dtype),
        torch.randn(128, device=device, dtype=dtype),
        torch.randn(128, 832, 1, 1, device=device, dtype=dtype),
        torch.randn(128, device=device, dtype=dtype),
        # inception5b
        torch.randn(384, 832, 1, 1, device=device, dtype=dtype),
        torch.randn(384, device=device, dtype=dtype),
        torch.randn(192, 832, 1, 1, device=device, dtype=dtype),
        torch.randn(192, device=device, dtype=