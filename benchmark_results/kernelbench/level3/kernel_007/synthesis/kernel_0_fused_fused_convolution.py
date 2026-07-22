import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def bias_relu_kernel(x_ptr, bias_ptr, out_ptr, n_elements, n_channels, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def bias_kernel(x_ptr, bias_ptr, out_ptr, n_elements, n_channels, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def linear_bias_kernel(x_ptr, bias_ptr, out_ptr, n_elements, n_channels, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    tl.store(out_ptr + offsets, out, mask=mask)


def conv2d_bias_relu(x, weight, bias, stride=1, padding=0):
    # Use cuDNN for conv, fuse bias+relu in Triton
    # conv without bias
    out = F.conv2d(x, weight, None, stride, padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_relu_kernel[grid](out, bias, result, n_elements, C)
    return result


def conv2d_bias(x, weight, bias, stride=1, padding=0):
    out = F.conv2d(x, weight, None, stride, padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_kernel[grid](out, bias, result, n_elements, C)
    return result


def inception_module(x, w1x1, b1x1, w3x3_r, b3x3_r, w3x3, b3x3,
    branch1x1 = conv2d_bias_relu(x, w1x1, b1x1, stride=1, padding=0)
    
    branch3x3 = conv2d_bias_relu(x, w3x3_r, b3x3_r, stride=1, padding=0)
    branch3x3 = conv2d_bias_relu(branch3x3, w3x3, b3x3, stride=1, padding=1)
    
    branch5x5 = conv2d_bias_relu(x, w5x5_r, b5x5_r, stride=1, padding=0)
    branch5x5 = conv2d_bias_relu(branch5x5, w5x5, b5x5, stride=1, padding=2)
    
    branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = conv2d_bias_relu(branch_pool, wpool, bpool, stride=1, padding=0)
    
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
    # conv1: 7x7, stride=2, padding=3
    x = conv2d_bias_relu(x, p_conv1_weight, p_conv1_bias, stride=2, padding=3)
    # maxpool1: 3x3, stride=2, padding=1
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    # conv2: 1x1
    x = conv2d_bias_relu(x, p_conv2_weight, p_conv2_bias, stride=1, padding=0)
    # conv3: 3x3, padding=1
    x = conv2d_bias_relu(x, p_conv3_weight, p_conv3_bias, stride=1, padding=1)
    # maxpool2: 3x3, stride=2, padding=1
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception3a
    x = inception_module(x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
    
    # inception3b
    x = inception_module(x,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
        p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
        p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
        p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
    
    # maxpool3
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception4a
    x = inception_module(x,
        p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
        p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
        p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
        p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
        p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
    
    # inception4b
    x = inception_module(x,
        p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
        p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
        p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
        p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
        p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
    
    # inception4c
    x = inception_module(x,
        p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
        p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
        p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
        p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
        p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
    
    # inception4d
    x = inception_module(x,
        p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
        p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
        p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
        p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
        p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
    
    # inception4e
    x = inception_module(x,
        p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
        p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
        p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
        p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
        p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
    
    # maxpool4
    x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception5a
    x = inception_module(x,
        p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
        p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
        p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
        p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
        p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
    
    # inception5b
    x = inception_module(x,
        p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
        p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
        p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
        p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
        p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
    
    # adaptive avg pool
    x = F.adaptive_avg_pool2d(x, (1, 1))
    # flatten
    x = x.flatten(1)
    # dropout (inference mode - no-op)
    # linear
    out = torch.mm(x, p_fc_weight.t())
    N = out.shape[0]
    n_elements = N * 1000
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    linear_bias_kernel[grid](out, p_fc_bias, result, n_elements, 1000)
    return result


def get_inputs():
    p_conv1_weight = torch.randn(64, 3, 7, 7, device='cuda')
    p_conv1_bias = torch.randn(64, device='cuda')
    p_conv2_weight = torch.randn(64, 64, 1, 1, device='cuda')
    p_conv2_bias = torch.randn(64, device='cuda')
    p_conv3_weight = torch.randn(192, 64, 3, 3, device='cuda')
    p_conv3_bias = torch.randn(192, device='cuda')
    p_inception3a_branch1x1_weight = torch.randn(64, 192, 1, 1, device='cuda')
    p_inception3a_branch1x1_bias = torch.randn(64, device='cuda')
    p_inception3a_branch3x3_0_weight = torch.randn(96, 192, 1, 1, device='cuda')
    p_inception3a_branch3x3_0_bias = torch.randn(96, device='cuda')
    p_inception3a_branch3x3_1_weight = torch.randn(128, 96, 3, 3, device='cuda')
    p_inception3a_branch3x3_1_bias = torch.randn(128, device='cuda')
    p_inception3a_branch5x5_0_weight = torch.randn(16, 192, 1, 1, device='cuda')
    p_inception3a_branch5x5_0_bias = torch.randn(16, device='cuda')
    p_inception3a_branch5x5_1_weight = torch.randn(32, 16, 5, 5, device='cuda')
    p_inception3a_branch5x5_1_bias = torch.randn(32, device='cuda')
    p_inception3a_branch_pool_1_weight = torch.randn(32, 192, 1, 1, device='cuda')
    p_inception3a_branch_pool_1_bias = torch.randn(32, device='cuda')
    p_inception3b_branch1x1_weight = torch.randn(128, 256, 1, 1, device='cuda')
    p_inception3b_branch1x1_bias = torch.randn(128, device='cuda')
    p_inception3b_branch3x3_0_weight = torch.randn(128, 256, 1, 1, device='cuda')
    p_inception3b_branch3x3_0_bias = torch.randn(128, device='cuda')
    p_inception3b_branch3x3_1_weight = torch.randn(192, 128, 3, 3, device='cuda')
    p_inception3b_branch3x3_1_bias = torch.randn(192, device='cuda')
    p_inception3b_branch5x5_0_weight = torch.randn(32, 256, 1, 1, device='cuda')
    p_inception3b_branch5x5_0_bias = torch.randn(32, device='cuda')
    p_inception3b_branch5x5_1_weight = torch.randn(96, 32, 5, 5, device='cuda')
    p_inception3b_branch5x5_1_bias = torch.randn(96, device='cuda')
    p_inception3b_branch_pool_1_weight = torch.randn(64, 256, 1, 1, device='cuda')
    p_inception3b_branch_pool_1_bias = torch.randn(64, device='cuda')
    p_inception4a_branch1x1_weight = torch.randn(192, 480, 1, 1, device='cuda')
    p_inception4a_branch1x1_bias = torch.randn(192, device='cuda')
    p_inception4a_branch3x3_0_weight = torch.randn(96, 480, 1, 1, device='cuda')
    p_inception4a_branch3x3_0_bias = torch.randn(96, device='cuda')
    p_inception4a_branch3x3_1_weight = torch.randn(208, 96, 3, 3, device='cuda')
    p_inception4a_branch3x3_1_bias = torch.randn(208, device='cuda')
    p_inception4a_branch5x5_0_weight = torch.randn(16, 480, 1, 1, device='cuda')
    p_inception4a_branch5x5_0_bias = torch.randn(16, device='cuda')
    p_inception4a_branch5x5_1_weight = torch.randn(48, 16, 5, 5, device='cuda')
    p_inception4a_branch5x5_1_bias = torch.randn(48, device='cuda')
    p_inception4a_branch_pool_1_weight = torch.randn(64, 480, 1, 1, device='cuda')
    p_inception4a_branch_pool_1_bias = torch.randn(64, device='cuda')
    p_inception4b_branch1x1_weight = torch.randn(160, 512, 1, 1, device='cuda')
    p_inception4b_branch1x1_bias = torch.randn(160, device='cuda')
    p_inception4b_branch3x3_0_weight = torch.randn(112, 512, 1, 1, device='cuda')
    p_inception4b_branch3x3_0_bias = torch.randn(112, device='cuda')
    p_inception4b_branch3x3_1_weight = torch.randn(224, 112, 3, 3, device='cuda')
    p_inception4b_branch3x3_1_bias = torch.randn(224, device='cuda')
    p_inception4b_branch5x5_0_weight = torch.randn(24, 512, 1, 1, device='cuda')
    p_inception4b_branch5x5_0_bias = torch.randn(24, device='cuda')
    p_inception4b_branch5x5_1_weight = torch.randn(64, 24, 5, 5, device='cuda')
    p_inception4b_branch5x5_1_bias = torch.randn(64, device='cuda')
    p_inception4b_branch_pool_1_weight = torch.randn(64, 512, 1, 1, device='cuda')
    p_inception4b_branch_pool_1_bias = torch.randn(64, device='cuda')
    p_inception4c_branch1x1_weight = torch.randn(128, 512, 1, 1, device='cuda')
    p_inception4c_branch1x1_bias = torch.randn(128, device='cuda')
    p_inception4c_branch3x3_0_weight = torch.randn(128, 512, 1, 1, device='cuda')
    p_inception4c_branch3x3_0_bias = torch.randn(128, device='cuda')
    p_inception4c_branch3x3_1_weight = torch.randn(256, 128, 3, 3, device='cuda')
    p_inception4c_branch3x3_1_bias = torch.randn(256, device='cuda')
    p_inception4c_branch5x5_0_weight = torch.randn(24, 512, 1, 1, device='cuda')
    p_inception4c_branch5x5_0_bias = torch.randn(24, device='cuda')
    p_inception4c_branch5x5_1_weight = torch.randn(64, 24, 5, 5, device='cuda')
    p_inception4c_branch5x5_1_bias = torch.randn(64, device='cuda')
    p_inception4c_branch_pool_1_weight = torch.randn(64, 512, 1, 1, device='cuda')
    p_inception4c_branch_pool_1_bias = torch.randn(64, device='cuda')
    p_inception4d_branch1x1_weight = torch.randn(112, 512, 1, 1, device='cuda')
    p_inception4d_branch1x1_bias = torch.randn