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
    key=['n_elements'],
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
    # bias index is channel dimension
    bias_idx = offsets % n_channels
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
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
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    bias_idx = offsets % n_channels
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    result = x + bias
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def linear_bias_kernel(
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
    bias_idx = offsets % n_channels
    bias = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0)
    result = x + bias
    tl.store(out_ptr + offsets, result, mask=mask)


def conv2d_relu(x, weight, bias, stride=1, padding=0):
    """Conv2d followed by bias add and ReLU, fused in Triton."""
    # Use PyTorch conv (cuDNN) without bias, then fuse bias+relu in Triton
    out = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_relu_kernel[grid](out, bias, result, n_elements, C)
    return result


def conv2d_bias(x, weight, bias, stride=1, padding=0):
    """Conv2d followed by bias add (no ReLU), fused in Triton."""
    out = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_add_kernel[grid](out, bias, result, n_elements, C)
    return result


def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    """Full inception module with fused bias+relu."""
    # Branch 1x1
    branch1 = conv2d_relu(x, w_1x1, b_1x1, stride=1, padding=0)
    
    # Branch 3x3
    branch3_r = conv2d_relu(x, w_3x3_r, b_3x3_r, stride=1, padding=0)
    branch3 = conv2d_relu(branch3_r, w_3x3, b_3x3, stride=1, padding=1)
    
    # Branch 5x5
    branch5_r = conv2d_relu(x, w_5x5_r, b_5x5_r, stride=1, padding=0)
    branch5 = conv2d_relu(branch5_r, w_5x5, b_5x5, stride=1, padding=2)
    
    # Branch pool
    pool = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = conv2d_relu(pool, w_pool, b_pool, stride=1, padding=0)
    
    return torch.cat([branch1, branch3, branch5, branch_pool], dim=1)


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
    # Initial layers
    # conv1: 3->64, 7x7, stride=2, padding=3
    x = conv2d_relu(x, p_conv1_weight, p_conv1_bias, stride=2, padding=3)
    # maxpool1: 3x3, stride=2, padding=1
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    # conv2: 64->64, 1x1
    x = conv2d_relu(x, p_conv2_weight, p_conv2_bias, stride=1, padding=0)
    # conv3: 64->192, 3x3, padding=1
    x = conv2d_relu(x, p_conv3_weight, p_conv3_bias, stride=1, padding=1)
    # maxpool2: 3x3, stride=2, padding=1
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception3a: in=192, out_1x1=64, reduce_3x3=96, out_3x3=128, reduce_5x5=16, out_5x5=32, pool_proj=32
    x = inception_module(x,
                         p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
                         p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
                         p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
                         p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
                         p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
    
    # inception3b: in=256, out_1x1=128, reduce_3x3=128, out_3x3=192, reduce_5x5=32, out_5x5=96, pool_proj=64
    x = inception_module(x,
                         p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
                         p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
                         p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
                         p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
                         p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
    
    # maxpool3: 3x3, stride=2, padding=1
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception4a: in=480, out_1x1=192, reduce_3x3=96, out_3x3=208, reduce_5x5=16, out_5x5=48, pool_proj=64
    x = inception_module(x,
                         p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
                         p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
                         p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
                         p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
                         p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
    
    # inception4b: in=512, out_1x1=160, reduce_3x3=112, out_3x3=224, reduce_5x5=24, out_5x5=64, pool_proj=64
    x = inception_module(x,
                         p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
                         p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
                         p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
                         p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
                         p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
    
    # inception4c: in=512, out_1x1=128, reduce_3x3=128, out_3x3=256, reduce_5x5=24, out_5x5=64, pool_proj=64
    x = inception_module(x,
                         p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
                         p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
                         p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
                         p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
                         p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
    
    # inception4d: in=512, out_1x1=112, reduce_3x3=144, out_3x3=288, reduce_5x5=32, out_5x5=64, pool_proj=64
    x = inception_module(x,
                         p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
                         p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
                         p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
                         p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
                         p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
    
    # inception4e: in=528, out_1x1=256, reduce_3x3=160, out_3x3=320, reduce_5x5=32, out_5x5=128, pool_proj=128
    x = inception_module(x,
                         p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
                         p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
                         p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
                         p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
                         p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
    
    # maxpool4: 3x3, stride=2, padding=1
    x = torch.nn.functional.max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception5a: in=832, out_1x1=256, reduce_3x3=160, out_3x3=320, reduce_5x5=32, out_5x5=128, pool_proj=128
    x = inception_module(x,
                         p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
                         p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
                         p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
                         p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
                         p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
    
    # inception5b: in=832, out_1x1=384, reduce_3x3=192, out_3x3=384, reduce_5x5=48, out_5x5=128, pool_proj=128
    x = inception_module(x,
                         p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
                         p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
                         p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
                         p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
                         p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
    
    # adaptive avg pool: output 1x1
    x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    # flatten
    x = x.flatten(1)
    # dropout (inference mode - no-op)
    # linear
    N, C = x.shape
    out = torch.mm(x, p_fc_weight.t())
    n_elements = N * 1000
    result = torch.empty(N, 1000, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    linear_bias_kernel[grid](out, p_fc_bias, result, n_elements, 1000)
    return result


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
        torch.randn(288, 144, 3, 3, device=device,