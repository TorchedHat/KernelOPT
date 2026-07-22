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
    c = offsets % n_channels
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    # Add bias and apply ReLU
    out = x + bias
    out = tl.maximum(out, tl.zeros_like(out))
    
    tl.store(out_ptr + offsets, out, mask=mask)


def apply_bias_relu(x, bias):
    """Apply bias add + ReLU to tensor x with bias."""
    # x shape: (N, C, H, W) or any shape where last meaningful dim is channels
    # We need to broadcast bias over spatial dims
    # Flatten to (N*H*W, C) effectively - but x is (N, C, H, W)
    # The channel dim is dim 1
    N, C, H, W = x.shape
    # Reshape to (N*H*W, C) - but stored as (N, C, H, W)
    # We'll work with flattened view where channel = index % C
    # Actually for NCHW: element i has channel = (i // (H*W)) % C
    # Let's permute to (N, H, W, C) for easier indexing
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    out = torch.empty_like(x_nhwc)
    n_elements = x_nhwc.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_relu_kernel[grid](
        x_nhwc, bias, out,
        n_elements, C,
    )
    return out.permute(0, 3, 1, 2).contiguous()


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
    c = offsets % n_channels
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + bias
    
    tl.store(out_ptr + offsets, out, mask=mask)


def apply_bias(x, bias):
    """Apply bias add to tensor x."""
    N, C, H, W = x.shape
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    out = torch.empty_like(x_nhwc)
    n_elements = x_nhwc.numel()
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_add_kernel[grid](
        x_nhwc, bias, out,
        n_elements, C,
    )
    return out.permute(0, 3, 1, 2).contiguous()


def conv2d_bias_relu(x, weight, bias, stride=1, padding=0):
    """Conv2d + bias + ReLU fused."""
    # Use PyTorch for conv (cuDNN)
    out = torch.nn.functional.conv2d(x, weight, None, stride, padding)
    return apply_bias_relu(out, bias)


def conv2d_bias(x, weight, bias, stride=1, padding=0):
    """Conv2d + bias."""
    out = torch.nn.functional.conv2d(x, weight, None, stride, padding)
    return apply_bias(out, bias)


def max_pool2d(x, kernel_size=3, stride=2, padding=1):
    return torch.nn.functional.max_pool2d(x, kernel_size, stride, padding)


def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    """Inception module: 4 branches concatenated."""
    # Branch 1: 1x1 conv
    branch1 = conv2d_bias_relu(x, w_1x1, b_1x1, stride=1, padding=0)
    
    # Branch 2: 1x1 reduce + 3x3 conv
    branch2 = conv2d_bias_relu(x, w_3x3_r, b_3x3_r, stride=1, padding=0)
    branch2 = conv2d_bias_relu(branch2, w_3x3, b_3x3, stride=1, padding=1)
    
    # Branch 3: 1x1 reduce + 5x5 conv
    branch3 = conv2d_bias_relu(x, w_5x5_r, b_5x5_r, stride=1, padding=0)
    branch3 = conv2d_bias_relu(branch3, w_5x5, b_5x5, stride=1, padding=2)
    
    # Branch 4: max pool + 1x1 conv
    branch4 = max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch4 = conv2d_bias_relu(branch4, w_pool, b_pool, stride=1, padding=0)
    
    return torch.cat([branch1, branch2, branch3, branch4], dim=1)


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
def linear_kernel(
    x_ptr,
    w_ptr,
    b_ptr,
    out_ptr,
    M,
    N,
    K,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Simple linear layer using element-wise approach - but we need tl.dot for matmul."""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Each element in output is row m, col n
    m = offsets // N
    n = offsets % N
    
    # Compute dot product
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # This is too slow for large K - use torch.mm instead
    # Just add bias here
    b = tl.load(b_ptr + n, mask=mask, other=0.0)
    # We'll handle the matmul separately
    x_val = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = x_val + b
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
def matmul_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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
        
        # w is (N, K), we want w[n, k]
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x, w, acc)
    
    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]
    
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


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
    x,
):
    # Initial convolutions
    # conv1: 3->64, 7x7, stride=2, padding=3
    x = conv2d_bias_relu(x, p_conv1_weight, p_conv1_bias, stride=2, padding=3)
    # maxpool1: 3x3, stride=2, padding=1
    x = max_pool2d(x, kernel_size=3, stride=2, padding=1)
    # conv2: 64->64, 1x1
    x = conv2d_bias_relu(x, p_conv2_weight, p_conv2_bias, stride=1, padding=0)
    # conv3: 64->192, 3x3, padding=1
    x = conv2d_bias_relu(x, p_conv3_weight, p_conv3_bias, stride=1, padding=1)
    # maxpool2: 3x3, stride=2, padding=1
    x = max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
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
    x = max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
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
    x = max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
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
    
    # adaptive avg pool: output (1, 1)
    x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
    
    # flatten
    x = x.flatten(1)
    
    # dropout (inference mode - no-op)
    # x = x  # dropout is identity at inference
    
    # linear: 1024 -> 1000
    M_val = x.shape[0]
    N_val = p_fc_weight.shape[0]
    K_val = x.shape[1]
    
    out = torch.empty((M_val, N_val), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M_val, meta['BLOCK_M']),
        triton.cdiv(N_val, meta['BLOCK_N']),
    )
    matmul_bias_kernel[grid](
        x, p_fc_weight, p_fc_bias, out,
        M_val, N_val, K_val,
        x.stride(0), x.stride(1),
        p_fc_weight.stride(0), p_fc_weight.stride(1),
        out.stride(0), out.stride(1),
    )
    
    return out


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'
    
    inputs = [
        torch.randn(64, 3, 7, 7, device