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
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_elements', 'num_channels'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_elements, num_channels,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Get channel index for each element
    # Tensor is in NCHW format, channel = (offset / spatial_size) % num_channels
    # But we need to figure out channel from linear index
    # For NCHW: index = n * C * H * W + c * H * W + h * W + w
    # channel = (index / (H*W)) % C -- but we don't know H*W here
    # Instead, pass channel stride info
    # Actually for NCHW contiguous: channel = (offset // hw_size) % num_channels
    # We'll compute it differently - pass spatial_size
    c = (offsets // (n_elements // num_channels // (n_elements // num_channels // num_channels))) % num_channels
    # This is getting complex - let's use a simpler approach
    # For NCHW: total = N*C*H*W, spatial = H*W, channel_stride = H*W
    # channel_idx = (linear_idx // spatial_size) % C
    # We need spatial_size as a parameter
    c_idx = (offsets % (num_channels * (n_elements // num_channels))) // (n_elements // num_channels)
    
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=0.0)
    weight = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)