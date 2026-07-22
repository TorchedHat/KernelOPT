import torch
import triton
import triton.language as tl
import math

# Fused BN + ReLU kernel (pointwise)
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
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C,
    eps,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Channel index: layout is (N, C, H, W) so channel = (offset // (H*W)) % C
    # But we pass C and numel = N*C*H*W, spatial = numel // (N*C) ... 
    # Actually we need channel index per element
    # For NCHW: element index = n*C*HW + c*HW + hw
    # channel = (offset // HW) % C ... but HW = numel / (N*C)
    # Simpler: channel = (offset // (numel // (N*C))) % C
    # We'll pass HW directly
    # Actually let's just pass spatial size
    # channel index = (offset // spatial) % C
    # We'll compute it from numel and C: spatial = numel / N / C ... need N
    # Let's just pass spatial_size
    # For simplicity, pass C and spatial_size
    # offset = n * C * spatial + c * spatial + s
    # c = (offset // spatial) % C
    # We'll use a different approach: pass spatial as parameter
    # Actually let's restructure: pass spatial_size
    pass

# Let me restructure with spatial_size parameter

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, spatial,
    eps: tl.constexpr,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = (offsets // spatial) % 65536  # channel index, will be masked
    
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)