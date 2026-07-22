import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm + ReLU kernel (inference mode)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['C', 'numel'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    C, numel,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute channel index: tensor is (N, C, H, W) stored contiguously
    # channel = (offset // (H*W)) % C
    # But we need to figure out C position from flat index
    # For NCHW layout: index = n*C*H*W + c*H*W + h*W + w
    # c = (index // (H*W)) % C
    # We don't know H*W directly, but we know numel = N*C*H*W
    # We'll pass spatial_size = H*W
    c_idx = (offsets // (numel // (numel // C))) % C
    # Simpler: pass HW as parameter
    # Actually let's restructure - pass HW explicitly
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    weight = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * weight + bias
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)