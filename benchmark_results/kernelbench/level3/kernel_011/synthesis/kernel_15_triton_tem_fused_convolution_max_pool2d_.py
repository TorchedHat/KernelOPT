import torch
import triton
import triton.language as tl
import math

# Fused bias+relu pointwise kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def bias_relu_kernel(x_ptr, bias_ptr, out_ptr, n_elements, n_channels, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = tl.maximum(x + b, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements', 'n_channels'],
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