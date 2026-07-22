import torch
import triton
import triton.language as tl

# Fused LayerNorm kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=16, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    stride_xm, stride_xn,
    stride_om, stride_on,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N
    
    x = tl.load(x_ptr + row * stride_xm + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
    
    mean = tl.sum(x, axis=0) / N
    x_centered = x - mean
    var = tl.sum(x_centered * x_centered, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    
    out = x_centered * rstd * w + b
    tl.store(out_ptr + row * stride_om + cols * stride_on, out.to(tl.float32), mask=mask)