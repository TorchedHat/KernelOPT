import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused BN + Activation (pointwise epilogue)
# ============================================================

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
    ACT: tl.constexpr,  # 0=none, 1=relu, 2=relu6/hardtanh
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    # Compute channel index
    c = (offs // HW) % C
    
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=mask, other=1.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0).to(tl.float32)
    
    x = (x - mean) / tl.sqrt(var + eps)
    x = x * w + b
    
    if ACT == 1:
        x = tl.maximum(x, 0.0)
    elif ACT == 2:
        x = tl.minimum(tl.maximum(x, 0.0), 6.0)
    
    tl.store(out_ptr + offs, x.to(tl.float32), mask=mask)


import torch
import triton
import triton.language as tl