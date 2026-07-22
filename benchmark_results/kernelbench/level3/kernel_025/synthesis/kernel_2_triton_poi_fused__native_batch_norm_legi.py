import torch
import triton
import triton.language as tl
import math

# BN+ReLU fused pointwise kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Compute channel index: layout is (N, C, H, W) -> flat index
    # channel = (offset / HW) % C
    c_idx = (offsets // HW) % C
    
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    
    if do_relu:
        out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def add_kernel(
    a_ptr, b_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    a = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    b = tl.load(b_ptr + offsets, mask=mask, other=0.0)
    tl.store(out_ptr + offsets, a + b, mask=mask)


# Implicit GEMM conv2d kernel for 1x1 convolutions with groups
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_grouped_kernel(
    X_ptr, W_ptr, Out_ptr,
    M, N, K,
    stride_xn, stride_xc, stride_xhw,
    stride_wout, stride_win,
    stride_on, stride_oc, stride_ohw,
    GROUPS: tl.constexpr,
    GROUP_IN_C: tl.constexpr,
    GROUP_OUT_C: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M = N_batch * H * W, N = OUT_C, K = IN_C
    # Each program handles one group
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode m -> (batch, hw)
    HW = M // (stride_xn // (stride_xc * GROUP_IN_C * GROUPS))  # This is tricky
    # Actually M = BATCH * HW
    # Let's just use flat indexing
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, GROUP_IN_C, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        
        # X layout: (N, C, H*W) with C being group's channels
        # x_ptr offset: batch * stride_xn + (pid_g * GROUP_IN_C + offs_k) * stride_xc + hw * stride_xhw
        # offs_m encodes (batch * HW + hw)
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < GROUP_IN_C)
        w_mask = (offs_k[:, None] < GROUP_IN_C) & (offs_n[None, :] < GROUP_OUT_C)
        
        # X: shape (BATCH, IN_C, HW), access as (offs_m, pid_g*GROUP_IN_C + offs_k)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xhw + (pid_g * GROUP_IN_C + offs_k[None, :]) * stride_xc
        w_ptrs = W_ptr + (pid_g * GROUP_OUT_C + offs_n[None, :]) * stride_wout + offs_k[:, None] * stride_win
        
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store output
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < GROUP_OUT_C)
    out_ptrs = Out_ptr + offs_m[:, None] * stride_ohw + (pid_g * GROUP_OUT_C + offs_n[None, :]) * stride_oc
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)