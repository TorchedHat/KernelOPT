import torch
import triton
import triton.language as tl
import math

# Kernel 1: Add bias and rearrange Q, K, V
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def bias_add_kernel(
    mm_ptr, bias_ptr, q_ptr, k_ptr, v_ptr,
    B, T, NH, HS,
    N: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each element in output Q/K/V: index = b * NH * T * HS + h * T * HS + t * HS + s
    # Total elements per Q/K/V: B * T * NH * HS
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # offs indexes into [B, NH, T, HS] layout
    s = offs % HS
    t = (offs // HS) % T
    h = (offs // (HS * T)) % NH
    b = offs // (HS * T * NH)
    
    # Source in mm output: [B, T, 3*NH*HS] = [B, T, 2304]
    # Q is at offset 0, K at NH*HS=768, V at 2*NH*HS=1536
    C = NH * HS  # 768
    
    src_q = b * (T * 3 * C) + t * (3 * C) + h * HS + s
    src_k = b * (T * 3 * C) + t * (3 * C) + C + h * HS + s
    src_v = b * (T * 3 * C) + t * (3 * C) + 2 * C + h * HS + s
    
    bias_q = h * HS + s
    bias_k = C + h * HS + s
    bias_v = 2 * C + h * HS + s
    
    q_val = tl.load(mm_ptr + src_q, mask=mask) + tl.load(bias_ptr + bias_q, mask=mask)
    k_val = tl.load(mm_ptr + src_k, mask=mask) + tl.load(bias_ptr + bias_k, mask=mask)
    v_val = tl.load(mm_ptr + src_v, mask=mask) + tl.load(bias_ptr + bias_v, mask=mask)
    
    tl.store(q_ptr + offs, q_val, mask=mask)
    tl.store(k_ptr + offs, k_val, mask=mask)
    tl.store(v_ptr + offs, v_val, mask=mask)