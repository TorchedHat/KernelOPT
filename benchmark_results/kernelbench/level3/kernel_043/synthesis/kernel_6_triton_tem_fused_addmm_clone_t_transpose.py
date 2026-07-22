import torch
import triton
import triton.language as tl
import math

# Flash attention kernel: fused Q@K^T * scale + causal mask + softmax + @V
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['T', 'HS'],
)
@triton.jit
def flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr, Bias_ptr,
    B, H, T, HS,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles one (batch, head, tile_m) block
    pid_bh = tl.program_id(0)
    pid_m = tl.program_id(1)
    
    pid_b = pid_bh // H
    pid_h = pid_bh % H
    
    # Row range for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_K)
    
    # Pointers to Q for this block
    Q_base = Q_ptr + pid_b * stride_qb + pid_h * stride_qh
    K_base = K_ptr + pid_b * stride_kb + pid_h * stride_kh
    V_base = V_ptr + pid_b * stride_vb + pid_h * stride_vh
    
    # Initialize accumulator, running max, running sum
    acc = tl.zeros((BLOCK_M, BLOCK_K), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    
    mask_m = offs_m < T
    
    # Load Q block: [BLOCK_M, HS] - iterate over HS in chunks of BLOCK_K
    # For flash attention, we need full Q rows
    # Let's load Q fully (HS=96 fits in registers with BLOCK_K=96)
    
    # Iterate over K/V blocks (columns of attention matrix)
    for start_n in range(0, T, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < T
        
        # Compute QK^T for this block: [BLOCK_M, BLOCK_N]
        qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        
        for k_start in range(0, HS, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < HS
            
            # Load Q tile [BLOCK_M, BLOCK_K]
            q_ptrs = Q_base + offs_m[:, None] * stride_qt + offs_k[None, :] * stride_qd
            q_tile = tl.load(q_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
            
            # Load K tile [BLOCK_N, BLOCK_K]
            k_ptrs = K_base + offs_n[:, None] * stride_kt + offs_k[None, :] * stride_kd
            k_tile = tl.load(k_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            
            qk = tl.dot(q_tile, tl.trans(k_tile), qk)
        
        qk = qk * scale
        
        # Causal mask: mask out positions where n > m
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        qk = tl.where(causal_mask & mask_m[:, None] & mask_n[None, :], qk, float('-inf'))
        
        # Online softmax update
        m_ij = tl.max(qk, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        # Mask invalid positions
        p = tl.where(mask_n[None, :], p, 0.0)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # Update accumulator
        acc = acc * alpha[:, None]
        
        # Load V tile [BLOCK_N, BLOCK_K] and accumulate
        for k_start in range(0, HS, BLOCK_K):
            offs_k = k_start + tl.arange(0, BLOCK_K)
            mask_k = offs_k < HS
            
            v_ptrs = V_base + offs_n[:, None] * stride_vt + offs_k[None, :] * stride_vd
            v_tile = tl.load(v_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
            
            # p: [BLOCK_M, BLOCK_N], v_tile: [BLOCK_N, BLOCK_K]
            acc_slice = tl.dot(p.to(v_tile.dtype), v_tile)
            # We need to accumulate into the right slice of acc
            # This approach won't work cleanly with BLOCK_K < HS
            # Let's restructure
        
        m_i = m_new
    
    # Normalize
    acc = acc / l_i[:, None]
    
    # Store output
    Out_base = Out_ptr + pid_b * stride_ob + pid_h * stride_oh
    for k_start in range(0, HS, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < HS
        out_ptrs = Out_base + offs_m[:, None] * stride_ot + offs_k[None, :] * stride_od
        tl.store(out_ptrs, acc[:, :].to(tl.float32), mask=mask_m[:, None] & mask_k[None, :])