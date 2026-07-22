import torch
import triton
import triton.language as tl
import math

# Fused kernel: masked_fill + softmax over attention scores
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 256}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=16, num_stages=2),
    ],
    key=['T'],
)
@triton.jit
def attn_softmax_kernel(
    B, H, T,
    stride_sb, stride_sh, stride_st, stride_sn,
    stride_bb, stride_bh, stride_bt, stride_bn,
    scale,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one row of the attention matrix
    row_idx = tl.program_id(0)
    total_rows = B * H * T
    if row_idx >= total_rows:
        return
    
    b = row_idx // (H * T)
    h = (row_idx // T) % H
    t = row_idx % T
    
    row_start = b * stride_sb + h * stride_sh + t * stride_st
    bias_row_start = t * stride_bt  # bias is [1,1,T,T], b=0,h=0
    
    # Load the row, apply scale and mask, compute softmax
    # First pass: find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    
    for off in tl.range(0, T, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < T
        
        score = tl.load(scores_ptr + row_start + cols * stride_sn, mask=mask, other=0.0).to(tl.float32)
        score = score * scale
        
        bias_val = tl.load(bias_ptr + bias_row_start + cols * stride_bn, mask=mask, other=0.0).to(tl.float32)
        # bias is tril(ones), so bias==0 means upper triangle -> fill with -inf
        score = tl.where(bias_val == 0, float('-inf'), score)
        
        score = tl.where(mask, score, float('-inf'))
        row_max = tl.maximum(row_max, tl.max(score, axis=0))
    
    # Second pass: compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    
    for off in tl.range(0, T, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < T
        
        score = tl.load(scores_ptr + row_start + cols * stride_sn, mask=mask, other=0.0).to(tl.float32)
        score = score * scale
        
        bias_val = tl.load(bias_ptr + bias_row_start + cols * stride_bn, mask=mask, other=0.0).to(tl.float32)
        score = tl.where(bias_val == 0, float('-inf'), score)
        score = tl.where(mask, score, float('-inf'))
        
        exp_val = tl.exp(score - row_max)
        exp_val = tl.where(mask, exp_val, 0.0)
        row_sum += tl.sum(exp_val, axis=0)
    
    # Third pass: normalize and store
    for off in tl.range(0, T, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < T
        
        score = tl.load(scores_ptr + row_start + cols * stride_sn, mask=mask, other=0.0).to(tl.float32)
        score = score * scale
        
        bias_val = tl.load(bias_ptr + bias_row_start + cols * stride_bn, mask=mask, other=0.0).to(tl.float32)
        score = tl.where(bias_val == 0, float('-inf'), score)
        score = tl.where(mask, score, float('-inf'))
        
        softmax_val = tl.exp(score - row_max) / row_sum
        softmax_val = tl.where(mask, softmax_val, 0.0)
        
        tl.store(scores_ptr + row_start + cols * stride_sn, softmax_val.to(tl.float32), mask=mask)


# Fused kernel: add bias to QKV projection output
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def add_bias_kernel(
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    # Each program handles BLOCK_SIZE elements
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M * N
    
    row = offset // N
    col = offset % N
    
    val = tl.load(out_ptr + offset, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col, mask=mask, other=0.0)
    tl.store(out_ptr + offset, val + bias, mask=mask)


# Fused kernel for output projection bias add
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
    ],
    key=['N'],
)
@triton.jit
def add_bias_proj_kernel(
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M * N
    
    col = offset % N
    
    val = tl.load(out_ptr + offset, mask=mask, other=0.0)
    bias = tl.load(bias_ptr + col, mask=mask, other=0.0)
    tl.store(out_ptr + offset, val + bias, mask=mask)


def kernel_function(p_c_attn_weight, p_c_attn_bias, p_c_proj_weight, p_c_proj_bias, b_bias, x):
    B, T, C = x.shape  # 128, 512, 768
    n_head = 8
    head_dim = C // n_head  # 96
    scale = 1.0 / math.sqrt(head_dim)
    
    # Step 1: QKV projection using cuBLAS
    # x: [B, T, C] -> [B*T, C]
    x_2d = x.reshape(B * T, C)
    # qkv: [B*T, 3C]
    qkv = torch.mm(x_2d, p_c_attn_weight.t())
    
    # Add bias using Triton kernel
    M_qkv = B * T
    N_qkv = 3 * C
    grid_bias = lambda meta: (triton.cdiv(M_qkv * N_qkv, meta['BLOCK_SIZE']),)
    add_bias_kernel[grid_bias](qkv, p_c_attn_bias, M_qkv, N_qkv)
    
    # Step 2: Reshape and split Q, K, V
    qkv_3d = qkv.reshape(B, T, 3 * C)
    
    # Q: [B, T, C] -> [B, n_head, T, head_dim]
    q = qkv_3d[:, :, :C].reshape(B, T, n_head, head_dim).permute(0, 2, 1, 3).contiguous()
    # K: [B, T, C] -> [B, n_head, T, head_dim]
    k = qkv_3d[:, :, C:2*C].reshape(B, T, n_head, head_dim).permute(0, 2, 1, 3).contiguous()
    # V: [B, T, C] -> [B, n_head, T, head_dim]
    v = qkv_3d[:, :, 2*C:].reshape(B, T, n_head, head_dim).permute(0, 2, 1, 3).contiguous()
    
    # Step 3: Attention scores Q @ K.T
    # q: [B, n_head, T, head_dim], k: [B, n_head, head_dim, T]
    k_t = k.transpose(-2, -1).contiguous()  # [B, n_head, head_dim, T]
    
    # Reshape for bmm: [B*n_head, T, head_dim] @ [B*n_head, head_dim, T]
    q_bh = q.reshape(B * n_head, T, head_dim)
    k_t_bh = k_t.reshape(B * n_head, head_dim, T)
    
    # att: [B*n_head, T, T]
    att = torch.bmm(q_bh, k_t_bh)  # [B*n_head, T, T]
    att = att.reshape(B, n_head, T, T)
    
    # Step 4: Apply scale, mask, softmax using Triton
    # b_bias: [1, 1, 1024, 1024], we need [:, :, :T, :T]
    bias_slice = b_bias[:, :, :T, :T].contiguous()  # [1, 1, T, T]
    
    total_rows = B * n_head * T
    grid_softmax = (total_rows,)
    
    attn_softmax_kernel[grid_softmax](
        att,
        bias_slice,
        B, n_head, T,
        att.stride(0), att.stride(1), att.stride(2), att.stride(3),
        bias_slice.stride(0), bias_slice.stride(1), bias_slice.stride(2), bias_slice.stride(3),
        scale,
    )
    # att is now softmax output: [B, n_head, T, T]
    
    # Step 5: att @ V
    att_bh = att.reshape(B * n_head, T, T)
    v_bh = v.reshape(B * n_head, T, head_dim)
    
    y = torch.bmm(att_bh, v_bh)  # [B*n_head, T, head_dim]
    y = y.reshape(B, n_head, T, head_dim)
    
    # Step 6: Reshape output
    y = y.transpose(1, 2).contiguous().reshape(B * T, C)
    
    # Step 7: Output projection using cuBLAS
    out = torch.mm(y, p_c_proj_weight.t())
    
    # Add bias
    M_out = B * T
    N_out = C
    grid_proj = lambda meta: (triton.cdiv(M_out * N_out, meta['BLOCK_SIZE']),)
    add_bias_proj_kernel[grid_proj](out, p_c_proj_bias, M_out, N_out)
    
    out = out.reshape(B, T, C)
    # dropout with p=0.0 is identity
    return out


def get_inputs():
    batch_size = 128
    seq_len = 512
    n_embd = 768
    n_head = 8
    max_seqlen = 1024
    
    p_c_attn_weight = torch.randn(2304, 768, device='cuda', dtype=torch.float32)
    p_c_attn_bias = torch.randn(2304, device='cuda', dtype=torch.float32)
    p_c_proj_weight = torch.randn(768, 768, device='cuda', dtype=torch.float32)
    p_c_proj_bias = torch.randn(768, device='cuda', dtype=torch.float32)
    b_bias = torch.tril(torch.ones(max_seqlen, max_seqlen, device='cuda', dtype=torch.float32)).view(1, 1, max_seqlen, max_seqlen)
    x = torch.randn(batch_size, seq_len, n_embd, device='cuda', dtype=torch.float32)
    
    return [p_c_attn_weight, p_c_attn_bias, p_c_proj_weight, p_c_proj_bias, b_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    
    if not args.profile_only:
        import torch.nn as nn
        import torch.nn.functional as F
        
        p_c_attn_weight, p_c_attn_bias, p_c_proj_weight, p_c_proj_bias, b_bias, x = inputs
        
        B, T, C = x.shape
        n_head = 8
        head_dim = C // n_head
        scale = 1.0 / math.sqrt(head_dim)
        
        # Reference implementation
        qkv = F.linear(x, p_c_attn_weight, p_c_attn_bias)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * scale
        att = att.masked_fill(b_bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        # dropout p=0.0 is identity
        y_ref = att @ v
        y_ref = y_ref.transpose(1, 2).contiguous().view(B, T, C)
        y_ref = F.linear(y_ref, p_c_proj_weight, p_c_proj_bias)
        # dropout p=0.0 is identity
        
        assert torch.allclose(output, y_ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - y_ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")