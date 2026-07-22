import torch
import triton
import triton.language as tl
import math

# QKV projection: [B*T, C] @ [C, 3C]^T + bias -> [B*T, 3C]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def qkv_gemm_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Store
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


# Flash attention kernel
# Each program handles one (batch, head) pair, processes all T queries
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64, 'BLOCK_S': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 128, 'BLOCK_S': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_T': 64, 'BLOCK_S': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 128, 'BLOCK_S': 128}, num_stages=2, num_warps=8),
    ],
    key=['T', 'HS'],
)
@triton.jit
def flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, O_ptr,
    bias_ptr,
    B, NH, T, HS,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    stride_biasb, stride_biash, stride_biast, stride_biass,
    scale,
    BLOCK_T: tl.constexpr,
    BLOCK_S: tl.constexpr,
):
    # Each program handles one (batch, head) pair and one tile of queries
    pid_bh = tl.program_id(0)
    pid_t = tl.program_id(1)
    
    b = pid_bh // NH
    h = pid_bh % NH
    
    # Query tile
    q_start = pid_t * BLOCK_T
    q_offs = q_start + tl.arange(0, BLOCK_T)
    d_offs = tl.arange(0, 64)  # head_size = 96, need to handle carefully
    
    # Load Q tile [BLOCK_T, HS]
    # We'll process HS in chunks if needed, but HS=96 fits in registers
    # Actually let's load Q fully
    q_ptrs = Q_ptr + b * stride_qb + h * stride_qh + q_offs[:, None] * stride_qt + d_offs[None, :] * stride_qd
    q_mask = (q_offs[:, None] < T) & (d_offs[None, :] < HS)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)
    
    # We need to handle HS=96 which isn't a power of 2
    # Load second half
    d_offs2 = 64 + tl.arange(0, 32)
    q_ptrs2 = Q_ptr + b * stride_qb + h * stride_qh + q_offs[:, None] * stride_qt + d_offs2[None, :] * stride_qd
    q_mask2 = (q_offs[:, None] < T) & (d_offs2[None, :] < HS)
    q2 = tl.load(q_ptrs2, mask=q_mask2, other=0.0)
    
    # Online softmax accumulators
    m_i = tl.full([BLOCK_T], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_T], dtype=tl.float32)
    acc = tl.zeros([BLOCK_T, 64], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_T, 32], dtype=tl.float32)
    
    # Iterate over key/value blocks
    for s in range(0, tl.cdiv(T, BLOCK_S)):
        s_start = s * BLOCK_S
        s_offs = s_start + tl.arange(0, BLOCK_S)
        
        # Load K tile [BLOCK_S, HS]
        k_ptrs = K_ptr + b * stride_kb + h * stride_kh + s_offs[:, None] * stride_kt + d_offs[None, :] * stride_kd
        k_mask = (s_offs[:, None] < T) & (d_offs[None, :] < HS)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        k_ptrs2 = K_ptr + b * stride_kb + h * stride_kh + s_offs[:, None] * stride_kt + d_offs2[None, :] * stride_kd
        k_mask2 = (s_offs[:, None] < T) & (d_offs2[None, :] < HS)
        k2 = tl.load(k_ptrs2, mask=k_mask2, other=0.0)
        
        # Compute QK^T [BLOCK_T, BLOCK_S]
        qk = tl.dot(q, tl.trans(k), allow_tf32=False)
        qk += tl.dot(q2, tl.trans(k2), allow_tf32=False)
        qk = qk * scale
        
        # Apply causal mask
        # bias is [1, 1, 1024, 1024], we need bias[:,:,:T,:T]
        # causal: position q_offs can attend to s_offs where s_offs <= q_offs
        causal_mask = q_offs[:, None] >= s_offs[None, :]
        # Also check bounds
        valid_mask = (q_offs[:, None] < T) & (s_offs[None, :] < T)
        qk = tl.where(causal_mask & valid_mask, qk, float('-inf'))
        
        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])
        # Mask out invalid positions
        p = tl.where(valid_mask, p, 0.0)
        
        l_i = l_i * alpha + tl.sum(p, axis=1)
        
        # Load V tile [BLOCK_S, HS]
        v_ptrs = V_ptr + b * stride_vb + h * stride_vh + s_offs[:, None] * stride_vt + d_offs[None, :] * stride_vd
        v = tl.load(v_ptrs, mask=k_mask, other=0.0)
        v_ptrs2 = V_ptr + b * stride_vb + h * stride_vh + s_offs[:, None] * stride_vt + d_offs2[None, :] * stride_vd
        v2 = tl.load(v_ptrs2, mask=k_mask2, other=0.0)
        
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v, allow_tf32=False)
        acc2 = acc2 * alpha[:, None] + tl.dot(p.to(tl.float32), v2, allow_tf32=False)
        
        m_i = m_new
    
    # Normalize
    acc = acc / l_i[:, None]
    acc2 = acc2 / l_i[:, None]
    
    # Store output
    o_ptrs = O_ptr + b * stride_ob + h * stride_oh + q_offs[:, None] * stride_ot + d_offs[None, :] * stride_od
    o_mask = (q_offs[:, None] < T) & (d_offs[None, :] < HS)
    tl.store(o_ptrs, acc.to(tl.float32), mask=o_mask)
    
    o_ptrs2 = O_ptr + b * stride_ob + h * stride_oh + q_offs[:, None] * stride_ot + d_offs2[None, :] * stride_od
    o_mask2 = (q_offs[:, None] < T) & (d_offs2[None, :] < HS)
    tl.store(o_ptrs2, acc2.to(tl.float32), mask=o_mask2)


# Output projection: [B*T, C] @ [C, C]^T + bias -> [B*T, C]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def proj_gemm_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // grid_n
    pid_n = pid % grid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Store
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


def kernel_function(p_c_attn_weight, p_c_attn_bias, p_c_proj_weight, p_c_proj_bias, b_bias, x):
    B, T, C = x.shape  # 128, 512, 768
    NH = 8
    HS = C // NH  # 96
    
    # Step 1: QKV projection [B*T, C] @ [C, 3C] + bias -> [B*T, 3C]
    BT = B * T
    qkv = torch.empty((BT, 3 * C), device=x.device, dtype=x.dtype)
    
    x_2d = x.reshape(BT, C)
    # p_c_attn_weight is [3C, C], we want x @ W^T = x @ [C, 3C]
    # So A = x_2d [BT, C], B = p_c_attn_weight^T [C, 3C]
    # B stored as [3C, C] with stride_bk=C (row), stride_bn=1 (col)
    # We want B[k, n] = p_c_attn_weight[n, k]
    
    M_qkv, N_qkv, K_qkv = BT, 3 * C, C
    grid_qkv = lambda meta: (triton.cdiv(M_qkv, meta['BLOCK_M']) * triton.cdiv(N_qkv, meta['BLOCK_N']),)
    
    qkv_gemm_kernel[grid_qkv](
        x_2d, p_c_attn_weight, p_c_attn_bias, qkv,
        M_qkv, N_qkv, K_qkv,
        x_2d.stride(0), x_2d.stride(1),
        p_c_attn_weight.stride(1), p_c_attn_weight.stride(0),  # transposed access
        qkv.stride(0), qkv.stride(1),
    )
    
    # Step 2: Reshape QKV and run flash attention
    # qkv: [BT, 3C] -> [B, T, 3C]
    qkv_3d = qkv.reshape(B, T, 3 * C)
    
    # Split into Q, K, V: each [B, T, C]
    # Then reshape to [B, NH, T, HS]
    # Q: qkv[:, :, :C], K: qkv[:, :, C:2C], V: qkv[:, :, 2C:]
    
    # Allocate Q, K, V as contiguous [B, NH, T, HS]
    Q = torch.empty((B, NH, T, HS), device=x.device, dtype=x.dtype)
    K_t = torch.empty((B, NH, T, HS), device=x.device, dtype=x.dtype)
    V = torch.empty((B, NH, T, HS), device=x.device, dtype=x.dtype)
    
    # Fill Q, K, V from qkv using a pointwise kernel
    # Q[b, h, t, d] = qkv_3d[b, t, h*HS + d]
    # K[b, h, t, d] = qkv_3d[b, t, C + h*HS + d]
    # V[b, h, t, d] = qkv_3d[b, t, 2*C + h*HS + d]
    
    total_qkv = B * NH * T * HS
    
    @triton.jit
    def split_qkv_kernel(qkv_ptr, Q_ptr, K_ptr, V_ptr,
                          B, NH, T, HS, C,
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < B * NH * T * HS
        
        # Decode index
        d = offs % HS
        t = (offs // HS) % T
        h = (offs // (HS * T)) % NH
        b = offs // (HS * T * NH)
        
        # Source index in qkv: [b, t, h*HS + d] for Q
        q_src = b * T * C * 3 + t * C * 3 + h * HS + d
        k_src = b * T * C * 3 + t * C * 3 + C + h * HS + d
        v_src = b * T * C * 3 + t * C * 3 + 2 * C + h * HS + d
        
        q_val = tl.load(qkv_ptr + q_src, mask=mask, other=0.0)
        k_val = tl.load(qkv_ptr + k_src, mask=mask, other=0.0)
        v_val = tl.load(qkv_ptr + v_src, mask=mask, other=0.0)
        
        tl.store(Q_ptr + offs, q_val, mask=mask)
        tl.store(K_ptr + offs, k_val, mask=mask)
        tl.store(V_ptr + offs, v_val, mask=mask)
    
    SPLIT_BLOCK = 1024
    split_qkv_kernel[(triton.cdiv(total_qkv, SPLIT_BLOCK),)](
        qkv, Q, K_t, V,
        B, NH, T, HS, C,
        BLOCK=SPLIT_BLOCK,
    )
    
    # Step 3: Flash attention
    # O: [B, NH, T, HS]
    O = torch.empty((B, NH, T, HS), device=x.device, dtype=x.dtype)
    scale = 1.0 / math.sqrt(HS)
    
    BLOCK_T_FA = 64
    grid_fa = (B * NH, triton.cdiv(T, BLOCK_T_FA))
    
    flash_attn_kernel[grid_fa](
        Q, K_t, V, O,
        b_bias,
        B, NH, T, HS,
        Q.stride(0), Q.stride(1), Q.stride(2), Q.stride(3),
        K_t.stride(0), K_t.stride(1), K_t.stride(2), K_t.stride(3),
        V.stride(0), V.stride(1), V.stride(2), V.stride(3),
        O.stride(0), O.stride(1), O.stride(2), O.stride(3),
        b_bias.stride(0), b_bias.stride(1), b_bias.stride(2), b_bias.stride(3),
        scale,
        BLOCK_T=BLOCK_T_FA,
        BLOCK_S=64,
    )
    
    # Step 4: Reshape O to [B*T, C]
    # O: [B, NH, T, HS] -> transpose to [B, T, NH, HS] -> [B*T, C]
    # Need to do this reshape with a kernel
    O_reshaped = torch.empty((BT, C), device=x.device, dtype=x.dtype)
    
    @triton.jit
    def reshape_output_kernel(O_ptr, out_ptr,
                               B, NH, T, HS,
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        total = B * T * NH * HS
        mask = offs < total
        
        # Output index: [b*T + t, h*HS + d] = [b, t, h, d] in [B, T, NH, HS]
        d = offs % HS
        h = (offs // HS) % NH
        t = (offs // (HS * NH)) % T
        b = offs // (HS * NH * T)
        
        # Source in O: [b, h, t, d]
        src = b * NH * T * HS + h * T * HS + t * HS + d
        val = tl.load(O_ptr + src, mask=mask, other=0.0)
        
        # Dest: [b*T + t, h*HS + d]
        dst = (b * T + t) * NH * HS + h * HS + d
        tl.store(out_ptr + dst, val, mask=mask)
    
    RESHAPE_BLOCK = 1024
    reshape_output_kernel[(triton.cdiv(BT * C, RESHAPE_BLOCK),)](
        O, O_reshaped,
        B, NH, T, HS,
        BLOCK=RESHAPE_BLOCK,
    )
    
    # Step 5: Output projection [B*T, C] @ [C, C]^T + bias -> [B*T, C]
    out = torch.empty((BT, C), device=x.device, dtype=x.dtype)
    
    M_proj, N_proj, K_proj = BT, C, C
    grid_proj = lambda meta: (triton.cdiv(M_proj, meta['BLOCK_M']) * triton.cdiv(N_proj, meta['BLOCK_N']),)
    
    proj_gemm_kernel[grid_proj](
        O_reshaped, p_c_proj_weight, p_c_proj_bias, out,
        M_proj, N_proj, K_proj,
        O_reshaped.stride(0), O_reshaped.stride(1),
        p_c_proj_weight.stride(1), p_c_proj_weight.stride(0),  # transposed
        out.stride(0), out.stride(1),
    )
    
    return out.reshape(B, T, C)


def get_inputs():
    batch_size = 128
    seq_len = 512
    n_embd = 768
    n_head = 8
    max_seqlen = 1024
    
    p_c_attn_weight = torch.randn(3 * n_embd, n_embd, device='cuda', dtype=torch.float32)
    p_c_attn_bias = torch.randn(3 * n_embd, device='cuda', dtype=torch.float32)
    p_c_proj_weight = torch.randn(n_embd, n_embd, device='cuda', dtype=torch.float32)
    p_c_proj_bias = torch.randn(n_embd, device='cuda', dtype=torch.float32)
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
        NH = 8
        HS = C // NH
        
        # Reference implementation
        qkv = F.linear(x, p_c_attn_weight, p_c_attn_bias)
        q, k, v = qkv.split(C, dim=2)
        k = k.view(B, T, NH, HS).transpose(1, 2)
        q = q.view(B, T, NH, HS).transpose(1, 2)
        v = v.view(B, T, NH, HS).transpose(1, 2)
        
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(HS))
        att = att.masked_fill(b_bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        ref = F.linear(y, p_c_proj_weight, p_c_proj_bias)
        
        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")