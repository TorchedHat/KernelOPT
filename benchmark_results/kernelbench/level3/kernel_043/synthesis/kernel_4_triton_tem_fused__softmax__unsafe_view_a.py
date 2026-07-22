import torch
import triton
import triton.language as tl
import math

# HEAD_DIM = 96, pad to 128 (next power of 2)
HEAD_DIM_PADDED = 128

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=3),
    ],
    key=['T', 'HEAD_DIM'],
)
@triton.jit
def flash_attn_causal_kernel(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    B, H, T,
    scale,
    HEAD_DIM: tl.constexpr,
    HEAD_DIM_PADDED: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # program_id(0) = query block index (within sequence)
    # program_id(1) = batch * head index
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // H
    head_idx = pid_bh % H

    # Query block start
    q_start = pid_m * BLOCK_M
    offs_m = q_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM_PADDED)

    # Pointers to Q block
    Q_base = Q_ptr + batch_idx * stride_qb + head_idx * stride_qh
    q_ptrs = Q_base + offs_m[:, None] * stride_qt + offs_d[None, :] * stride_qd
    q_mask = (offs_m[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
    q = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Initialize accumulators
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM_PADDED], dtype=tl.float32)

    K_base = K_ptr + batch_idx * stride_kb + head_idx * stride_kh
    V_base = V_ptr + batch_idx * stride_vb + head_idx * stride_vh

    # Causal: only attend to positions <= current query position
    # For query block [q_start, q_start+BLOCK_M), attend to keys [0, q_start+BLOCK_M)
    # Last key block may be partial (causal masking within block)
    causal_end = q_start + BLOCK_M  # exclusive upper bound for key blocks

    for n_start in range(0, causal_end, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)

        # Load K block [BLOCK_N, HEAD_DIM_PADDED]
        k_ptrs = K_base + offs_n[:, None] * stride_kt + offs_d[None, :] * stride_kd
        k_mask = (offs_n[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)

        # Compute attention scores [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale

        # Apply causal mask: query i can only attend to key j if j <= i
        causal_mask = offs_m[:, None] >= offs_n[None, :]
        # Also mask out-of-bounds keys
        valid_mask = offs_n[None, :] < T
        scores = tl.where(causal_mask & valid_mask, scores, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])

        l_i = alpha * l_i + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        # Load V block [BLOCK_N, HEAD_DIM_PADDED]
        v_ptrs = V_base + offs_n[:, None] * stride_vt + offs_d[None, :] * stride_vd
        v_mask = (offs_n[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc = acc + tl.dot(p.to(v.dtype), v, out_dtype=tl.float32)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output
    Out_base = Out_ptr + batch_idx * stride_ob + head_idx * stride_oh
    out_ptrs = Out_base + offs_m[:, None] * stride_ot + offs_d[None, :] * stride_od
    out_mask = (offs_m[:, None] < T) & (offs_d[None, :] < HEAD_DIM)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    pid = tl.program_id(0)
    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    width = GROUP_M * grid_n
    group_id = pid // width
    group_size = tl.minimum(grid_m - group_id * GROUP_M, GROUP_M)
    pid_m = group_id * GROUP_M + (pid % group_size)
    pid_n = (pid % width) // group_size

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = (offs_k[None, :] + k * BLOCK_K) < K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask, other=0.0)
        k_mask2 = (offs_k[:, None] + k * BLOCK_K) < K
        b = tl.load(B_ptrs, mask=k_mask2 & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)


def kernel_function(p_c_attn_weight, p_c_attn_bias, p_c_proj_weight, p_c_proj_bias, b_bias, x):
    B, T, C = x.shape
    n_head = 8
    head_dim = C // n_head  # 96
    scale = 1.0 / math.sqrt(head_dim)

    # Step 1: Linear projection x @ c_attn_weight.T + c_attn_bias
    # x: [B, T, C] -> [B*T, C], weight: [2304, 768] -> [768, 2304]
    x_2d = x.reshape(B * T, C)
    qkv = torch.empty(B * T, 3 * C, device=x.device, dtype=x.dtype)

    M_qkv = B * T
    N_qkv = 3 * C
    K_qkv = C

    grid_qkv = lambda meta: (triton.cdiv(M_qkv, meta['BLOCK_M']) * triton.cdiv(N_qkv, meta['BLOCK_N']),)
    matmul_bias_kernel[grid_qkv](
        x_2d, p_c_attn_weight.t().contiguous(), p_c_attn_bias, qkv,
        M_qkv, N_qkv, K_qkv,
        x_2d.stride(0), x_2d.stride(1),
        p_c_attn_weight.stride(0), p_c_attn_weight.stride(1),  # weight is [2304, 768], t() gives [768, 2304]
        qkv.stride(0), qkv.stride(1),
        GROUP_M=8,
        HAS_BIAS=True,
    )

    # Reshape qkv to [B, T, 3, n_head, head_dim] and split
    qkv_3d = qkv.reshape(B, T, 3 * C)

    # Extract Q, K, V and reshape to [B, n_head, T, head_dim]
    # Q: [B, T, C] -> [B, n_head, T, head_dim]
    q = qkv_3d[:, :, :C].reshape(B, T, n_head, head_dim).permute(0, 2, 1, 3).contiguous()
    k = qkv_3d[:, :, C:2*C].reshape(B, T, n_head, head_dim).permute(0, 2, 1, 3).contiguous()
    v = qkv_3d[:, :, 2*C:].reshape(B, T, n_head, head_dim).permute(0, 2, 1, 3).contiguous()

    # Step 2: Flash Attention with causal mask
    # q, k, v: [B, n_head, T, head_dim]
    attn_out = torch.empty(B, n_head, T, head_dim, device=x.device, dtype=x.dtype)

    grid_attn = lambda meta: (
        triton.cdiv(T, meta['BLOCK_M']),
        B * n_head,
    )

    flash_attn_causal_kernel[grid_attn](
        q, k, v, attn_out,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
        B, n_head, T,
        scale,
        HEAD_DIM=head_dim,
        HEAD_DIM_PADDED=HEAD_DIM_PADDED,
    )

    # Step 3: Reshape attention output [B, n_head, T, head_dim] -> [B*T, C]
    y = attn_out.permute(0, 2, 1, 3).contiguous().reshape(B * T, C)

    # Step 4: Output projection y @ c_proj_weight.T + c_proj_bias
    out = torch.empty(B * T, C, device=x.device, dtype=x.dtype)

    M_proj = B * T
    N_proj = C
    K_proj = C

    grid_proj = lambda meta: (triton.cdiv(M_proj, meta['BLOCK_M']) * triton.cdiv(N_proj, meta['BLOCK_N']),)
    matmul_bias_kernel[grid_proj](
        y, p_c_proj_weight.t().contiguous(), p_c_proj_bias, out,
        M_proj, N_proj, K_proj,
        y.stride(0), y.stride(1),
        p_c_proj_weight.stride(0), p_c_proj_weight.stride(1),
        out.stride(0), out.stride(1),
        GROUP_M=8,
        HAS_BIAS=True,
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
    b_bias = torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen).cuda()
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
        p_c_attn_weight, p_c_attn_bias, p_c_proj_weight, p_c_proj_bias, b_bias, x = inputs
        B, T, C = x.shape
        n_head = 8
        n_embd = C

        # Reference implementation
        import torch.nn.functional as F

        qkv = F.linear(x, p_c_attn_weight, p_c_attn_bias)
        q, k, v = qkv.split(n_embd, dim=2)
        head_dim = C // n_head
        k = k.view(B, T, n_head, head_dim).transpose(1, 2)
        q = q.view(B, T, n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, n_head, head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(head_dim))
        att = att.masked_fill(b_bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y_ref = att @ v
        y_ref = y_ref.transpose(1, 2).contiguous().view(B, T, C)
        y_ref = F.linear(y_ref, p_c_proj_weight, p_c_proj_bias)

        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - y_ref).abs().max().item():.6f}")
        assert torch.allclose(output, y_ref, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={(output - y_ref).abs().max().item():.4e}"
        print("Kernel executed successfully.")