import torch
import triton
import triton.language as tl
import math

# ─── LayerNorm kernel ───────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 768}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_N)
    mask = cols < N

    x = tl.load(x_ptr + row * N + cols, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = tl.math.rsqrt(var + eps)
    x_norm = diff * rstd

    w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    out = x_norm * w + b
    tl.store(out_ptr + row * N + cols, out.to(tl.float32), mask=mask)


# ─── GEMM kernel (M x K) x (K x N) -> (M x N) with bias ────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=mask)


# ─── Attention score kernel: Q @ K^T * scale + mask -> softmax ───────────────
# Input: Q [B, H, T, hs], K [B, H, T, hs], bias [1,1,T,T]
# Output: att [B, H, T, T]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 32, 'BLOCK_HS': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 64, 'BLOCK_HS': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 32, 'BLOCK_HS': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 64, 'BLOCK_HS': 64}, num_warps=8, num_stages=2),
    ],
    key=['T', 'hs'],
)
@triton.jit
def attn_score_softmax_kernel(
    q_ptr, k_ptr, bias_ptr, out_ptr,
    B, H, T, hs,
    scale,
    stride_qb, stride_qh, stride_qt, stride_qhs,
    stride_kb, stride_kh, stride_kt, stride_khs,
    stride_ob, stride_oh, stride_ot, stride_os,
    BLOCK_T: tl.constexpr,
    BLOCK_HS: tl.constexpr,
):
    # Each program handles one (b, h, t_row) -> computes row of attention scores
    bh_idx = tl.program_id(0)
    t_row = tl.program_id(1)

    b_idx = bh_idx // H
    h_idx = bh_idx % H

    # Load Q row: [hs]
    q_base = q_ptr + b_idx * stride_qb + h_idx * stride_qh + t_row * stride_qt
    hs_offs = tl.arange(0, BLOCK_HS)

    # Compute Q @ K^T for all T columns
    # We'll do this in tiles over hs dimension
    # Result: scores[T]
    # Then apply mask and softmax

    # First pass: compute raw scores
    scores = tl.zeros((BLOCK_T,), dtype=tl.float32)

    # We need to iterate over T columns in blocks
    # For each column t_col, score = sum_hs q[hs] * k[t_col, hs]
    # This is a dot product

    # Actually let's do it differently - iterate over t_col blocks
    # For each BLOCK_T of columns, compute partial dot products

    # Load full Q row first (hs=96 fits in registers)
    q_row = tl.zeros((BLOCK_HS,), dtype=tl.float32)
    for hs_start in range(0, tl.cdiv(hs, BLOCK_HS)):
        hs_idx = hs_start * BLOCK_HS + hs_offs
        hs_mask = hs_idx < hs
        q_chunk = tl.load(q_base + hs_idx * stride_qhs, mask=hs_mask, other=0.0)
        q_row = tl.where(hs_mask, q_chunk, q_row)

    # Now compute scores for all T columns
    # We'll store them in a buffer - but T=512 is large
    # Let's use online softmax approach

    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((), dtype=tl.float32)

    # We need to store all scores for the second pass
    # T=512, store in SRAM via loop

    # Actually for softmax we need two passes or online algorithm
    # Let's do online softmax with blocks of T

    # Store scores temporarily - we'll recompute in second pass
    # First pass: find max
    t_cols = tl.arange(0, BLOCK_T)

    # Bias base for this row
    bias_base = bias_ptr + t_row * 1024  # bias shape [1,1,1024,1024], stride last dim=1

    for t_start in range(0, tl.cdiv(T, BLOCK_T)):
        t_idx = t_start * BLOCK_T + t_cols
        t_mask = t_idx < T

        # Compute dot product q_row . k[t_idx]
        score = tl.zeros((BLOCK_T,), dtype=tl.float32)
        k_base = k_ptr + b_idx * stride_kb + h_idx * stride_kh
        for hs_start in range(0, tl.cdiv(hs, BLOCK_HS)):
            hs_idx2 = hs_start * BLOCK_HS + hs_offs
            hs_mask2 = hs_idx2 < hs
            # Load K tile: [BLOCK_T, BLOCK_HS]
            k_ptrs = k_base + t_idx[:, None] * stride_kt + hs_idx2[None, :] * stride_khs
            k_tile = tl.load(k_ptrs, mask=t_mask[:, None] & hs_mask2[None, :], other=0.0)
            q_chunk2 = tl.where(hs_mask2, q_row, 0.0)  # [BLOCK_HS]
            score += tl.sum(k_tile * q_chunk2[None, :], axis=1)

        score = score * scale

        # Apply causal mask
        bias_vals = tl.load(bias_base + t_idx, mask=t_mask, other=0.0)
        score = tl.where(t_mask & (bias_vals == 0.0), float('-inf'), score)
        score = tl.where(~t_mask, float('-inf'), score)

        # Online softmax update
        block_max = tl.max(score, axis=0)
        new_max = tl.maximum(row_max, block_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(score - new_max), axis=0)
        row_max = new_max

    # Second pass: normalize and store
    out_base = out_ptr + b_idx * stride_ob + h_idx * stride_oh + t_row * stride_ot

    for t_start in range(0, tl.cdiv(T, BLOCK_T)):
        t_idx = t_start * BLOCK_T + t_cols
        t_mask = t_idx < T

        # Recompute scores
        score = tl.zeros((BLOCK_T,), dtype=tl.float32)
        k_base = k_ptr + b_idx * stride_kb + h_idx * stride_kh
        for hs_start in range(0, tl.cdiv(hs, BLOCK_HS)):
            hs_idx2 = hs_start * BLOCK_HS + hs_offs
            hs_mask2 = hs_idx2 < hs
            k_ptrs = k_base + t_idx[:, None] * stride_kt + hs_idx2[None, :] * stride_khs
            k_tile = tl.load(k_ptrs, mask=t_mask[:, None] & hs_mask2[None, :], other=0.0)
            q_chunk2 = tl.where(hs_mask2, q_row, 0.0)
            score += tl.sum(k_tile * q_chunk2[None, :], axis=1)

        score = score * scale
        bias_vals = tl.load(bias_base + t_idx, mask=t_mask, other=0.0)
        score = tl.where(t_mask & (bias_vals == 0.0), float('-inf'), score)
        score = tl.where(~t_mask, float('-inf'), score)

        softmax_out = tl.exp(score - row_max) / row_sum
        softmax_out = tl.where(t_mask, softmax_out, 0.0)
        tl.store(out_base + t_idx * stride_os, softmax_out, mask=t_mask)


# ─── Attention value kernel: att @ V ────────────────────────────────────────
# att [B, H, T, T], V [B, H, T, hs] -> out [B, H, T, hs]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T_OUT': 32, 'BLOCK_T_IN': 32, 'BLOCK_HS': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T_OUT': 64, 'BLOCK_T_IN': 32, 'BLOCK_HS': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T_OUT': 32, 'BLOCK_T_IN': 64, 'BLOCK_HS': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T_OUT': 64, 'BLOCK_T_IN': 64, 'BLOCK_HS': 32}, num_warps=8, num_stages=2),
    ],
    key=['T', 'hs'],
)
@triton.jit
def attn_value_kernel(
    att_ptr, v_ptr, out_ptr,
    B, H, T, hs,
    stride_ab, stride_ah, stride_at, stride_as,
    stride_vb, stride_vh, stride_vt, stride_vhs,
    stride_ob, stride_oh, stride_ot, stride_ohs,
    BLOCK_T_OUT: tl.constexpr,
    BLOCK_T_IN: tl.constexpr,
    BLOCK_HS: tl.constexpr,
):
    # Each program: one (b, h, t_out_tile, hs_tile)
    pid = tl.program_id(0)
    bh_idx = tl.program_id(1)
    hs_tile = tl.program_id(2)

    b_idx = bh_idx // H
    h_idx = bh_idx % H

    t_out_offs = pid * BLOCK_T_OUT + tl.arange(0, BLOCK_T_OUT)
    hs_offs = hs_tile * BLOCK_HS + tl.arange(0, BLOCK_HS)
    t_in_offs = tl.arange(0, BLOCK_T_IN)

    acc = tl.zeros((BLOCK_T_OUT, BLOCK_HS), dtype=tl.float32)

    att_base = att_ptr + b_idx * stride_ab + h_idx * stride_ah
    v_base = v_ptr + b_idx * stride_vb + h_idx * stride_vh

    for t_in_start in range(0, tl.cdiv(T, BLOCK_T_IN)):
        t_in_idx = t_in_start * BLOCK_T_IN + t_in_offs
        t_in_mask = t_in_idx < T

        # Load att tile [BLOCK_T_OUT, BLOCK_T_IN]
        att_ptrs = att_base + t_out_offs[:, None] * stride_at + t_in_idx[None, :] * stride_as
        att_tile = tl.load(att_ptrs,
                           mask=(t_out_offs[:, None] < T) & t_in_mask[None, :],
                           other=0.0)

        # Load V tile [BLOCK_T_IN, BLOCK_HS]
        v_ptrs = v_base + t_in_idx[:, None] * stride_vt + hs_offs[None, :] * stride_vhs
        v_tile = tl.load(v_ptrs,
                         mask=t_in_mask[:, None] & (hs_offs[None, :] < hs),
                         other=0.0)

        acc = tl.dot(att_tile, v_tile, acc, allow_tf32=False)

    out_base = out_ptr + b_idx * stride_ob + h_idx * stride_oh
    out_ptrs = out_base + t_out_offs[:, None] * stride_ot + hs_offs[None, :] * stride_ohs
    tl.store(out_ptrs,
             acc.to(tl.float32),
             mask=(t_out_offs[:, None] < T) & (hs_offs[None, :] < hs))


# ─── GELU kernel ─────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(x_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    c = 0.7978845608028654  # sqrt(2/pi)
    x3 = x * x * x
    inner = c * (x + 0.044715 * x3)
    tanh_val = tl.math.tanh(inner)
    out = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


# ─── Add kernel ──────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def add_kernel(a_ptr, b_ptr, out_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


def run_layernorm(x, weight, bias, eps=1e-5):
    B, T, C = x.shape
    M = B * T
    out = torch.empty_like(x)
    grid = (M,)
    layernorm_kernel[grid](
        x.reshape(M, C), weight, bias, out.reshape(M, C),
        M, C, eps,
    )
    return out


def run_gemm(a, b, bias=None):
    # a: [M, K], b: [N, K] (weight matrix), output: [M, N]
    M, K = a.shape
    N = b.shape[0]
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_bias_kernel[grid](
        c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(1), b.stride(0),  # b is [N, K], so stride_bk=b.stride(1), stride_bn=b.stride(0)
        c.stride(0), c.stride(1),
        HAS_BIAS=(bias is not None),
    )
    return c


def kernel_function(
    p_ln_1_weight, p_ln_1_bias,
    p_attn_c_attn_weight, p_attn_c_attn_bias,
    p_attn_c_proj_weight, p_attn_c_proj_bias,
    p_ln_2_weight, p_ln_2_bias,
    p_mlp_c_fc_weight, p_mlp_c_fc_bias,
    p_mlp_c_proj_weight, p_mlp_c_proj_bias,
    b_attn_bias,
    x,
):
    B, T, C = x.shape  # 128, 512, 768
    n_head = 8
    hs = C // n_head  # 96

    # ── 1. LayerNorm 1 ──────────────────────────────────────────────────────
    ln1_out = run_layernorm(x, p_ln_1_weight, p_ln_1_bias)

    # ── 2. QKV projection ───────────────────────────────────────────────────
    # ln1_out: [B*T, C], weight: [2304, 768]
    ln1_flat = ln1_out.reshape(B * T, C)
    qkv = run_gemm(ln1_flat, p_attn_c_attn_weight, p_attn_c_attn_bias)
    # qkv: [B*T, 2304]
    qkv = qkv.reshape(B, T, 3 * C)

    # Split Q, K, V
    q = qkv[:, :, :C].reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()
    k = qkv[:, :, C:2*C].reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()
    v = qkv[:, :, 2*C:].reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()
    # q, k, v: [B, H, T, hs]

    # ── 3. Attention scores + softmax ───────────────────────────────────────
    scale = 1.0 / math.sqrt(hs)
    att = torch.empty((B, n_head, T, T), device=x.device, dtype=torch.float32)

    # bias: [1, 1, 1024, 1024] -> use [:, :, :T, :T]
    bias_slice = b_attn_bias[0, 0, :T, :T].contiguous()  # [T, T]

    grid_attn = (B * n_head, T)
    attn_score_softmax_kernel[grid_attn](
        q, k, bias_slice, att,
        B, n_head, T, hs,
        scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        att.stride(0), att.stride(1), att.stride(2), att.stride(3),
    )

    # ── 4. att @ V ──────────────────────────────────────────────────────────
    attn_out = torch.empty((B, n_head, T, hs), device=x.device, dtype=torch.float32)
    n_t_tiles = triton.cdiv(T, 32)  # will be overridden by autotune
    grid_av = lambda meta: (
        triton.cdiv(T, meta['BLOCK_T_OUT']),
        B * n_head,
        triton.cdiv(hs, meta['BLOCK_HS']),
    )
    attn_value_kernel[grid_av](
        att, v, attn_out,
        B, n_head, T, hs,
        att.stride(0), att.stride(1), att.stride(2), att.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
    )

    # ── 5. Reshape and project ───────────────────────────────────────────────
    y = attn_out.permute(0, 2, 1, 3).contiguous().reshape(B * T, C)
    proj_out = run_gemm(y, p_attn_c_proj_weight, p_attn_c_proj_bias)
    proj_out = proj_out.reshape(B, T, C)

    # ── 6. Residual 1 ────────────────────────────────────────────────────────
    N_res = B * T * C
    x1 = torch.empty_like(x)
    grid_add = lambda meta: (triton.cdiv(N_res, meta['BLOCK_SIZE']),)
    add_kernel[grid_add](x.reshape(-1), proj_out.reshape(-1), x1.reshape(-1), N_res)

    # ── 7. LayerNorm 2 ───────────────────────────────────────────────────────
    ln2_out = run_layernorm(x1, p_ln_2_weight, p_ln_2_bias)

    # ── 8. MLP FC ────────────────────────────────────────────────────────────
    ln2_flat = ln2_out.reshape(B * T, C)
    fc_out = run_gemm(ln2_flat, p_mlp_c_fc_weight, p_mlp_c_fc_bias)
    # fc_out: [B*T, 3072]

    # ── 9. GELU ──────────────────────────────────────────────────────────────
    N_fc = B * T * 3072
    gelu_out = torch.empty_like(fc_out)
    grid_gelu = lambda meta: (triton.cdiv(N_fc, meta['BLOCK_SIZE']),)
    gelu_kernel[grid_gelu](fc_out.reshape(-1), gelu_out.reshape(-1), N_fc)

    # ── 10. MLP Proj ─────────────────────────────────────────────────────────
    proj2_out = run_gemm(gelu_out, p_mlp_c_proj_weight, p_mlp_c_proj_bias)
    proj2_out = proj2_out.reshape(B, T, C)

    # ── 11. Residual 2 ───────────────────────────────────────────────────────
    x2 = torch.empty_like(x1)
    add_kernel[grid_add](x1.reshape(-1), proj2_out.reshape(-1), x2.reshape(-1), N_res)

    return x2


def get_inputs():
    torch.manual_seed(42)
    B, T, C = 128, 512, 768
    n_head = 8
    max_seqlen = 1024

    p_ln_1_weight = torch.randn(C, device='cuda')
    p_ln_1_bias = torch.randn(C, device='cuda')
    p_attn_c_attn_weight = torch.randn(3 * C, C, device='cuda') * 0.02
    p_attn_c_attn_bias = torch.randn(3 * C, device='cuda') * 0.02
    p_attn_c_proj_weight = torch.randn(C, C, device='cuda') * 0.02
    p_attn_c_proj_bias = torch.randn(C, device='cuda') * 0.02
    p_ln_2_weight = torch.randn(C, device='cuda')
    p_ln_2_bias = torch.randn(C, device='cuda')
    p_mlp_c_fc_weight = torch.randn(4 * C, C, device='cuda') * 0.02
    p_mlp_c_fc_bias = torch.randn(4 * C, device='cuda') * 0.02
    p_mlp_c_proj_weight = torch.randn(C, 4 * C, device='cuda') * 0.02