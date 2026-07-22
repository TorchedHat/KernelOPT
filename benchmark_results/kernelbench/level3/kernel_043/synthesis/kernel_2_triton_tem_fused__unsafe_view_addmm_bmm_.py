import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: QKV projection  (x @ W_qkv.T + b_qkv)
#   x: [B*T, C]  W: [3C, C]  -> out: [B*T, 3C]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def qkv_proj_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile [BLOCK_K, BLOCK_N]  (W is [N, K], we want W.T so row=k, col=n)
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    b_offs = offs_n
    b_mask = offs_n < N
    bias = tl.load(B_ptr + b_offs, mask=b_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Attention scores  Q @ K.T * scale + mask -> softmax
#   Q: [B*nh, T, hs]  K: [B*nh, T, hs]  -> att: [B*nh, T, T]
#   One program per (batch_head, query_row)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64,  'BLOCK_HS': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 128, 'BLOCK_HS': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 64,  'BLOCK_HS': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 128, 'BLOCK_HS': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_T': 256, 'BLOCK_HS': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_T': 512, 'BLOCK_HS': 32}, num_stages=2, num_warps=8),
    ],
    key=['T', 'HS'],
)
@triton.jit
def attn_score_softmax_kernel(
    Q_ptr, K_ptr, Bias_ptr, Out_ptr,
    B_nh, T, HS,
    stride_qb, stride_qt, stride_qh,
    stride_kb, stride_kt, stride_kh,
    stride_ob, stride_ot, stride_ok,
    scale,
    BLOCK_T: tl.constexpr, BLOCK_HS: tl.constexpr,
):
    """
    Computes att[bh, qt, :] = softmax(Q[bh,qt,:] @ K[bh,:,:].T * scale + causal_mask)
    """
    bh_idx = tl.program_id(0)
    qt_idx = tl.program_id(1)

    if bh_idx >= B_nh or qt_idx >= T:
        return

    # Q row: [HS]
    q_base = Q_ptr + bh_idx * stride_qb + qt_idx * stride_qt
    # K matrix: [T, HS]
    k_base = K_ptr + bh_idx * stride_kb

    # We compute dot products with all K rows up to T
    # For numerical stability, do online softmax
    # First pass: compute all scores, find max, compute softmax

    # We'll tile over T (key dimension) and HS (head size)
    # For each key token kt, score[kt] = sum_h Q[qt,h] * K[kt,h] * scale

    # Initialize running max and sum for online softmax
    running_max = tl.full((), float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros((), dtype=tl.float32)

    # We need to store all scores to normalize later
    # Strategy: two-pass over key tokens
    # Pass 1: compute scores and find max
    # Pass 2: compute exp and normalize

    # Allocate output row pointer
    out_base = Out_ptr + bh_idx * stride_ob + qt_idx * stride_ot

    # Pass 1: compute scores for all key positions, store raw scores, find max
    # We process key tokens in blocks of BLOCK_T
    num_kt_blocks = tl.cdiv(T, BLOCK_T)

    # First, compute all scores and find global max
    # We'll do this in two passes over key tokens

    # --- Pass 1: find max score ---
    row_max = tl.full((), float('-inf'), dtype=tl.float32)

    for kt_block in range(0, num_kt_blocks):
        kt_offs = kt_block * BLOCK_T + tl.arange(0, BLOCK_T)
        kt_mask = kt_offs < T

        # Compute score[kt] = Q[qt,:] . K[kt,:] * scale
        # Tile over HS
        score = tl.zeros([BLOCK_T], dtype=tl.float32)
        for hs_block in range(0, tl.cdiv(HS, BLOCK_HS)):
            hs_offs = hs_block * BLOCK_HS + tl.arange(0, BLOCK_HS)
            hs_mask = hs_offs < HS

            # Load Q[qt, hs_offs]: shape [BLOCK_HS]
            q_ptrs = q_base + hs_offs * stride_qh
            q_vals = tl.load(q_ptrs, mask=hs_mask, other=0.0)  # [BLOCK_HS]

            # Load K[kt_offs, hs_offs]: shape [BLOCK_T, BLOCK_HS]
            k_ptrs = k_base + kt_offs[:, None] * stride_kt + hs_offs[None, :] * stride_kh
            k_mask2 = kt_mask[:, None] & hs_mask[None, :]
            k_vals = tl.load(k_ptrs, mask=k_mask2, other=0.0)  # [BLOCK_T, BLOCK_HS]

            # score += K @ q  (dot product for each kt)
            score = score + tl.sum(k_vals * q_vals[None, :], axis=1)

        score = score * scale

        # Apply causal mask: positions where kt > qt get -inf
        # bias[0,0,:T,:T] == 0 means upper triangle -> fill with -inf
        # causal: att[qt, kt] is valid only if kt <= qt
        causal_mask = kt_offs <= qt_idx
        score = tl.where(causal_mask & kt_mask, score, float('-inf'))

        # Update max
        block_max = tl.max(score, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # --- Pass 2: compute exp(score - max) and sum, store normalized ---
    row_sum = tl.zeros((), dtype=tl.float32)

    for kt_block in range(0, num_kt_blocks):
        kt_offs = kt_block * BLOCK_T + tl.arange(0, BLOCK_T)
        kt_mask = kt_offs < T

        score = tl.zeros([BLOCK_T], dtype=tl.float32)
        for hs_block in range(0, tl.cdiv(HS, BLOCK_HS)):
            hs_offs = hs_block * BLOCK_HS + tl.arange(0, BLOCK_HS)
            hs_mask = hs_offs < HS

            q_ptrs = q_base + hs_offs * stride_qh
            q_vals = tl.load(q_ptrs, mask=hs_mask, other=0.0)

            k_ptrs = k_base + kt_offs[:, None] * stride_kt + hs_offs[None, :] * stride_kh
            k_mask2 = kt_mask[:, None] & hs_mask[None, :]
            k_vals = tl.load(k_ptrs, mask=k_mask2, other=0.0)

            score = score + tl.sum(k_vals * q_vals[None, :], axis=1)

        score = score * scale
        causal_mask = kt_offs <= qt_idx
        score = tl.where(causal_mask & kt_mask, score, float('-inf'))

        exp_score = tl.exp(score - row_max)
        exp_score = tl.where(kt_mask, exp_score, 0.0)
        row_sum = row_sum + tl.sum(exp_score, axis=0)

        # Store raw exp scores temporarily (we'll normalize in pass 3)
        out_ptrs = out_base + kt_offs * stride_ok
        tl.store(out_ptrs, exp_score, mask=kt_mask)

    # --- Pass 3: normalize ---
    for kt_block in range(0, num_kt_blocks):
        kt_offs = kt_block * BLOCK_T + tl.arange(0, BLOCK_T)
        kt_mask = kt_offs < T

        out_ptrs = out_base + kt_offs * stride_ok
        exp_score = tl.load(out_ptrs, mask=kt_mask, other=0.0)
        normalized = exp_score / row_sum
        tl.store(out_ptrs, normalized, mask=kt_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: att @ V  ->  y
#   att: [B*nh, T, T]  V: [B*nh, T, hs]  -> y: [B*nh, T, hs]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=2, num_warps=8),
    ],
    key=['T', 'HS', 'BNH'],
)
@triton.jit
def attn_value_kernel(
    Att_ptr, V_ptr, Out_ptr,
    BNH, T, HS,
    stride_ab, stride_at, stride_ak,
    stride_vb, stride_vt, stride_vh,
    stride_ob, stride_ot, stride_oh,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Computes y[bh, qt, :] = att[bh, qt, :] @ V[bh, :, :]
    """
    bh_idx = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # query tokens
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # head size dims
    offs_k = tl.arange(0, BLOCK_K)                     # key tokens

    att_base = Att_ptr + bh_idx * stride_ab
    v_base = V_ptr + bh_idx * stride_vb
    out_base = Out_ptr + bh_idx * stride_ob

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(T, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        # Load att tile [BLOCK_M, BLOCK_K]
        att_ptrs = att_base + offs_m[:, None] * stride_at + k_offs[None, :] * stride_ak
        att_mask = (offs_m[:, None] < T) & (k_offs[None, :] < T)
        att_tile = tl.load(att_ptrs, mask=att_mask, other=0.0)

        # Load V tile [BLOCK_K, BLOCK_N]
        v_ptrs = v_base + k_offs[:, None] * stride_vt + offs_n[None, :] * stride_vh
        v_mask = (k_offs[:, None] < T) & (offs_n[None, :] < HS)
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc = tl.dot(att_tile, v_tile, acc, allow_tf32=False)

    # Store
    out_ptrs = out_base + offs_m[:, None] * stride_ot + offs_n[None, :] * stride_oh
    out_mask = (offs_m[:, None] < T) & (offs_n[None, :] < HS)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 4: Output projection  y @ W_proj.T + b_proj
#   y: [B*T, C]  W: [C, C]  -> out: [B*T, C]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def proj_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(p_c_attn_weight, p_c_attn_bias, p_c_proj_weight, p_c_proj_bias, b_bias, x):
    """
    """
    B, T, C = x.shape
    n_head = 8
    hs = C // n_head  # 96
    scale = 1.0 / math.sqrt(hs)

    # Flatten x for matmul
    x_flat = x.reshape(B * T, C).contiguous()

    # ── Step 1: QKV projection ──────────────────────────────────────────────
    qkv = torch.empty(B * T, 3 * C, device=x.device, dtype=torch.float32)
    M_qkv = B * T
    N_qkv = 3 * C
    K_qkv = C
    grid_qkv = (triton.cdiv(M_qkv, 128), triton.cdiv(N_qkv, 128))
    qkv_proj_kernel[grid_qkv](
        x_flat, p_c_attn_weight, p_c_attn_bias, qkv,
        M_qkv, N_qkv, K_qkv,
        x_flat.stride(0), x_flat.stride(1),
        p_c_attn_weight.stride(0), p_c_attn_weight.stride(1),
        qkv.stride(0), qkv.stride(1),
    )

    # Reshape QKV: [B*T, 3C] -> [B, T, 3C] -> split -> [B, T, C] each
    qkv_3d = qkv.reshape(B, T, 3 * C)
    q_raw = qkv_3d[:, :, :C].contiguous()        # [B, T, C]
    k_raw = qkv_3d[:, :, C:2*C].contiguous()     # [B, T, C]
    v_raw = qkv_3d[:, :, 2*C:].contiguous()      # [B, T, C]

    # Reshape to [B, T, nh, hs] then transpose to [B, nh, T, hs]
    q = q_raw.reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()  # [B, nh, T, hs]
    k = k_raw.reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()  # [B, nh, T, hs]
    v = v_raw.reshape(B, T, n_head, hs).permute(0, 2, 1, 3).contiguous()  # [B, nh, T, hs]

    # Flatten batch and heads: [B*nh, T, hs]
    BNH = B * n_head
    q_flat = q.reshape(BNH, T, hs)
    k_flat = k.reshape(BNH, T, hs)
    v_flat = v.reshape(BNH, T, hs)

    # ── Step 2: Attention scores + softmax ──────────────────────────────────
    att = torch.empty(BNH, T, T, device=x.device, dtype=torch.float32)
    grid_att = (BNH, T)
    attn_score_softmax_kernel[grid_att](
        q_flat, k_flat, b_bias, att,
        BNH, T, hs,
        q_flat.stride(0), q_flat.stride(1), q_flat.stride(2),
        k_flat.stride(0), k_flat.stride(1), k_flat.stride(2),
        att.stride(0), att.stride(1), att.stride(2),
        scale,
    )

    # ── Step 3: att @ V ─────────────────────────────────────────────────────
    y_att = torch.empty(BNH, T, hs, device=x.device, dtype=torch.float32)
    grid_av = (BNH, triton.cdiv(T, 64), triton.cdiv(hs, 64))
    attn_value_kernel[grid_av](
        att, v_flat, y_att,
        BNH, T, hs,
        att.stride(0), att.stride(1), att.stride(2),
        v_flat.stride(0), v_flat.stride(1), v_flat.stride(2),
        y_att.stride(0), y_att.stride(1), y_att.stride(2),
    )

    # Reshape back: [B*nh, T, hs] -> [B, nh, T, hs] -> [B, T, nh, hs] -> [B, T, C]
    y_reshaped = y_att.reshape(B, n_head, T, hs).permute(0, 2, 1, 3).contiguous().reshape(B * T, C)

    # ── Step 4: Output projection ────────────────────────────────────────────
    out = torch.empty(B * T, C, device=x.device, dtype=torch.float32)
    M_proj = B * T
    N_proj = C
    K_proj = C
    grid_proj = (triton.cdiv(M_proj, 128), triton.cdiv(N_proj, 128))
    proj_kernel[grid_proj](
        y_reshaped, p_c_proj_weight, p_c_proj_bias, out,
        M_proj, N_proj, K_proj,
        y_reshaped.stride(0), y_reshaped.stride(1),
        p_c_proj_weight.stride(0), p_c_proj_weight.stride(1),
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
    b_bias = torch.tril(torch.ones(max_seqlen, max_seqlen, device='cuda')).view(1, 1, max_seqlen, max_seqlen)
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
        hs = C // n_head
        scale = 1.0 / math.sqrt(hs)

        # Reference implementation
        qkv_ref = x.reshape(B * T, C) @ p_c_attn_weight.T + p_c_attn_bias
        qkv_ref = qkv_ref.reshape(B, T, 3 * C)
        q_ref = qkv_ref[:, :, :C].reshape(B, T, n_head, hs).permute(0, 2, 1, 3)
        k_ref = qkv_ref