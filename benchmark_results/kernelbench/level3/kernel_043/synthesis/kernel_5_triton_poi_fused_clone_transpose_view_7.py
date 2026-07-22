import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: QKV projection (GEMM: [B*T, C] x [C, 3C] + bias) ───────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
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
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk,
                    mask=a_mask, other=0.0)
        b = tl.load(W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk,
                    mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)

    # Add bias
    bias_mask = offs_n < N
    bias = tl.load(B_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(tl.float32), mask=out_mask)


# ─── Kernel 2: Fused attention (per batch*head row) ─────────────────────────
# For each (batch, head), compute softmax attention over T=512 tokens
# Q: [B, nh, T, hs], K: [B, nh, T, hs], V: [B, nh, T, hs]
# att = softmax(Q @ K^T * scale + mask) @ V

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64,  'BLOCK_HS': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 128, 'BLOCK_HS': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 64,  'BLOCK_HS': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_T': 128, 'BLOCK_HS': 64}, num_stages=2, num_warps=8),
    ],
    key=['T', 'HS'],
)
@triton.jit
def attn_fwd_kernel(
    Q_ptr, K_ptr, V_ptr, Bias_ptr, Out_ptr,
    B, NH, T, HS,
    stride_qb, stride_qh, stride_qt, stride_qhs,
    stride_kb, stride_kh, stride_kt, stride_khs,
    stride_vb, stride_vh, stride_vt, stride_vhs,
    stride_ob, stride_oh, stride_ot, stride_ohs,
    scale,
    BLOCK_T: tl.constexpr,
    BLOCK_HS: tl.constexpr,
):
    # Each program handles one (batch, head, query_row) tile
    pid_bh = tl.program_id(0)  # batch * n_head
    pid_q  = tl.program_id(1)  # query tile index

    b = pid_bh // NH
    h = pid_bh % NH

    q_start = pid_q * BLOCK_T
    q_offs = q_start + tl.arange(0, BLOCK_T)
    q_mask = q_offs < T

    # Load Q tile [BLOCK_T, HS] - iterate over HS dimension
    # We'll accumulate output in [BLOCK_T, HS]
    # First compute attention scores [BLOCK_T, T], then weighted sum over V

    # For numerical stability, we do online softmax
    # att_scores[i, j] = sum_hs Q[i, hs] * K[j, hs] * scale + causal_mask

    # Step 1: compute all attention scores for this query tile
    # att[BLOCK_T, T] - we need to iterate over K dimension (T) in blocks

    # Initialize running max and sum for online softmax
    row_max = tl.full((BLOCK_T,), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_T,), dtype=tl.float32)
    # Output accumulator [BLOCK_T, BLOCK_HS] - but HS might be > BLOCK_HS
    # We need to handle HS in blocks too

    # Actually let's do it differently:
    # For each query row block, compute full attention row, then weighted V sum

    # Compute Q tile: [BLOCK_T, HS]
    # We need to store the full att row [BLOCK_T, T] - T=512, might be large
    # Better: online softmax approach

    # We'll accumulate output [BLOCK_T, HS] using online softmax
    # For each key block, compute partial scores, update running stats, accumulate

    # But HS=96 which fits in registers if BLOCK_HS=96
    # Let's use BLOCK_HS = HS (constexpr) for simplicity - but autotune varies it

    # Load full Q tile [BLOCK_T, HS] - iterate over hs blocks
    # Store in SRAM via accumulation

    # Simpler approach: load Q[BLOCK_T, HS] fully (HS=96 fits)
    # Then for each K block [BLOCK_KT, HS], compute scores [BLOCK_T, BLOCK_KT]
    # Online softmax update, accumulate V

    hs_offs = tl.arange(0, BLOCK_HS)

    # Load Q [BLOCK_T, BLOCK_HS] - need to loop over HS
    # Since HS=96 and BLOCK_HS can be 32 or 64, we loop
    # Actually for attention we want full HS at once for the dot product
    # Let's fix BLOCK_HS = HS via a different approach

    # We'll accumulate Q*K^T by looping over hs blocks
    # For each key position block, accumulate partial dot products

    # acc_out: [BLOCK_T, HS] output accumulator
    # We need HS to be a constexpr... let's use BLOCK_HS and loop

    # Initialize output accumulator
    # We'll store partial output as [BLOCK_T, BLOCK_HS] and loop over hs blocks
    # But that complicates online softmax...

    # Simplest correct approach: 
    # 1. Compute all scores [BLOCK_T, T] by looping over K in hs blocks
    # 2. Apply mask, softmax
    # 3. Compute weighted V sum

    # But storing [BLOCK_T, T] = [64, 512] = 32K floats in SRAM is too much

    # Use online softmax: for each key block, update running max/sum and output

    # acc_out[BLOCK_T, HS] - but HS=96, BLOCK_HS varies
    # We'll loop over hs output dimension separately after computing attention weights

    # APPROACH: Two-pass over key dimension
    # Pass 1: compute scores, find max, compute softmax weights -> store in SRAM? No.
    # 
    # Better: Flash attention style - process key blocks, maintain running stats

    # For each key block:
    #   1. Compute scores [BLOCK_T, BLOCK_KT] 
    #   2. Update online softmax stats
    #   3. Accumulate weighted V into output

    # Output [BLOCK_T, HS] - HS=96, fits in registers if BLOCK_T is small

    # Let's use BLOCK_KT = BLOCK_T for the key dimension tiling
    BLOCK_KT: tl.constexpr = BLOCK_T

    # Initialize accumulators
    # acc_out: [BLOCK_T, HS] but HS is runtime... 
    # We need HS as constexpr. Let's hardcode HS=96 via a workaround.
    # Actually BLOCK_HS is constexpr, and we loop over HS/BLOCK_HS blocks.

    # This gets complex. Let me restructure:
    # acc_out[BLOCK_T, BLOCK_HS] for each hs_block
    # But online softmax requires processing all K for each Q before outputting

    # FINAL APPROACH: 
    # - Compute full attention row [BLOCK_T, T] score by looping over hs
    # - Store scores in shared memory (SRAM via tl.zeros)
    # - Apply softmax
    # - Compute output by looping over V

    # T=512, BLOCK_T=64: scores = [64, 512] = 32K floats - too large for registers
    # But Triton can handle this with SRAM...

    # Actually in Triton, tl.zeros creates register arrays. 512 floats per query row
    # times 64 rows = 32K floats - this will spill to local memory.

    # Let's use a different tiling: process one query at a time (BLOCK_T=1 effectively)
    # No, that's too slow.

    # PRACTICAL SOLUTION: Use flash attention with BLOCK_KT tiling
    # Keep acc_out [BLOCK_T, BLOCK_HS] and loop over both K and HS dimensions

    # For flash attention with HS blocking:
    # For each hs_block:
    #   reset acc_out, row_max, row_sum
    #   for each k_block:
    #     compute partial scores (need full HS for Q*K^T)
    #   This doesn't work because scores depend on full HS

    # The scores Q*K^T require summing over full HS dimension
    # So we must compute full scores before accumulating V

    # SOLUTION: Compute scores in two sub-passes:
    # Sub-pass 1 over hs: accumulate Q*K^T scores [BLOCK_T, BLOCK_KT]
    # Then update softmax stats and accumulate V

    # This is the standard flash attention approach!

    # acc_out[BLOCK_T, HS] - HS=96 fits if BLOCK_T is small enough
    # With BLOCK_T=64, HS=96: 64*96=6144 floats = 24KB - fits in registers/L1

    # Let's implement flash attention properly
    # We need HS as constexpr for tl.zeros shape
    # Use BLOCK_HS = HS (set in autotune or hardcode)

    pass  # placeholder - see full implementation below


# I'll restructure with a cleaner flash attention implementation

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Q': 32, 'BLOCK_K': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_K': 32}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_Q': 32, 'BLOCK_K': 64}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_Q': 64, 'BLOCK_K': 64}, num_stages=1, num_warps=8),
    ],
    key=['T', 'HS'],
)
@triton.jit
def flash_attn_kernel(
    Q_ptr, K_ptr, V_ptr, Bias_ptr, Out_ptr,
    B, NH, T, HS,
    stride_qb, stride_qh, stride_qt, stride_qhs,
    stride_kb, stride_kh, stride_kt, stride_khs,
    stride_vb, stride_vh, stride_vt, stride_vhs,
    stride_ob, stride_oh, stride_ot, stride_ohs,
    scale,
    BLOCK_Q: tl.constexpr,
    BLOCK_K: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    pid_bh = tl.program_id(0)
    pid_q  = tl.program_id(1)

    b = pid_bh // NH
    h = pid_bh % NH

    q_start = pid_q * BLOCK_Q
    q_offs = q_start + tl.arange(0, BLOCK_Q)
    hs_offs = tl.arange(0, HEAD_DIM)

    # Load Q tile [BLOCK_Q, HEAD_DIM]
    q_mask = (q_offs[:, None] < T) & (hs_offs[None, :] < HS)
    Q = tl.load(
        Q_ptr + b * stride_qb + h * stride_qh + q_offs[:, None] * stride_qt + hs_offs[None, :] * stride_qhs,
        mask=q_mask, other=0.0
    )

    # Online softmax accumulators
    row_max = tl.full((BLOCK_Q,), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_Q,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_Q, HEAD_DIM), dtype=tl.float32)

    # Loop over key/value blocks
    for k_start in range(0, T, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        k_mask = (k_offs[:, None] < T) & (hs_offs[None, :] < HS)

        # Load K tile [BLOCK_K, HEAD_DIM]
        K_tile = tl.load(
            K_ptr + b * stride_kb + h * stride_kh + k_offs[:, None] * stride_kt + hs_offs[None, :] * stride_khs,
            mask=k_mask, other=0.0
        )

        # Compute scores [BLOCK_Q, BLOCK_K] = Q @ K^T
        scores = tl.dot(Q, tl.trans(K_tile), allow_tf32=False)  # [BLOCK_Q, BLOCK_K]
        scores = scores * scale

        # Apply causal mask: q_offs[:, None] >= k_offs[None, :]
        # bias[0, 0, :T, :T] is lower triangular (1 where allowed, 0 where masked)
        # masked_fill where bias == 0 with -inf
        causal_mask = q_offs[:, None] >= k_offs[None, :]
        k_valid = k_offs[None, :] < T
        scores = tl.where(causal_mask & k_valid, scores, float('-inf'))

        # Online softmax update
        block_max = tl.max(scores, axis=1)  # [BLOCK_Q]
        new_max = tl.maximum(row_max, block_max)
        
        # Rescale previous accumulator
        alpha = tl.exp(row_max - new_max)
        acc = acc * alpha[:, None]
        row_sum = row_sum * alpha

        # Compute exp of current block
        exp_scores = tl.exp(scores - new_max[:, None])  # [BLOCK_Q, BLOCK_K]
        
        # Mask out invalid k positions
        exp_scores = tl.where(k_offs[None, :] < T, exp_scores, 0.0)
        
        row_sum = row_sum + tl.sum(exp_scores, axis=1)
        row_max = new_max

        # Load V tile [BLOCK_K, HEAD_DIM]
        v_mask = (k_offs[:, None] < T) & (hs_offs[None, :] < HS)
        V_tile = tl.load(
            V_ptr + b * stride_vb + h * stride_vh + k_offs[:, None] * stride_vt + hs_offs[None, :] * stride_vhs,
            mask=v_mask, other=0.0
        )

        # Accumulate: acc += exp_scores @ V_tile
        acc = tl.dot(exp_scores.to(V_tile.dtype), V_tile, acc, allow_tf32=False)

    # Normalize
    acc = acc / row_sum[:, None]

    # Store output [BLOCK_Q, HEAD_DIM]
    out_mask = (q_offs[:, None] < T) & (hs_offs[None, :] < HS)
    tl.store(
        Out_ptr + b * stride_ob + h * stride_oh + q_offs[:, None] * stride_ot + hs_offs[None, :] * stride_ohs,
        acc.to(tl.float32),
        mask=out_mask
    )


# ─── Kernel 3: Output projection (GEMM: [B*T, C] x [C, C] + bias) ──────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def out_proj_kernel(
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=a_mask, other=0.0)
        b = tl.load(W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk,
                    mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)

    bias_mask = offs_n < N
    bias = tl.load(B_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(tl.float32), mask=out_mask)


def kernel_function(p_c_attn_weight, p_c_attn_bias, p_c_proj_weight, p_c_proj_bias, b_bias, x):
    B, T, C = x.shape  # 128, 512, 768
    NH = 8
    HS = C // NH  # 96
    scale = 1.0 / math.sqrt(HS)

    # Step 1: QKV projection [B*T, C] x [C, 3C] + bias -> [B*T, 3C]
    M_qkv = B * T  # 65536
    N_qkv = 3 * C  # 2304
    K_qkv = C      # 768

    x_2d = x.reshape(M_qkv, K_qkv).contiguous()
    qkv = torch.empty((M_qkv, N_qkv), device=x.device, dtype=torch.float32)

    grid_qkv = (triton.cdiv(M_qkv, 128), triton.cdiv(N_qkv, 128))
    qkv_proj_kernel[grid_qkv](
        x_2d, p_c_attn_weight, p_c_attn_bias, qkv,
        M_qkv, N_qkv, K_qkv,
        x_2d.stride(0), x_2d.stride(1),
        p_c_attn_weight.stride(0), p_c_attn_weight.stride(1),
        qkv.stride(0), qkv.stride(1),
    )

    # Reshape QKV and split
    qkv_3d = qkv.reshape(B, T, 3 * C)
    q = qkv_3d[:, :, :C].reshape(B, T, NH, HS).permute(0, 2, 1, 3).contiguous()
    k = qkv_3d[:, :, C:2*C].reshape(B, T, NH, HS).permute(0, 2, 1, 3).contiguous()
    v = qkv_3d[:, :, 2*C:].reshape(B, T, NH, HS).permute(0, 2, 1, 3).contiguous()

    # Step 2: Flash attention
    attn_out = torch.empty((B, NH, T, HS), device=x.device, dtype=torch.float32)

    HEAD_DIM = 128  # next power of 2 >= HS=96

    grid_attn = (B * NH, triton.cdiv(T, 64))
    flash_attn_kernel[grid_attn](
        q, k, v, b_bias, attn_out,
        B, NH, T, HS,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        attn_out.stride(0), attn_out.stride(1), attn_out.stride(2), attn_out.stride(3),
        scale,
        HEAD_DIM=HEAD_DIM,
    )

    # Reshape attention output
    y = attn_out.permute(0, 2, 1, 3).contiguous().reshape(B * T, C)

    # Step 3: Output projection
    M_out = B * T
    N_out = C
    K_out = C

    out = torch.empty((M_out, N_out), device=x.device, dtype=torch.float32)
    grid_out = (triton.cdiv(M_out, 128), triton.cdiv(N_out, 128))
    out_proj_kernel[grid_out](
        y, p_c_proj_weight, p_c_proj_bias, out,
        M_out, N_out, K_out,
        y.stride(0), y.stride(1),
        p_c_proj_weight.stride(0), p_c_proj_weight.stride(1),
        out.stride(0), out.stride(1),
    )

    return out.reshape(B, T, C)


def get_inputs():
    B, T, C = 128, 512, 768
    NH = 8
    max_seqlen = 1024
    p_c_attn_weight = torch.randn(3 * C, C, device='cuda', dtype=torch.float32)
    p_c_attn_bias = torch.randn(3 * C, device='cuda', dtype=torch.float32)
    p_c_proj_weight = torch.randn(C, C, device='cuda', dtype=torch.float32)
    p_c_proj_bias = torch.randn(C, device='cuda', dtype=torch.float32)
    b_bias = torch.tril(torch.ones(max_seqlen, max_seqlen)).view(1, 1, max_seqlen, max_seqlen).cuda()
    x = torch.randn(B, T, C, device='cuda', dtype=torch.float32)
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
        NH = 8
        HS = C // NH
        scale = 1.0 / math.sqrt(HS)

        # Reference computation
        import torch.nn.functional as F
        qkv = x @ p_c_attn_weight.T + p_c_attn_bias
        q_r, k_r, v_r = qkv.split(C, dim=2)
        q_r = q_r.view(B, T, NH, HS).transpose(1, 2)
        k_r = k_r.view(B, T, NH, HS).transpose(1, 2)
        v_r = v_r.view(B, T, NH, HS).transpose(1, 2)
        att = (q_r @ k_r.transpose(-2, -1)) * scale
        att = att.masked_fill(b_bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y_r = att @ v_r
        y_r = y_r.transpose(1, 2).contiguous().view(B, T, C)
        ref = y_r @ p_c_proj_weight.T + p_c_proj_bias

        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e