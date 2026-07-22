import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: Permute BCHW -> SBC (contiguous copy) ───────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Y': 32, 'BLOCK_X': 32}, num_warps=4),
        triton.Config({'BLOCK_Y': 16, 'BLOCK_X': 64}, num_warps=4),
        triton.Config({'BLOCK_Y': 64, 'BLOCK_X': 16}, num_warps=4),
        triton.Config({'BLOCK_Y': 32, 'BLOCK_X': 64}, num_warps=8),
    ],
    key=['S', 'B', 'C'],
)
@triton.jit
def permute_bchw_to_sbc_kernel(
    in_ptr, out_ptr,
    S, B, C,
):
    # in: (B, C, S) with strides (C*S, S, 1)
    # out: (S, B, C) contiguous
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    offs_s = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_bc = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_b = offs_bc // C
    offs_c = offs_bc % C
    mask_s = offs_s < S
    mask_bc = offs_bc < B * C
    mask = mask_s[:, None] & mask_bc[None, :]
    # in[b, c, s] = in_ptr[b*C*S + c*S + s]
    in_idx = offs_b[None, :] * (C * S) + offs_c[None, :] * S + offs_s[:, None]
    vals = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    # out[s, b, c] = out_ptr[s*B*C + b*C + c]
    out_idx = offs_s[:, None] * (B * C) + offs_b[None, :] * C + offs_c[None, :]
    tl.store(out_ptr + out_idx, vals, mask=mask)


# ─── Kernel 2: GEMM for in_proj: (S*B, 128) x (128, 384) + bias -> (S*B, 384) ─
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
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
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ─── Kernel 3: Split QKV + scale Q ─────────────────────────────────────────
# Input: qkv (S, B, 384) contiguous
# Output: q_scaled (num_heads, S*B, head_dim), k (num_heads, S*B, head_dim), v (num_heads, S*B, head_dim)
# where num_heads=8 (wait, original says 4 heads but embed_dim=128 -> head_dim=32, but computation graph shows 8 heads)
# Looking at computation graph: view to [16384, 8, 32] -> 8 heads, head_dim=32
# But original model says num_heads=4... Let me check: embed_dim=128, num_heads=4 -> head_dim=32
# But the computation graph shows [16384, 8, 32]... 
# Wait: S=16384, B=2, so S*B=32768. view [16384, 8, 32] means S=16384, heads=8, head_dim=32
# That means num_heads=8 in the actual computation? But model says 4...
# Actually looking more carefully: the model uses num_heads=4 but embed_dim=128
# head_dim = 128/4 = 32. But the view is [16384, 8, 32]... 
# 16384 = S (128*128), 8 = B*num_heads = 2*4, 32 = head_dim
# So the batch dimension is folded into heads! That makes sense for the BMM.
# So effectively: num_heads_total = B * num_heads = 2 * 4 = 8
# And the sequence length for BMM is S = 16384

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=4),
        triton.Config({'BLOCK': 1024}, num_warps=8),
        triton.Config({'BLOCK': 128}, num_warps=2),
    ],
    key=['total'],
)
@triton.jit
def split_qkv_scale_kernel(
    S, B, num_heads, head_dim,
    scale,
):
    # qkv layout: (S*B, 384) = (S*B, 3, num_heads, head_dim)
    # We want q[h, s, d], k[h, s, d], v[h, s, d]
    # where h = batch*num_heads + head, s = seq_pos
    # 
    # qkv[sb, 3, nh, hd] where sb = s*B + b
    # q_out[b*nh + h, s, hd]
    # 
    # Let's index by (h_total, s, d) where h_total = b*num_heads + h
    # h_total in [0, B*num_heads), s in [0, S), d in [0, head_dim)
    # 
    # sb = s * B + b  (b = h_total // num_heads)
    # qkv_idx = sb * 384 + qkv_type * num_heads * head_dim + h * head_dim + d
    #         = (s * B + b) * 384 + qkv_type * 128 + h * head_dim + d
    
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    
    # Decode: offs = h_total * S * head_dim + s * head_dim + d
    d = offs % head_dim
    s = (offs // head_dim) % S
    h_total = offs // (S * head_dim)
    
    h = h_total % num_heads
    b = h_total // num_heads
    
    sb = s * B + b
    base = sb * (3 * num_heads * head_dim)
    
    q_src = base + 0 * num_heads * head_dim + h * head_dim + d
    k_src = base + 1 * num_heads * head_dim + h * head_dim + d
    v_src = base + 2 * num_heads * head_dim + h * head_dim + d
    
    q_val = tl.load(qkv_ptr + q_src, mask=mask, other=0.0)
    k_val = tl.load(qkv_ptr + k_src, mask=mask, other=0.0)
    v_val = tl.load(qkv_ptr + v_src, mask=mask, other=0.0)
    
    q_val = q_val * scale
    
    tl.store(q_ptr + offs, q_val, mask=mask)
    tl.store(k_ptr + offs, k_val, mask=mask)
    tl.store(v_ptr + offs, v_val, mask=mask)


# ─── Kernel 4: Batched attention scores: Q @ K^T ────────────────────────────
# Q: (num_heads_total, S, head_dim), K: (num_heads_total, S, head_dim)
# attn: (num_heads_total, S, S)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['S', 'head_dim', 'num_heads_total'],
)
@triton.jit
def bmm_qk_kernel(
    Q_ptr, K_ptr, Out_ptr,
    S, head_dim, num_heads_total,
):
    # Each program handles one (head, row_tile, col_tile)
    pid_h = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    Q_base = pid_h * S * head_dim
    K_base = pid_h * S * head_dim
    Out_base = pid_h * S * S
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(head_dim, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        q_mask = (offs_m[:, None] < S) & (k_offs[None, :] < head_dim)
        k_mask = (offs_n[:, None] < S) & (k_offs[None, :] < head_dim)
        q = tl.load(Q_ptr + Q_base + offs_m[:, None] * head_dim + k_offs[None, :], mask=q_mask, other=0.0)
        k = tl.load(K_ptr + K_base + offs_n[:, None] * head_dim + k_offs[None, :], mask=k_mask, other=0.0)
        acc = tl.dot(q, tl.trans(k), acc, allow_tf32=False)
    
    out_mask = (offs_m[:, None] < S) & (offs_n[None, :] < S)
    tl.store(Out_ptr + Out_base + offs_m[:, None] * S + offs_n[None, :], acc, mask=out_mask)


# ─── Kernel 5: Softmax over last dim ────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(
    In_ptr, Out_ptr,
    M, N,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    row_ptr = In_ptr + row * N
    x = tl.load(row_ptr + offs, mask=mask, other=float('-inf'))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    x_sum = tl.sum(x_exp, axis=0)
    out = x_exp / x_sum
    tl.store(Out_ptr + row * N + offs, out, mask=mask)


# ─── Kernel 6: Batched attention output: attn_weights @ V ───────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['S', 'head_dim', 'num_heads_total'],
)
@triton.jit
def bmm_av_kernel(
    A_ptr, V_ptr, Out_ptr,
    S, head_dim, num_heads_total,
):
    pid_h = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    A_base = pid_h * S * S
    V_base = pid_h * S * head_dim
    Out_base = pid_h * S * head_dim
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(S, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < S) & (k_offs[None, :] < S)
        v_mask = (k_offs[:, None] < S) & (offs_n[None, :] < head_dim)
        a = tl.load(A_ptr + A_base + offs_m[:, None] * S + k_offs[None, :], mask=a_mask, other=0.0)
        v = tl.load(V_ptr + V_base + k_offs[:, None] * head_dim + offs_n[None, :], mask=v_mask, other=0.0)
        acc = tl.dot(a, v, acc, allow_tf32=False)
    
    out_mask = (offs_m[:, None] < S) & (offs_n[None, :] < head_dim)
    tl.store(Out_ptr + Out_base + offs_m[:, None] * head_dim + offs_n[None, :], acc, mask=out_mask)


# ─── Kernel 7: Merge heads + out_proj GEMM ──────────────────────────────────
# attn_out: (num_heads_total, S, head_dim) -> reshape to (S*B, embed_dim) then GEMM
# But we need to handle the transpose: (num_heads_total, S, head_dim) -> (S, B, num_heads, head_dim) -> (S*B, embed_dim)
# num_heads_total = B * num_heads
# attn_out[h_total, s, d] where h_total = b*num_heads + h
# merged[s, b, h, d] = attn_out[b*num_heads+h, s, d]
# merged_flat[s*B+b, h*head_dim+d] = attn_out[b*num_heads+h, s, d]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 256}, num_warps=4),
        triton.Config({'BLOCK': 512}, num_warps=4),
        triton.Config({'BLOCK': 1024}, num_warps=8),
        triton.Config({'BLOCK': 128}, num_warps=2),
    ],
    key=['total'],
)
@triton.jit
def merge_heads_kernel(
    out_ptr,    # (S*B, embed_dim) = (S*B, num_heads*head_dim)
    S, B, num_heads, head_dim,
    total: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < total
    
    # out[sb, embed] where sb = s*B+b, embed = h*head_dim+d
    embed_dim = num_heads * head_dim
    d = offs % head_dim
    h = (offs // head_dim) % num_heads
    b = (offs // (num_heads * head_dim)) % B
    s = offs // (B * num_heads * head_dim)
    
    h_total = b * num_heads + h
    src_idx = h_total * S * head_dim + s * head_dim + d
    
    sb = s * B + b
    dst_idx = sb * embed_dim + h * head_dim + d
    
    val = tl.load(attn_ptr + src_idx, mask=mask, other=0.0)
    tl.store(out_ptr + dst_idx, val, mask=mask)


# ─── Kernel 8: out_proj GEMM: (S*B, embed_dim) x (embed_dim, embed_dim) + bias ─
# Reuse gemm_bias_kernel


# ─── Kernel 9: Residual add + LayerNorm ─────────────────────────────────────
# x_res: (S*B, embed_dim) = original x permuted
# attn_out: (S*B, embed_dim)
# out: (S*B, embed_dim) = LayerNorm(x_res + attn_out)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 64}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def residual_layernorm_kernel(
    M, N,
    eps,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N
    
    x_res = tl.load(x_res_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    attn = tl.load(attn_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    
    x = x_res + attn
    
    # Compute mean
    mean = tl.sum(x, axis=0) / N
    # Compute variance
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    # Normalize
    x_norm = diff / tl.sqrt(var + eps)
    
    weight = tl.load(weight_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    bias = tl.load(bias_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    
    out = x_norm * weight + bias
    tl.store(out_ptr + row * N + offs, out.to(tl.float32), mask=mask)


# ─── Kernel 10: Permute SBC -> BCHW ─────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Y': 32, 'BLOCK_X': 32}, num_warps=4),
        triton.Config({'BLOCK_Y': 16, 'BLOCK_X': 64}, num_warps=4),
        triton.Config({'BLOCK_Y': 64, 'BLOCK_X': 16}, num_warps=4),
        triton.Config({'BLOCK_Y': 32, 'BLOCK_X': 64}, num_warps=8),
    ],
    key=['S', 'B', 'C'],
)
@triton.jit
def permute_sbc_to_bchw_kernel(
    in_ptr, out_ptr,
    S, B, C,
):
    # in: (S, B, C) contiguous
    # out: (B, C, S) -> view as (B, C, H, W)
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    offs_s = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)
    offs_bc = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)
    offs_b = offs_bc // C
    offs_c = offs_bc % C
    mask_s = offs_s < S
    mask_bc = offs_bc < B * C
    mask = mask_s[:, None] & mask_bc[None, :]
    # in[s, b, c] = in_ptr[s*B*C + b*C + c]
    in_idx = offs_s[:, None] * (B * C) + offs_b[None, :] * C + offs_c[None, :]
    vals = tl.load(in_ptr + in_idx, mask=mask, other=0.0)
    # out[b, c, s] = out_ptr[b*C*S + c*S + s]
    out_idx = offs_b[None, :] * (C * S) + offs_c[None, :] * S + offs_s[:, None]
    tl.store(out_ptr + out_idx, vals, mask=mask)


def kernel_function(
):
    B, C, H, W = x.shape
    S = H * W          # 16384
    embed_dim = C      # 128
    num_heads = 4      # from model
    head_dim = embed_dim // num_heads  # 32
    num_heads_total = B * num_heads    # 8
    scale = 1.0 / math.sqrt(head_dim)  # 0.1767...
    
    device = x.device
    dtype = x.dtype
    
    # Step 1: Permute x from (B, C, H*W) to (S, B, C) contiguous
    x_sbc = torch.empty((S, B, C), device=device, dtype=dtype)
    grid_perm = (triton.cdiv(S, 32), triton.cdiv(B * C, 32))
    permute_bchw_to_sbc_kernel[grid_perm](
        x, x_sbc,
    )
    
    # Step 2: in_proj GEMM: (S*B, C) x (C, 3*C) + bias -> (S*B, 3*C)
    x_flat = x_sbc.reshape(S * B, C)  # (32768, 128)
    qkv = torch.empty((S * B, 3 * embed_dim), device=device, dtype=dtype)
    grid_gemm1 = (triton.cdiv(S * B, 64), triton.cdiv(3 * embed_dim, 128))
    gemm_bias_kernel[grid_gemm1](
        x_flat, p_attn_in_proj_weight.t().contiguous(), p_attn_in_proj_bias, qkv,
        S * B, 3 * embed_dim, embed_dim,
        embed_dim, 1,
        1, 3 * embed_dim,
    )
    # qkv shape: (S*B, 384) = (S*B, 3, num_heads, head_dim) logically
    # But we need to split into Q, K, V with shape (num_heads_total, S, head_dim)
    
    # Step 3: Split QKV
    # qkv is (S*B, 384). We need to interpret as (S, B, 3, num_heads, head_dim)
    # and produce q/k/v of shape (num_heads_total, S, head_dim) = (B*num_heads, S, head_dim)
    total_qkv = num_heads_total * S * head_dim
    q_scaled = torch.empty((num_heads_total, S, head_dim), device=device, dtype=dtype)
    k_buf = torch.empty((num_heads_total, S, head_dim), device=device, dtype=dtype)
    v_buf = torch.empty((num_heads_total, S, head_dim), device=device, dtype=dtype)
    
    grid_split = (triton.cdiv(total_qkv, 256