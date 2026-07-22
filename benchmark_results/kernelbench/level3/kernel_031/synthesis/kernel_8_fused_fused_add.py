import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: permute + copy (NCHW -> seq_len, batch, channels) ───────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_Y': 32, 'BLOCK_X': 32}, num_warps=4),
        triton.Config({'BLOCK_Y': 16, 'BLOCK_X': 64}, num_warps=4),
        triton.Config({'BLOCK_Y': 64, 'BLOCK_X': 16}, num_warps=4),
        triton.Config({'BLOCK_Y': 32, 'BLOCK_X': 64}, num_warps=8),
    ],
    key=['seq_len', 'channels'],
)
@triton.jit
def permute_kernel(
    in_ptr, out_ptr,
    batch, seq_len, channels,
    BLOCK_Y: tl.constexpr, BLOCK_X: tl.constexpr,
):
    # in: (batch, channels, seq_len) strides: (channels*seq_len, seq_len, 1)
    # out: (seq_len, batch, channels) strides: (batch*channels, channels, 1)
    pid_y = tl.program_id(0)
    pid_x = tl.program_id(1)
    offs_y = pid_y * BLOCK_Y + tl.arange(0, BLOCK_Y)  # seq_len dim
    offs_x = pid_x * BLOCK_X + tl.arange(0, BLOCK_X)  # channels dim
    mask_y = offs_y < seq_len
    mask_x = offs_x < channels
    mask = mask_y[:, None] & mask_x[None, :]
    for b in range(batch):
        # in[b, c, s] = in_ptr[b*channels*seq_len + c*seq_len + s]
        # out[s, b, c] = out_ptr[s*batch*channels + b*channels + c]
        in_ptrs = in_ptr + b * channels * seq_len + offs_x[None, :] * seq_len + offs_y[:, None]
        vals = tl.load(in_ptrs, mask=mask, other=0.0)
        out_ptrs = out_ptr + offs_y[:, None] * batch * channels + b * channels + offs_x[None, :]
        tl.store(out_ptrs, vals, mask=mask)


# ─── Kernel 2: GEMM for in_proj: (32768, 128) x (128, 384) + bias ─────────────
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
    stride_cm, stride_cn,
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
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc.to(tl.float32), mask=c_mask)


# ─── Kernel 3: Reshape QKV + scale Q ──────────────────────────────────────────
# Input: qkv_raw (32768, 384) = (seq_len*batch, 3*embed_dim)
# Output Q: (num_heads, seq_len*batch, head_dim) scaled
#         K: (num_heads, seq_len*batch, head_dim)
#         V: (num_heads, seq_len*batch, head_dim)
# The layout after in_proj is: for each (s,b) pair, we have [q0..q127, k0..k127, v0..v127]
# But the actual layout from the computation graph is more complex due to the permute/reshape
# Let me trace carefully:
# After mm: (32768, 384) where 32768 = 16384*2 (seq_len * batch)
# reshape to (16384, 2, 384) - so dim0=seq, dim1=batch, dim2=3*embed
# add bias (384,)
# reshape to (16384, 2, 3, 128) - split last dim into 3 heads groups
# permute [3,1,2,0,4] on unsqueezed (1,16384,2,3,128) -> (3,16384,2,1,128)
# squeeze -> (3, 16384, 2, 128)
# clone -> contiguous (3, 16384, 2, 128)
# select 0 -> Q: (16384, 2, 128), select 1 -> K, select 2 -> V
# reshape Q to (16384, 8, 32), permute to (8, 16384, 32), scale by 1/sqrt(32)
# reshape K to (16384, 8, 32), permute to (8, 16384, 32)
# reshape V to (16384, 8, 32), permute to (8, 16384, 32)
# So Q[s,b,c] = qkv_raw[s*2+b, c] (first 128 channels)
# K[s,b,c] = qkv_raw[s*2+b, 128+c]
# V[s,b,c] = qkv_raw[s*2+b, 256+c]
# After reshape to (16384, 8, 32): Q[s, h, d] = Q[s, b, h*32+d] ... wait
# Q has shape (16384, 2, 128) - seq_len=16384, batch=2, embed=128
# reshape to (16384, 8, 32) - this merges batch and embed? No...
# Wait: (16384, 2, 128) reshaped to (16384, 8, 32) means 2*128=256=8*32 ✓
# So Q[s, h, d] where h in [0,8), d in [0,32)
# The original Q[s, b, c] maps to Q[s, b*4 + c//32, c%32] (since 128/32=4 heads per batch)
# Then permute (1,0,2) -> (8, 16384, 32)
# For attention: bmm Q_scaled (8, 16384, 32) x K^T (8, 32, 16384) -> (8, 16384, 16384)
# Then softmax over last dim
# Then bmm attn (8, 16384, 16384) x V (8, 16384, 32) -> (8, 16384, 32)
# transpose (1,0,2) -> (16384, 8, 32), contiguous, reshape to (32768, 128)
# out_proj: (32768, 128) x (128, 128) + bias -> (32768, 128)
# reshape to (16384, 2, 128)
# view as (2, 4, 16384, 16384) - wait this doesn't make sense dimensionally
# (16384, 2, 128) has 16384*2*128 = 4194304 elements
# (2, 4, 16384, 16384) has 2*4*16384*16384 = 2147483648 elements - that's way too big!
# Let me re-read... oh wait, the mean.dim is over [2, 4, 16384, 16384]
# But that's 2*4*16384^2 elements which is enormous. That can't be right.
# Let me re-read the computation graph more carefully.
# 
# Actually looking again at the graph:
# view.default -> shape=[2, 4, 16384, 16384] - this seems wrong for the data size
# But wait - the attn_weights from bmm are (8, 16384, 16384) = 8*16384^2 elements
# Reshape to (2, 4, 16384, 16384) makes sense: 2*4=8 ✓
# mean.dim -> shape=[2, 16384, 16384] - mean over heads dim (dim=1)
# But this is the attention weights, not the output!
# 
# Wait, looking at the original PyTorch code:
# attn_output, _ = self.attn(x, x, x)
# The _ is the attention weights. In nn.MultiheadAttention, the second return is
# the averaged attention weights. So the computation graph includes computing
# the average attention weights even though they're not used in the output!
# 
# The actual output path is:
# attn_output (16384, 2, 128) + x (16384, 2, 128) -> add
# layer_norm -> (16384, 2, 128)
# permute (1,2,0) -> (2, 128, 16384)
# view -> (2, 128, 128, 128)
#
# The mean.dim is computing the average attention weights (returned as second output
# of MultiheadAttention) but it feeds into add.Tensor with x (residual connection)
# Wait no - let me re-read:
# add.Tensor -> shape=[16384, 2, 128] - this is attn_output + x (residual)
# The mean.dim output shape=[2, 16384, 16384] doesn't feed into the add...
# 
# Actually I think the computation graph shows:
# - view.default -> [2, 4, 16384, 16384]: reshape attn weights (8, 16384, 16384) -> (2, 4, 16384, 16384)
# - mean.dim -> [2, 16384, 16384]: average over heads (this is the returned attn_weights, unused)
# - add.Tensor: attn_output + x (residual, x is the permuted input)
# These are separate branches.
#
# So the actual computation I need to implement:
# 1. x_perm = permute(x.view(B,C,H*W), [2,0,1]) -> (seq_len, B, C) = (16384, 2, 128)
# 2. qkv = x_perm.reshape(32768, 128) @ W_in.T + b_in -> (32768, 384)
# 3. qkv = qkv.view(16384, 2, 384) -> split into Q,K,V each (16384, 2, 128)
# 4. Q = Q.view(16384, 8, 32).permute(1,0,2) * scale -> (8, 16384, 32)
#    K = K.view(16384, 8, 32).permute(1,0,2) -> (8, 16384, 32)
#    V = V.view(16384, 8, 32).permute(1,0,2) -> (8, 16384, 32)
# 5. attn = Q @ K.transpose(-1,-2) -> (8, 16384, 16384)
# 6. attn = softmax(attn, dim=-1)
# 7. out = attn @ V -> (8, 16384, 32)
# 8. out = out.permute(1,0,2).contiguous().view(32768, 128)
# 9. out = out @ W_out.T + b_out -> (32768, 128)
# 10. out = out.view(16384, 2, 128)
# 11. out = out + x_perm (residual)
# 12. out = layer_norm(out)
# 13. out = out.permute(1,2,0).view(B,C,H,W)
#
# The attention matrix is 8 * 16384 * 16384 * 4 bytes = 8GB which is huge.
# This is a very large attention computation. We need to be careful about memory.
# 
# For the attention, I'll implement standard (non-flash) attention since the
# computation graph explicitly shows the full attention matrix.
# Actually with seq_len=16384, the attention matrix alone is 8*16384^2*4 = 8GB
# This won't fit in GPU memory for most GPUs. But the spec says to implement it.
# Let me implement it as specified.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
    ],
    key=['seq_len', 'batch', 'num_heads', 'head_dim'],
)
@triton.jit
def extract_qkv_kernel(
    qkv_ptr,       # (seq_len*batch, 3*embed_dim) = (32768, 384)
    q_out_ptr,     # (num_heads, seq_len*batch, head_dim) = (8, 32768, 32)  -- wait
    k_out_ptr,
    v_out_ptr,
    seq_len, batch, num_heads, head_dim, embed_dim,
    scale,
    BLOCK_N: tl.constexpr,
):
    # We output Q,K,V each as (num_heads, seq_len, batch, head_dim)
    # Actually let's output as (num_heads, seq_len*batch, head_dim) for bmm
    # Wait, the attention is computed per-head over all (seq_len*batch) tokens?
    # No! Let me re-read.
    # 
    # From the graph: bmm (8, 16384, 16384) - so it's 8 heads, 16384 x 16384
    # The 16384 is seq_len, and batch=2 is handled separately somehow.
    # 
    # Actually looking at the shapes more carefully:
    # Q after permute: (8, 16384, 32) - 8 heads, 16384 seq positions, 32 head_dim
    # But batch=2... so the 8 heads must include the batch dimension!
    # 8 = num_heads * batch = 4 * 2
    # So the attention is computed with effective_heads = num_heads * batch = 8
    # and seq_len = 16384 (= H*W = 128*128)
    # 
    # This makes sense! The reshape (16384, 2, 128) -> (16384, 8, 32) merges
    # batch and head dimensions: 2*128 = 256 = 8*32
    # So head h in [0,8) corresponds to batch b = h//4, head_in_batch = h%4
    # 
    # So effectively: Q[h, s, d] where h = b*4 + head_in_batch
    # From Q_raw[s, b, c]: Q[b*4 + c//32, s, c%32]
    # 
    # For the bmm: (8, 16384, 32) x (8, 32, 16384) -> (8, 16384, 16384)
    # This computes attention between all 16384 positions for each of 8 "heads"
    # (which are really batch*num_heads combined)
    
    # pid = which (head, seq_pos) we're processing
    pid = tl.program_id(0)
    total = num_heads * seq_len  # 8 * 16384
    
    head_idx = pid // tl.cdiv(seq_len, BLOCK_N)
    block_idx = pid % tl.cdiv(seq_len, BLOCK_N)
    
    offs = block_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < seq_len
    
    # head_idx in [0, 8): b = head_idx // 4, h = head_idx % 4
    b = head_idx // (num_heads // batch)  # batch index
    h = head_idx % (num_heads // batch)   # head within batch
    
    # Q_raw[s, b, c] is stored as qkv[s*batch + b, c] (first embed_dim channels)
    # For head h, we want channels [h*head_dim, (h+1)*head_dim)
    s = offs  # seq positions
    
    # Load Q for this head
    src_row = s * batch + b  # row in qkv matrix
    src_col_base = h * head_dim  # starting column for Q
    
    q_vals = tl.load(
        qkv_ptr + src_row[:, None] * (3 * embed_dim) + (src_col_base + tl.arange(0, 32))[None, :],
        mask=mask[:, None],
        other=0.0
    ) * scale
    
    k_vals = tl.load(
        qkv_ptr + src_row[:, None] * (3 * embed_dim) + (embed_dim + src_col_base + tl.arange(0, 32))[None, :],
        mask=mask[:, None],
        other=0.0
    )
    
    v_vals = tl.load(
        qkv_ptr + src_row[:, None] * (3 * embed_dim) + (2 * embed_dim + src_col_base + tl.arange(0, 32))[None, :],
        mask=mask[:, None],
        other=0.0
    )
    
    # Store to (num_heads, seq_len, head_dim) layout
    out_base = head_idx * seq_len * head_dim + offs[:, None] * head_dim + tl.arange(0, 32)[None, :]
    tl.store(q_out_ptr + out_base, q_vals, mask=mask[:, None])
    tl.store(k_out_ptr + out_base, k_vals, mask=mask[:, None])
    tl.store(v_out_ptr + out_base, v_vals, mask=mask[:, None])


# ─── Kernel 4: Attention scores: Q @ K^T ──────────────────────────────────────
# Q: (num_heads, seq_len, head_dim), K: (num_heads, seq_len, head_dim)
# Output: (num_heads, seq_len, seq_len)
# This is a batched GEMM: for each head, (seq_len, head_dim) x (head_dim, seq_len)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['num_heads', 'seq_len', 'head_dim'],
)
@triton.jit
def attn_scores_kernel(
    Q_ptr, K_ptr, out_ptr,
    num_heads, seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (num_heads * cdiv(seq_len, BLOCK_M) * cdiv(seq_len, BLOCK_N),)
    pid = tl.program_id(0)
    n_blocks_n = tl.cdiv(seq_len, BLOCK_N)
    n_blocks_m = tl.cdiv(seq_len, BLOCK_M)
    
    head_idx = pid // (n_blocks_m * n_blocks_n)
    rem = pid % (n_blocks_m * n_blocks_n)
    pid_m = rem // n_blocks_n
    pid_n = rem % n_blocks_n
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    Q_base = head_idx * seq_len * head_dim
    K_base = head_idx * seq_len * head_dim
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(head_dim, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        q_mask = (offs_m[:, None] < seq_len) & (k_offs[None, :] < head_dim)
        k_mask = (offs_n[:, None] < seq_len) & (k_offs[None, :] < head_dim)
        q = tl.load(Q_base + Q_ptr + offs_m[:, None] * head_dim + k_offs[None, :], mask=q_mask, other=0.0)
        k = tl.load(K_base + K_ptr + offs_n[:, None] * head_dim + k_offs[None, :], mask=k_mask, other=0.0)
        acc = tl.dot(q, tl.trans(k), acc=acc)
    
    out_base = head_idx * seq_len * seq_len
    out_mask = (offs_m[:, None] < seq_len) & (offs_n[None, :] < seq_len)
    tl.store(out_ptr + out_base + offs_m[:, None] * seq_len + offs_n[None, :], acc, mask=out_mask)


# ─── Kernel 5: Softmax over last dim ──────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=8),
        triton.Config({'BLOCK_N': 2048}, num_warps=16),
    ],
    key=['n_cols'],
)
@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    n_rows, n_cols,
    BLOCK_N: tl.constexpr,
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return
    row_ptr = input_ptr + row_idx * n_cols
    out_ptr = output_ptr + row_idx * n_cols
    
    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        vals = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))
    
    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, n_cols, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        vals = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        row_sum += tl.sum(tl.exp(vals - row_max), axis=0)
    
    # Normalize
    for off in range(0, n_cols, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < n_cols
        vals = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        out = tl.exp(vals - row_max) / row_sum
        tl.store(out_ptr + cols, out, mask=mask)


# ─── Kernel 6: Attn @ V ────────────────────────────────────────────────────────
# attn: (num_heads, seq_len, seq_len), V: (num_heads, seq_len, head_dim)
# out: (num_heads, seq_len, head_dim)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['num_heads', 'seq_len', 'head_dim'],
)
@triton.jit
def attn_value_kernel(
    attn_ptr, V_ptr, out_ptr,
    num_heads, seq_len, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    n_blocks_n = tl.cdiv(head_dim, BLOCK_N)
    n_blocks_m = tl.cdiv(seq_len, BLOCK_M)
    
    head_idx = pid // (n_blocks_m * n_blocks_n)
    rem = pid % (n_blocks_m * n_blocks_n)
    pid_m = rem // n_blocks_n
    pid_n = rem % n_blocks_n
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    attn_base = head_idx * seq_len * seq_len
    V_base = head_idx * seq_len * head_dim
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(seq_len, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_mask = (offs