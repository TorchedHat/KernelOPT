import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: Permute copy (NCHW -> seq_len, batch, embed) ───────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def permute_kernel(inp_ptr, out_ptr, B, C, HW, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    # out[hw, b, c] = inp[b, c, hw]
    # out index: hw * B * C + b * C + c
    # inp index: b * C * HW + c * HW + hw
    hw = (offs // (B * C))
    rem = offs % (B * C)
    b = rem // C
    c = rem % C
    inp_idx = b * C * HW + c * HW + hw
    tmp = tl.load(inp_ptr + inp_idx, mask=mask)
    tl.store(out_ptr + offs, tmp, mask=mask)


# ─── Kernel 2: GEMM for QKV projection ───────────────────────────────────────
# x: [M, K], W: [N, K], bias: [N] -> out: [M, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bn, stride_bk,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
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
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_ptr + offs_n[None, :] * stride_bn + k_offs[:, None] * stride_bk
        
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, out_dtype=tl.float32)
    
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
    
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


# ─── Kernel 3: Prepare Q, K, V with scaling ──────────────────────────────────
# Input: qkv [M, 3*num_heads*head_dim]
# Output: q_scaled [num_heads, M, head_dim], k [num_heads, M, head_dim], v [num_heads, M, head_dim]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def prepare_qkv_kernel(
    qkv_ptr, q_ptr, k_ptr, v_ptr,
    M, num_heads, head_dim, scale,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    # offs indexes into [num_heads, M, head_dim]
    hd = offs % head_dim
    m = (offs // head_dim) % M
    h = offs // (head_dim * M)
    
    # qkv layout: [M, 3, num_heads, head_dim] -> [M, num_heads, 3, head_dim]
    # Actually qkv is [M, 3*num_heads*head_dim] = [M, 3*embed_dim]
    # q = qkv[:, :embed_dim], k = qkv[:, embed_dim:2*embed_dim], v = qkv[:, 2*embed_dim:]
    # q reshaped to [M, num_heads, head_dim], then transposed to [num_heads, M, head_dim]
    embed_dim = num_heads * head_dim
    
    q_src = qkv_ptr + m * (3 * embed_dim) + h * head_dim + hd
    k_src = qkv_ptr + m * (3 * embed_dim) + embed_dim + h * head_dim + hd
    v_src = qkv_ptr + m * (3 * embed_dim) + 2 * embed_dim + h * head_dim + hd
    
    q_val = tl.load(q_src, mask=mask, other=0.0)
    k_val = tl.load(k_src, mask=mask, other=0.0)
    v_val = tl.load(v_src, mask=mask, other=0.0)
    
    tl.store(q_ptr + offs, q_val * scale, mask=mask)
    tl.store(k_ptr + offs, k_val, mask=mask)
    tl.store(v_ptr + offs, v_val, mask=mask)


# ─── Kernel 4: Batched attention GEMM (Q @ K^T) ──────────────────────────────
# Q: [B_heads, M, head_dim], K: [B_heads, M, head_dim]
# Out: [B_heads, M, M]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M_dim', 'N_dim', 'K_dim'],
)
@triton.jit
def batched_qk_kernel(
    Q_ptr, K_ptr, Out_ptr,
    B_heads, M_dim, N_dim, K_dim,
    stride_qb, stride_qm, stride_qk,
    stride_kb, stride_kn, stride_kk,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K_dim, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        q_ptrs = Q_ptr + pid_b * stride_qb + offs_m[:, None] * stride_qm + k_offs[None, :] * stride_qk
        k_ptrs = K_ptr + pid_b * stride_kb + offs_n[None, :] * stride_kn + k_offs[:, None] * stride_kk
        
        q_mask = (offs_m[:, None] < M_dim) & (k_offs[None, :] < K_dim)
        k_mask = (offs_n[None, :] < N_dim) & (k_offs[:, None] < K_dim)
        
        q = tl.load(q_ptrs, mask=q_mask, other=0.0)
        k = tl.load(k_ptrs, mask=k_mask, other=0.0)
        acc = tl.dot(q, k, acc, out_dtype=tl.float32)
    
    out_ptrs = Out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M_dim) & (offs_n[None, :] < N_dim)
    tl.store(out_ptrs, acc, mask=out_mask)


# ─── Kernel 5: Softmax over last dim ─────────────────────────────────────────
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
    inp_ptr, out_ptr,
    M, N,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    
    row_ptr = inp_ptr + row * stride_m
    out_row_ptr = out_ptr + row * stride_m
    
    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(x, axis=0))
    
    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)
    
    # Normalize
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(row_ptr + cols, mask=mask, other=float('-inf'))
        out = tl.exp(x - row_max) / row_sum
        tl.store(out_row_ptr + cols, out, mask=mask)


# ─── Kernel 6: Batched attention GEMM (Attn @ V) ─────────────────────────────
# Attn: [B_heads, M, M], V: [B_heads, M, head_dim]
# Out: [B_heads, M, head_dim]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M_dim', 'N_dim', 'K_dim'],
)
@triton.jit
def batched_av_kernel(
    A_ptr, V_ptr, Out_ptr,
    B_heads, M_dim, N_dim, K_dim,
    stride_ab, stride_am, stride_ak,
    stride_vb, stride_vk, stride_vn,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(K_dim, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        v_ptrs = V_ptr + pid_b * stride_vb + k_offs[:, None] * stride_vk + offs_n[None, :] * stride_vn
        
        a_mask = (offs_m[:, None] < M_dim) & (k_offs[None, :] < K_dim)
        v_mask = (k_offs[:, None] < K_dim) & (offs_n[None, :] < N_dim)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        v = tl.load(v_ptrs, mask=v_mask, other=0.0)
        acc = tl.dot(a, v, acc, out_dtype=tl.float32)
    
    out_ptrs = Out_ptr + pid_b * stride_ob + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M_dim) & (offs_n[None, :] < N_dim)
    tl.store(out_ptrs, acc, mask=out_mask)


# ─── Kernel 7: Transpose + reshape attn output ───────────────────────────────
# Input: [num_heads, M, head_dim] -> Output: [M, embed_dim]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def transpose_reshape_kernel(
    inp_ptr, out_ptr,
    M, num_heads, head_dim,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    # out[m, h*head_dim + hd] = inp[h, m, hd]
    # out index: m * embed_dim + h * head_dim + hd
    embed_dim = num_heads * head_dim
    m = offs // embed_dim
    rem = offs % embed_dim
    h = rem // head_dim
    hd = rem % head_dim
    
    inp_idx = h * M * head_dim + m * head_dim + hd
    val = tl.load(inp_ptr + inp_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


# ─── Kernel 8: Fused residual + LayerNorm ────────────────────────────────────
# attn_out: [M, embed_dim], x_orig: [M, embed_dim]
# weight: [embed_dim], bias: [embed_dim]
# out: [M, embed_dim]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 64}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def fused_residual_layernorm_kernel(
    attn_ptr, x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, eps,
    stride_m,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    
    row_attn = attn_ptr + row * stride_m
    row_x = x_ptr + row * stride_m
    row_out = out_ptr + row * stride_m
    
    # Compute mean of (attn + x)
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        a = tl.load(row_attn + cols, mask=mask, other=0.0).to(tl.float32)
        xv = tl.load(row_x + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(a + xv, axis=0)
    mean = mean / N
    
    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        a = tl.load(row_attn + cols, mask=mask, other=0.0).to(tl.float32)
        xv = tl.load(row_x + cols, mask=mask, other=0.0).to(tl.float32)
        diff = (a + xv) - mean
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)
    
    # Normalize and apply weight/bias
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        a = tl.load(row_attn + cols, mask=mask, other=0.0).to(tl.float32)
        xv = tl.load(row_x + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        
        val = ((a + xv) - mean) * rstd * w + b
        tl.store(row_out + cols, val, mask=mask)


# ─── Kernel 9: Final permute back ────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def permute_back_kernel(inp_ptr, out_ptr, B, C, HW, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    
    # inp[hw, b, c] -> out[b, c, hw]
    # inp index: hw * B * C + b * C + c
    # out index: b * C * HW + c * HW + hw
    hw = offs // (B * C)
    rem = offs % (B * C)
    b = rem // C
    c = rem % C
    out_idx = b * C * HW + c * HW + hw
    
    val = tl.load(inp_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + out_idx, val, mask=mask)


def kernel_function(
    p_attn_in_proj_bias,
    p_attn_in_proj_weight,
    p_attn_out_proj_weight,
    p_attn_out_proj_bias,
    p_norm_weight,
    p_norm_bias,
    x,
):
    B, C, H, W = x.shape
    HW = H * W
    embed_dim = C
    num_heads = 8  # from the computation graph: 8 heads, head_dim=32 (128/4=32... wait)
    # From graph: view [16384, 8, 32] -> 8 heads, head_dim=32, embed_dim=128=8*16... 
    # Actually num_heads=8 from the graph (8, 16384, 32)
    # But original model uses num_heads=4... let me check
    # The graph shows [16384, 8, 32] which means 8 heads with head_dim=32
    # But original model says num_heads=4... 
    # Wait: embed_dim=128, num_heads=4 -> head_dim=32
    # But graph shows 8 heads... Let me re-read
    # "view.default -> shape=[16384, 8, 32]" - this is 8 heads
    # But model init says num_heads=4... 
    # Actually looking at the bmm: [8, 16384, 16384] - definitely 8 heads
    # The model uses num_heads=4 but the graph shows 8... 
    # Wait, nn.MultiheadAttention with embed_dim=128, num_heads=4 -> head_dim=32
    # But the graph shows 8 heads... Let me check: 128/4=32, but graph shows [16384, 8, 32]
    # Hmm, maybe num_heads=4 in the model but the internal implementation uses 2*num_heads?
    # No. Let me just use what the graph shows: 8 heads, head_dim=32
    # Actually wait - the model says num_heads=4 in get_init_inputs but the graph shows 8 heads
    # This is confusing. Let me use num_heads from the weight shape:
    # in_proj_weight: [384, 128] -> 3 * embed_dim = 384 -> embed_dim=128
    # The graph shows [16384, 8, 32] -> 8 heads * 32 = 256 != 128
    # Wait: 8 * 32 = 256... but embed_dim=128. That can't be right.
    # Oh wait: [16384, 8, 32] where 8*32=256... but embed_dim=128
    # Hmm, let me re-read: "view.default -> shape=[16384, 8, 32]"
    # 16384 * 8 * 32 = 4194304 = 16384 * 256... but embed_dim=128
    # Actually: 16384 is seq_len, 8 is num_heads... but 8*32=256 != 128
    # Wait, I think I'm misreading. Let me look at the batch dimension.
    # x: [2, 128, 128, 128] -> B=2, C=128, H=128, W=128
    # seq_len = H*W = 16384, batch=2, embed_dim=128
    # After permute: [16384, 2, 128]
    # After linear: [16384, 2, 384] (= [seq_len, batch, 3*embed_dim])
    # Then view [16384, 8, 32]: this is [seq_len*batch, num_heads, head_dim]?
    # 16384 * 2 = 32768, but view shows 16384...
    # Oh! The view is [16384, 8, 32] where 16384 = seq_len, 8 = num_heads... 
    # but 8*32=256 != 128. Unless batch is folded in?
    # Actually: select.int gives [16384, 2, 128] (Q), then view gives [16384, 8, 32]
    # 16384 * 8 * 32 = 16384 * 256 != 16384 * 2 * 128 = 16384 * 256. YES! 2*128=256=8*32
    # So the view folds batch into heads: [seq_len, batch*num_heads, head_dim]
    # With batch=2, num_heads=4, head_dim=32: 2*4=8 heads total, head_dim=32
    # That makes sense! num_heads=4 in model, but effective_heads = batch *