import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: permute + clone (NCHW -> seq_len, batch, C) ───────────────────

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
def permute_nchw_to_snc_kernel(
    inp_ptr, out_ptr,
    B, C, HW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    # out[hw, b, c] = inp[b, c, hw]
    # inp shape: (B, C, HW), strides: (C*HW, HW, 1)
    # out shape: (HW, B, C), strides: (B*C, C, 1)
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    # out index: offs = hw * B * C + b * C + c
    c = offs % C
    tmp = offs // C
    b = tmp % B
    hw = tmp // B
    # inp index: b * C * HW + c * HW + hw
    inp_idx = b * C * HW + c * HW + hw
    val = tl.load(inp_ptr + inp_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


# ─── Kernel 2: GEMM for in_proj: (S*B, C) x (C, 3C) + bias ─────────────────

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
        acc = tl.dot(a, b, acc, allow_tf32=False)
    # add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, acc, mask=c_mask)


# ─── Kernel 3: Rearrange QKV + scale Q ──────────────────────────────────────
# Input: qkv (S, B, 3, H, D) contiguous -> output Q(H,S*B,D), K(H,S*B,D), V(H,S*B,D)
# Actually: qkv shape after linear: (S*B, 3*C) -> reshape to (S, B, 3, num_heads, head_dim)
# Q[h, sb, d] = qkv[sb, 0, h, d] * scale  (sb = s*B + b)
# K[h, sb, d] = qkv[sb, 1, h, d]
# V[h, sb, d] = qkv[sb, 2, h, d]

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
def rearrange_qkv_kernel(
    SB, H, D,
    scale,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    # offs = h * SB * D + sb * D + d
    d = offs % D
    tmp = offs // D
    sb = tmp % SB
    h = tmp // SB
    # qkv[sb, qkv_idx, h, d] where qkv is (SB, 3, H, D) = (SB, 3*C)
    # qkv[sb, q_or_k_or_v, h, d] = qkv_ptr[sb * 3*H*D + q_or_k_or_v * H*D + h*D + d]
    C = H * D
    q_idx = sb * (3 * C) + 0 * C + h * D + d
    k_idx = sb * (3 * C) + 1 * C + h * D + d
    v_idx = sb * (3 * C) + 2 * C + h * D + d
    q_val = tl.load(qkv_ptr + q_idx, mask=mask, other=0.0)
    k_val = tl.load(qkv_ptr + k_idx, mask=mask, other=0.0)
    v_val = tl.load(qkv_ptr + v_idx, mask=mask, other=0.0)
    tl.store(Q_ptr + offs, q_val * scale, mask=mask)
    tl.store(K_ptr + offs, k_val, mask=mask)
    tl.store(V_ptr + offs, v_val, mask=mask)


# ─── Kernel 4: Batched attention scores: (H, SB, D) x (H, D, SB) -> (H, SB, SB) ─
# This is too large (16384x16384 per head) to do in Triton efficiently in one kernel
# We'll use a tiled approach per batch element

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['SB', 'D'],
)
@triton.jit
def batched_qk_kernel(
    Q_ptr, K_ptr, Out_ptr,
    H, SB, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Grid: (H, cdiv(SB, BLOCK_M), cdiv(SB, BLOCK_N))
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    Q_base = Q_ptr + pid_h * SB * D
    K_base = K_ptr + pid_h * SB * D
    for k in range(0, tl.cdiv(D, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        q_mask = (offs_m[:, None] < SB) & (k_offs[None, :] < D)
        k_mask = (offs_n[:, None] < SB) & (k_offs[None, :] < D)
        q = tl.load(Q_base + offs_m[:, None] * D + k_offs[None, :], mask=q_mask, other=0.0)
        k_t = tl.load(K_base + offs_n[:, None] * D + k_offs[None, :], mask=k_mask, other=0.0)
        acc = tl.dot(q, tl.trans(k_t), acc, allow_tf32=False)
    out_mask = (offs_m[:, None] < SB) & (offs_n[None, :] < SB)
    tl.store(Out_ptr + pid_h * SB * SB + offs_m[:, None] * SB + offs_n[None, :], acc, mask=out_mask)


# ─── Kernel 5: Softmax over last dim of (H, SB, SB) ────────────────────────

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
def softmax_rows_kernel(
    inp_ptr, out_ptr,
    num_rows, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= num_rows:
        return
    base = inp_ptr + row * N
    out_base = out_ptr + row * N
    # find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        vals = tl.load(base + cols, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))
    # sum exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        vals = tl.load(base + cols, mask=mask, other=float('-inf'))
        row_sum += tl.sum(tl.exp(vals - row_max), axis=0)
    # store
    for off in range(0, tl.cdiv(N, BLOCK_N)):
        cols = off * BLOCK_N + tl.arange(0, BLOCK_N)
        mask = cols < N
        vals = tl.load(base + cols, mask=mask, other=float('-inf'))
        out_vals = tl.exp(vals - row_max) / row_sum
        tl.store(out_base + cols, out_vals, mask=mask)


# ─── Kernel 6: Batched attn @ V: (H, SB, SB) x (H, SB, D) -> (H, SB, D) ───

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['SB', 'D'],
)
@triton.jit
def batched_av_kernel(
    A_ptr, V_ptr, Out_ptr,
    H, SB, D,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_h = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    A_base = A_ptr + pid_h * SB * SB
    V_base = V_ptr + pid_h * SB * D
    for k in range(0, tl.cdiv(SB, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_mask = (offs_m[:, None] < SB) & (k_offs[None, :] < SB)
        v_mask = (k_offs[:, None] < SB) & (offs_n[None, :] < D)
        a = tl.load(A_base + offs_m[:, None] * SB + k_offs[None, :], mask=a_mask, other=0.0)
        v = tl.load(V_base + k_offs[:, None] * D + offs_n[None, :], mask=v_mask, other=0.0)
        acc = tl.dot(a, v, acc, allow_tf32=False)
    out_mask = (offs_m[:, None] < SB) & (offs_n[None, :] < D)
    tl.store(Out_ptr + pid_h * SB * D + offs_m[:, None] * D + offs_n[None, :], acc, mask=out_mask)


# ─── Kernel 7: Transpose (H, SB, D) -> (SB, H, D) then reshape to (SB, H*D) ─

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
def transpose_hsd_to_shd_kernel(
    inp_ptr, out_ptr,
    H, SB, D,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    # inp: (H, SB, D), out: (SB, H*D) = (SB, C)
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    # out index: sb * H * D + h * D + d
    d = offs % D
    tmp = offs // D
    h = tmp % H
    sb = tmp // H
    # inp index: h * SB * D + sb * D + d
    inp_idx = h * SB * D + sb * D + d
    val = tl.load(inp_ptr + inp_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


# ─── Kernel 8: GEMM for out_proj: (SB, C) x (C, C) + bias ──────────────────
# Reuse gemm_bias_kernel


# ─── Kernel 9: Fused mean over heads + residual + LayerNorm ─────────────────
# The computation graph shows:
#   attn_output: (SB, C) -> view (2, 4, SB/2, SB/2) -> mean over dim [1] -> (2, SB/2, SB/2)
# Wait, let me re-read...
# view.default -> shape=[2, 4, 16384, 16384] -- this seems wrong for the output
# Actually looking more carefully:
# After out_proj: (32768, 128) -> view (16384, 2, 128)
# Then view (2, 4, 16384, 16384) -- this doesn't make sense dimensionally
# 16384 * 2 * 128 = 4194304, but 2 * 4 * 16384 * 16384 = 2147483648 -- way too big
# 
# Wait, I think the mean.dim is computing mean over the attention weights, not the output.
# Looking at the original code: attn_output, _ = self.attn(x, x, x)
# The _ is the attention weights averaged over heads.
# The mean.dim -> shape=[2, 16384, 16384] is the attention weights (not used in forward pass output)
# 
# So the actual computation is:
# 1. attn_output: (SB, C) after out_proj
# 2. x (residual): (SB, C) = permuted input
# 3. x = norm(attn_output + x)
# 4. reshape back

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 256}, num_warps=8),
        triton.Config({'BLOCK_C': 64}, num_warps=2),
        triton.Config({'BLOCK_C': 512}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def residual_layernorm_kernel(
    SB, C,
    eps,
    BLOCK_C: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= SB:
        return
    base_a = attn_ptr + row * C
    base_r = residual_ptr + row * C
    base_o = out_ptr + row * C
    # Compute mean and variance
    mean = tl.zeros((), dtype=tl.float32)
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, tl.cdiv(C, BLOCK_C)):
        cols = off * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = cols < C
        a = tl.load(base_a + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(base_r + cols, mask=mask, other=0.0).to(tl.float32)
        x = a + r
        mean += tl.sum(tl.where(mask, x, tl.zeros_like(x)), axis=0)
    mean = mean / C
    for off in range(0, tl.cdiv(C, BLOCK_C)):
        cols = off * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = cols < C
        a = tl.load(base_a + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(base_r + cols, mask=mask, other=0.0).to(tl.float32)
        x = a + r
        diff = tl.where(mask, x - mean, tl.zeros_like(x))
        var += tl.sum(diff * diff, axis=0)
    var = var / C
    inv_std = 1.0 / tl.sqrt(var + eps)
    # Normalize and scale
    for off in range(0, tl.cdiv(C, BLOCK_C)):
        cols = off * BLOCK_C + tl.arange(0, BLOCK_C)
        mask = cols < C
        a = tl.load(base_a + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(base_r + cols, mask=mask, other=0.0).to(tl.float32)
        x = a + r
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * inv_std * w + b
        tl.store(base_o + cols, out.to(tl.float32), mask=mask)


# ─── Kernel 10: permute (SB, C) -> (B, C, HW) -> (B, C, H, W) ───────────────

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
def permute_snc_to_nchw_kernel(
    inp_ptr, out_ptr,
    B, C, HW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    # inp: (HW, B, C) -> out: (B, C, HW)
    # inp[hw, b, c] -> out[b, c, hw]
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    # out index: b * C * HW + c * HW + hw
    hw = offs % HW
    tmp = offs // HW
    c = tmp % C
    b = tmp // C
    # inp index: hw * B * C + b * C + c
    inp_idx = hw * B * C + b * C + c
    val = tl.load(inp_ptr + inp_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


def kernel_function(
):
    B, C, H, W = x.shape
    HW = H * W
    SB = HW * B  # seq_len * batch = 16384 * 2 = 32768
    num_heads = 8  # from the computation graph (8 heads, head_dim=16... wait)
    # From graph: view.default -> shape=[16384, 8, 32] -- so 8 heads, head_dim=32
    # But num_heads in original model is 4... let me check
    # Original: num_heads=4, embed_dim=128, head_dim=32
    # But graph shows [16384, 8, 32] -- that's 8 heads!
    # Wait: nn.MultiheadAttention with num_heads=4 uses 4 heads
    # But the graph shows 8 heads... Let me re-read
    # "view.default -> shape=[16384, 8, 32]" -- 8*32=256=2*128, so this is 2 batches merged?
    # Actually: seq_len=16384 (H*W), batch=2, so SB=32768
    # The view [16384, 8, 32]: this is per-batch? No...
    # Looking at the graph more carefully:
    # select.int -> shape=[16384, 2, 128] (Q for all batches)
    # view.default -> shape=[16384, 8, 32] -- 8 heads, head_dim=32, but 8*32=256!=128
    # Hmm, 16384 * 8 * 32 = 4194304 = 16384 * 256 = 16384 * 2 * 128
    # So it's treating batch=2 as part of the head dimension? 2*4=8 heads
    # Yes! The batch dimension is merged with heads: num_heads_total = batch * num_heads = 2 * 4 = 8
    # So effectively: Q shape [8, 16384, 32] where 8 = 2*4 (batch*heads)
    
    num_heads = 4  # actual heads per batch
    head_dim = C // num_heads  # 128 // 4 = 32
    H_total = B * num_heads  # 8 (merged batch+heads)
    
    device = x.device
    dtype = x.dtype
    
    # Step 1: Permute x from (B, C, H, W) to (HW, B, C) = (16384, 2, 128)
    # Then reshape to (SB, C) = (32768, 128) for GEMM
    x_perm = torch.empty(HW, B, C, device=device, dtype=dtype)
    numel_perm = HW * B * C
    grid_perm = lambda meta: (triton.cdiv(numel_perm, meta['BLOCK_SIZE']),)
    permute_nchw_to_snc_kernel[grid_perm](
        x, x_perm,
        B, C, HW,
        numel_perm,
    )
    #