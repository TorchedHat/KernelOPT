import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Fused LayerNorm
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * N
    out_ptr = out_ptr + row * N

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, y.to(tl.float32), mask=mask)


def fused_layer_norm(x, weight, bias, eps=1e-5):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    out = torch.empty_like(x)
    grid = (M,)
    layer_norm_kernel[grid](
        x, weight, bias, out,
        M, N, eps,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: GEMM with bias (M x K) x (K x N) + bias
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
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
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
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

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptr + offs_cm[:, None] * stride_cm + offs_cn[None, :] * stride_cn, acc, mask=mask)


def gemm_bias(a, b, bias):
    # a: (M, K), b: (N, K) [weight is stored transposed], bias: (N,)
    # computes a @ b.T + bias
    assert a.ndim == 2 and b.ndim == 2
    M, K = a.shape
    N = b.shape[0]
    assert b.shape[1] == K
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_bias_kernel[grid](
        a, b, bias, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(1), b.stride(0),  # b transposed: stride_bk=b.stride(1), stride_bn=b.stride(0)
        c.stride(0), c.stride(1),
    )
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Attention QK^T matmul + scale + causal mask + softmax
# Shape: Q,K,V are (B, nh, T, hs)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_T': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['T', 'hs'],
)
@triton.jit
def attn_qk_softmax_kernel(
    q_ptr, k_ptr, out_ptr,
    B, nh, T, hs,
    scale,
    stride_qb, stride_qh, stride_qt, stride_qd,
    stride_kb, stride_kh, stride_kt, stride_kd,
    stride_ob, stride_oh, stride_ot, stride_os,
    BLOCK_T: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one row of the attention matrix: (b, h, t_row, :)
    # program_id(0) = b * nh * T + h * T + t_row
    pid = tl.program_id(0)
    b = pid // (nh * T)
    h = (pid // T) % nh
    t_row = pid % T

    q_base = q_ptr + b * stride_qb + h * stride_qh + t_row * stride_qt
    k_base = k_ptr + b * stride_kb + h * stride_kh

    # Load q row: (hs,)
    q_offs = tl.arange(0, BLOCK_K)
    q = tl.load(q_base + q_offs * stride_qd, mask=q_offs < hs, other=0.0)

    # Compute attention scores for all t_col <= t_row (causal)
    # We compute in blocks of BLOCK_T
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((), dtype=tl.float32)

    # First pass: compute max for numerical stability
    # We'll do online softmax
    # Store partial scores in registers isn't feasible for T=512
    # Instead, do two passes over T

    # Actually for T=512, we can load all scores at once if BLOCK_T=512
    # Let's do it in a single pass with online softmax

    # Online softmax over t_col dimension
    # We need to store the scores temporarily - but T can be 512
    # Use a loop with online softmax

    # Accumulate softmax online
    # We'll write scores to output first, then normalize
    # But that requires two passes over output memory

    # For simplicity, do two passes:
    # Pass 1: compute all scores, find max
    # Pass 2: compute exp, sum, normalize

    # Pass 1: find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T
        causal_mask = t_offs <= t_row

        # Compute dot product q . k[t_col]
        # k[t_col] shape: (hs,) for each t_col in BLOCK_T
        # We need to compute q (hs,) dot k_t (hs,) for each t_col
        # Load k block: (BLOCK_T, hs)
        k_ptrs = k_base + t_offs[:, None] * stride_kt + q_offs[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=t_mask[:, None] & (q_offs[None, :] < hs), other=0.0)
        # score = q . k_t * scale
        score = tl.sum(q[None, :] * k_block, axis=1) * scale
        # Apply causal mask
        score = tl.where(causal_mask & t_mask, score, float('-inf'))
        block_max = tl.max(score, axis=0)
        row_max = tl.maximum(row_max, block_max)

    # Pass 2: compute exp sum and write normalized scores
    row_sum = tl.zeros((), dtype=tl.float32)
    for t_start in range(0, T, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T
        causal_mask = t_offs <= t_row

        k_ptrs = k_base + t_offs[:, None] * stride_kt + q_offs[None, :] * stride_kd
        k_block = tl.load(k_ptrs, mask=t_mask[:, None] & (q_offs[None, :] < hs), other=0.0)
        score = tl.sum(q[None, :] * k_block, axis=1) * scale
        score = tl.where(causal_mask & t_mask, score, float('-inf'))
        exp_score = tl.exp(score - row_max)
        exp_score = tl.where(t_mask, exp_score, 0.0)
        row_sum += tl.sum(exp_score, axis=0)

        # Store exp scores temporarily
        out_ptrs = out_ptr + b * stride_ob + h * stride_oh + t_row * stride_ot + t_offs * stride_os
        tl.store(out_ptrs, exp_score, mask=t_mask)

    # Pass 3: normalize
    for t_start in range(0, T, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T
        out_ptrs = out_ptr + b * stride_ob + h * stride_oh + t_row * stride_ot + t_offs * stride_os
        exp_score = tl.load(out_ptrs, mask=t_mask, other=0.0)
        normalized = exp_score / row_sum
        tl.store(out_ptrs, normalized, mask=t_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 4: Attention value aggregation: att @ V
# att: (B, nh, T, T), V: (B, nh, T, hs) -> out: (B, nh, T, hs)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64,  'BLOCK_HS': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 128, 'BLOCK_HS': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_T': 64,  'BLOCK_HS': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_T': 128, 'BLOCK_HS': 64}, num_warps=8, num_stages=2),
    ],
    key=['T', 'hs'],
)
@triton.jit
def attn_av_kernel(
    att_ptr, v_ptr, out_ptr,
    B, nh, T, hs,
    stride_ab, stride_ah, stride_at, stride_as,
    stride_vb, stride_vh, stride_vt, stride_vd,
    stride_ob, stride_oh, stride_ot, stride_od,
    BLOCK_T: tl.constexpr,
    BLOCK_HS: tl.constexpr,
):
    # Each program: (b, h, t_row, hs_block)
    pid = tl.program_id(0)
    n_hs_blocks = tl.cdiv(hs, BLOCK_HS)
    b = pid // (nh * T * n_hs_blocks)
    rem = pid % (nh * T * n_hs_blocks)
    h = rem // (T * n_hs_blocks)
    rem2 = rem % (T * n_hs_blocks)
    t_row = rem2 // n_hs_blocks
    hs_block = rem2 % n_hs_blocks

    att_base = att_ptr + b * stride_ab + h * stride_ah + t_row * stride_at
    v_base = v_ptr + b * stride_vb + h * stride_vh

    hs_offs = hs_block * BLOCK_HS + tl.arange(0, BLOCK_HS)
    acc = tl.zeros((BLOCK_HS,), dtype=tl.float32)

    for t_start in range(0, T, BLOCK_T):
        t_offs = t_start + tl.arange(0, BLOCK_T)
        t_mask = t_offs < T

        att = tl.load(att_base + t_offs * stride_as, mask=t_mask, other=0.0)
        v_ptrs = v_base + t_offs[:, None] * stride_vt + hs_offs[None, :] * stride_vd
        v_block = tl.load(v_ptrs, mask=t_mask[:, None] & (hs_offs[None, :] < hs), other=0.0)
        acc += tl.sum(att[:, None] * v_block, axis=0)

    out_base = out_ptr + b * stride_ob + h * stride_oh + t_row * stride_ot
    tl.store(out_base + hs_offs * stride_od, acc, mask=hs_offs < hs)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 5: Fused GELU activation (NewGELU)
# ─────────────────────────────────────────────────────────────────────────────
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
    # NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # math.sqrt(2.0 / math.pi)
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    tanh_val = triton.language.extra.libdevice.tanh(inner)
    y = 0.5 * x * (1.0 + tanh_val)
    tl.store(out_ptr + offs, y, mask=mask)


def fused_gelu(x):
    N = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 6: Elementwise add
# ─────────────────────────────────────────────────────────────────────────────
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


def triton_add(a, b):
    N = a.numel()
    out = torch.empty_like(a)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](a, b, out, N)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 7: QKV reshape + transpose kernel
# Input: qkv (B, T, 3*C), output q,k,v each (B, nh, T, hs)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def qkv_split_transpose_kernel(
    qkv_ptr, q_ptr, k_ptr, v_ptr,
    B, T, nh, hs,
    total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # qkv: (B, T, 3*nh*hs) contiguous
    # q,k,v: (B, nh, T, hs) contiguous
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total  # total = B * nh * T * hs

    # Decode index in (B, nh, T, hs) layout
    hs_idx = offs % hs
    t_idx = (offs // hs) % T
    nh_idx = (offs // (hs * T)) % nh
    b_idx = offs // (hs * T * nh)

    # Source index in qkv: (B, T, 3*nh*hs)
    # q starts at 0, k at nh*hs, v at 2*nh*hs
    C = nh * hs
    qkv_base = b_idx * (T * 3 * C) + t_idx * (3 * C) + nh_idx * hs + hs_idx

    q_val = tl.load(qkv_ptr + qkv_base, mask=mask, other=0.0)
    k_val = tl.load(qkv_ptr + qkv_base + C, mask=mask, other=0.0)
    v_val = tl.load(qkv_ptr + qkv_base + 2 * C, mask=mask, other=0.0)

    tl.store(q_ptr + offs, q_val, mask=mask)
    tl.store(k_ptr + offs, k_val, mask=mask)
    tl.store(v_ptr + offs, v_val, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 8: Transpose (B, nh, T, hs) -> (B, T, C) contiguous
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def transpose_heads_kernel(
    inp_ptr, out_ptr,
    B, nh, T, hs,
    total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # inp: (B, nh, T, hs) contiguous -> out: (B, T, nh*hs) contiguous
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # Decode from output layout (B, T, nh, hs)
    hs_idx = offs % hs
    nh_idx = (offs // hs) % nh
    t_idx = (offs // (hs * nh)) % T
    b_idx = offs // (hs * nh * T)

    # Source in (B, nh, T, hs)
    src_idx = b_idx * (nh * T * hs) + nh_idx * (T * hs) + t_idx * hs + hs_idx

    val = tl.load(inp_ptr + src_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel function
# ─────────────────────────────────────────────────────────────────────────────
def kernel_function(
    p_ln_1_weight,
    p_ln_1_bias,
    p_attn_c_attn_weight,
    p_attn_c_attn_bias,
    p_attn_c_proj_weight,
    p_attn_c_proj_bias,
    p_ln_2_weight,
    p_ln_2_bias,
    p_mlpf_c_fc_weight,
    p_mlpf_c_fc_bias,
    p_mlpf_c_proj_weight,
    p_mlpf_c_proj_bias,
    b_attn_bias,
    x,
):
    B, T, C = x.shape  # 128, 512, 768
    n_head = 8
    hs = C // n_head  # 96

    # ── Step 1: LayerNorm 1 ──────────────────────────────────────────────────
    ln1_out = fused_layer_norm(x, p_ln_1_weight, p_ln_1_bias)

    # ── Step 2: QKV projection ───────────────────────────────────────────────
    # ln1_out: (B, T, C) -> reshape to (B*T, C)
    ln1_flat = ln1_out.reshape(B * T, C)
    # p_attn_c_attn_weight: (2304, 768), bias: (2304,)
    qkv_flat = gemm_bias(ln1_flat, p_attn_c_attn_weight, p_attn_c_attn_bias)
    # qkv_flat: (