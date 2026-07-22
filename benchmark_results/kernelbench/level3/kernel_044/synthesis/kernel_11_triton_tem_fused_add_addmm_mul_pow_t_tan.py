import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Fused LayerNorm
# ─────────────────────────────────────────────────────────────────────────────
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
def layer_norm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N,
    stride_m,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * stride_m
    out_ptr = out_ptr + row * stride_m

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, y.to(tl.float32), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Fused GEMM (tl.dot based) for linear layers
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8,  num_stages=3),
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
        acc = tl.dot(a, b, acc=acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Attention score computation + masking + softmax
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64},  num_warps=4),
        triton.Config({'BLOCK_T': 128}, num_warps=8),
        triton.Config({'BLOCK_T': 256}, num_warps=8),
        triton.Config({'BLOCK_T': 512}, num_warps=16),
    ],
    key=['T'],
)
@triton.jit
def attn_softmax_kernel(
    att_ptr, bias_ptr,
    out_ptr,
    B, NH, T,
    scale,
    BLOCK_T: tl.constexpr,
):
    # Each program handles one row of the attention matrix: (b, h, t_row)
    pid = tl.program_id(0)
    total_rows = B * NH * T
    if pid >= total_rows:
        return

    b = pid // (NH * T)
    h = (pid // T) % NH
    t_row = pid % T

    att_row_ptr = att_ptr + b * (NH * T * T) + h * (T * T) + t_row * T
    out_row_ptr = out_ptr + b * (NH * T * T) + h * (T * T) + t_row * T

    # Load bias mask for this row: bias[0, 0, t_row, :T]
    # bias shape: [1, 1, 1024, 1024], we need [:, :, t_row, :T]
    bias_row_ptr = bias_ptr + t_row * 1024  # stride along last dim is 1

    # Online softmax with masking
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    row_sum = tl.zeros((), dtype=tl.float32)

    # Pass 1: find max
    for off in range(0, T, BLOCK_T):
        cols = off + tl.arange(0, BLOCK_T)
        mask = cols < T
        att_val = tl.load(att_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        att_val = att_val * scale
        # Load causal bias
        bias_val = tl.load(bias_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        # Apply causal mask: where bias == 0, set to -inf
        att_val = tl.where(mask & (bias_val != 0.0), att_val, float('-inf'))
        cur_max = tl.max(att_val, axis=0)
        new_max = tl.maximum(row_max, cur_max)
        row_sum = row_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(att_val - new_max), axis=0)
        row_max = new_max

    # Pass 2: normalize and store
    for off in range(0, T, BLOCK_T):
        cols = off + tl.arange(0, BLOCK_T)
        mask = cols < T
        att_val = tl.load(att_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        att_val = att_val * scale
        bias_val = tl.load(bias_row_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        att_val = tl.where(mask & (bias_val != 0.0), att_val, float('-inf'))
        softmax_val = tl.exp(att_val - row_max) / row_sum
        softmax_val = tl.where(mask, softmax_val, 0.0)
        tl.store(out_row_ptr + cols, softmax_val.to(tl.float32), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 4: Batched GEMM for attention (Q @ K^T and att @ V)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8,  num_stages=3),
    ],
    key=['M', 'N', 'K', 'BATCH'],
)
@triton.jit
def batched_gemm_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BATCH: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = a_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = c_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 5: Fused GELU activation (NewGELU)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(
    x_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654  # math.sqrt(2.0 / math.pi)
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    gelu = 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    tl.store(out_ptr + offs, gelu.to(tl.float32), mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 6: Transpose + reshape for attention output
# (B, NH, T, HS) -> (B, T, NH*HS) contiguous
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def transpose_reshape_kernel(
    x_ptr, out_ptr,
    B, NH, T, HS,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Output index: (b, t, nh, hs) in (B, T, NH, HS) layout
    # out[b, t, nh, hs] = x[b, nh, t, hs]
    total = B * T * NH * HS
    b = offs // (T * NH * HS)
    rem = offs % (T * NH * HS)
    t = rem // (NH * HS)
    rem2 = rem % (NH * HS)
    nh = rem2 // HS
    hs = rem2 % HS

    # Source index in (B, NH, T, HS)
    src_idx = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs

    val = tl.load(x_ptr + src_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 7: Pointwise add (residual connection)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def add_kernel(
    a_ptr, b_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(a_ptr + offs, mask=mask, other=0.0)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0)
    tl.store(out_ptr + offs, a + b, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 8: Prepare Q, K^T, V from QKV projection
# QKV: (B, T, 3*C) -> Q:(B,NH,T,HS), K^T:(B,NH,HS,T), V:(B,NH,T,HS)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def prepare_qkv_kernel(
    B, T, NH, HS, C,
    N,         # total elements = B*NH*T*HS
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # offs indexes into (B, NH, T, HS) output space
    b  = offs // (NH * T * HS)
    rem = offs % (NH * T * HS)
    nh = rem // (T * HS)
    rem2 = rem % (T * HS)
    t  = rem2 // HS
    hs = rem2 % HS

    # Source in QKV: qkv[b, t, q_offset + nh*HS + hs]
    q_src = b * (T * 3 * C) + t * (3 * C) + 0 * C + nh * HS + hs
    k_src = b * (T * 3 * C) + t * (3 * C) + 1 * C + nh * HS + hs
    v_src = b * (T * 3 * C) + t * (3 * C) + 2 * C + nh * HS + hs

    q_val = tl.load(qkv_ptr + q_src, mask=mask, other=0.0)
    k_val = tl.load(qkv_ptr + k_src, mask=mask, other=0.0)
    v_val = tl.load(qkv_ptr + v_src, mask=mask, other=0.0)

    # Q: (B, NH, T, HS)
    q_dst = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs
    tl.store(q_ptr + q_dst, q_val, mask=mask)

    # K^T: (B, NH, HS, T)
    kt_dst = b * (NH * HS * T) + nh * (HS * T) + hs * T + t
    tl.store(kt_ptr + kt_dst, k_val, mask=mask)

    # V: (B, NH, T, HS)
    v_dst = b * (NH * T * HS) + nh * (T * HS) + t * HS + hs
    tl.store(v_ptr + v_dst, v_val, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel_function
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
    p_mlpf___closure___0_cell_contents_c_fc_weight,
    p_mlpf___closure___0_cell_contents_c_fc_bias,
    p_mlpf___closure___0_cell_contents_c_proj_weight,
    p_mlpf___closure___0_cell_contents_c_proj_bias,
    b_attn_bias,
    x,
):
    B, T, C = x.shape  # 128, 512, 768
    NH = 8
    HS = C // NH  # 96
    device = x.device
    dtype = x.dtype

    # ── Step 1: LayerNorm 1 ──────────────────────────────────────────────────
    ln1_out = torch.empty_like(x)
    grid_ln = (B * T,)
    layer_norm_kernel[grid_ln](
        x, p_ln_1_weight, p_ln_1_bias, ln1_out,
        B * T, C, C, 1e-5,
    )

    # ── Step 2: QKV projection: (B*T, C) @ (C, 3C)^T + bias ─────────────────
    # ln1_out: (B, T, C) -> (B*T, C)
    ln1_flat = ln1_out.view(B * T, C)
    qkv_out = torch.empty(B * T, 3 * C, device=device, dtype=dtype)
    M_qkv = B * T
    N_qkv = 3 * C
    K_qkv = C
    grid_qkv = (triton.cdiv(M_qkv, 64), triton.cdiv(N_qkv, 64))
    gemm_bias_kernel[grid_qkv](
        ln1_flat, p_attn_c_attn_weight, p_attn_c_attn_bias, qkv_out,
        M_qkv, N_qkv, K_qkv,
        ln1_flat.stride(0), ln1_flat.stride(1),
        p_attn_c_attn_weight.stride(0), p_attn_c_attn_weight.stride(1),
        qkv_out.stride(0), qkv_out.stride(1),
        True,
    )
    # qkv_out: (B*T, 3C) -> (B, T, 3C)
    qkv = qkv_out.view(B, T, 3 * C)

    # ── Step 3: Prepare Q, K^T, V ────────────────────────────────────────────
    N_qkv_total = B * NH * T * HS
    q_buf  = torch.empty(B, NH, T,  HS, device=device, dtype=dtype)
    kt_buf = torch.empty(B, NH, HS, T,  device=device, dtype=dtype)
    v_buf  = torch.empty(B, NH, T,  HS, device=device, dtype=dtype)

    grid_qkv2 = (triton.cdiv(N_qkv_total, 512),)
    prepare_qkv_kernel[grid_qkv2](
        qkv, q_buf, kt_buf, v_buf,
        B, T, NH, HS, C,
        N_qkv_total,
    )

    # ── Step 4: Attention scores: Q @ K^T -> (B, NH, T, T) ──────────────────
    # q_buf: (B, NH, T, HS), kt_buf: (B, NH, HS, T)
    # We treat batch dim as B*NH
    att_scores = torch.empty(B, NH, T, T, device=device, dtype=dtype)
    BNH = B * NH
    grid_att = (triton.cdiv(T, 64), triton.cdiv(T, 64), BNH)
    batched_gemm_kernel[grid_att](
        q_buf.view(BNH, T, HS),
        kt_buf.view(BNH, HS, T),
        att_scores.view(BNH, T, T),
        T, T, HS,
        T * HS, HS, 1,
        HS * T, HS, 1,
        T * T, T, 1,
        BNH,
    )

    # ── Step 5: Scale + mask + softmax ───────────────────────────────────────
    scale = 1.0 / math.sqrt(HS)
    att_soft = torch.empty(B, NH, T, T, device=device, dtype=dtype)
    grid_soft = (B * NH * T,)
    attn_softmax_kernel[grid_soft](
        att_scores, b_attn_bias,
        att_soft,
        B, NH, T,
        scale,
    )

    # ── Step 6: att @ V -> (B, NH, T, HS) ───────────────────────────────────
    # att_soft: (B, NH, T, T), v_buf: (B, NH, T, HS)
    attn_out = torch.empty(B, NH, T, HS, device=device, dtype=dtype)
    grid_av = (triton.cdiv(T, 64), triton.cdiv(HS, 64), BNH)
    batched_gemm_kernel[grid_av](
        att_soft.view(BNH, T, T),
        v_buf.view(BNH, T, HS),
        attn_out.view(BNH, T, HS),
        T, HS, T,
        T * T, T, 1,
        T * HS, HS, 1,
        T * HS, HS, 1,