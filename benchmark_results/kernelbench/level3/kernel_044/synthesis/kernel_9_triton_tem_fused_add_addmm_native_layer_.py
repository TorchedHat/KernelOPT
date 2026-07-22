import torch
import triton
import triton.language as tl
import math

# ============================================================
# Kernel 1: Fused LayerNorm
# ============================================================
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
    rstd = tl.math.rsqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(b_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, y.to(tl.float32), mask=mask)


# ============================================================
# Kernel 2: Fused bias add + split into Q, K, V with reshape/transpose
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def bias_split_qkv_kernel(
    qkv_ptr, bias_ptr,
    q_ptr, k_ptr, v_ptr,
    B, T, n_head, head_dim,
    total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # qkv_ptr: [B*T, 3*n_head*head_dim]
    # q_ptr, k_ptr, v_ptr: [B, n_head, T, head_dim]
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total  # total = B * n_head * T * head_dim

    # Decode index: offs -> (b, nh, t, hd)
    hd = offs % head_dim
    t = (offs // head_dim) % T
    nh = (offs // (head_dim * T)) % n_head
    b = offs // (head_dim * T * n_head)

    # Source index in qkv: [B*T, 3*n_head*head_dim]
    # q is at offset 0, k at n_head*head_dim, v at 2*n_head*head_dim
    C = n_head * head_dim
    src_row = b * T + t
    q_col = nh * head_dim + hd
    k_col = C + nh * head_dim + hd
    v_col = 2 * C + nh * head_dim + hd

    q_src = src_row * (3 * C) + q_col
    k_src = src_row * (3 * C) + k_col
    v_src = src_row * (3 * C) + v_col

    q_bias_idx = q_col
    k_bias_idx = k_col
    v_bias_idx = v_col

    q_val = tl.load(qkv_ptr + q_src, mask=mask, other=0.0) + tl.load(bias_ptr + q_bias_idx, mask=mask, other=0.0)
    k_val = tl.load(qkv_ptr + k_src, mask=mask, other=0.0) + tl.load(bias_ptr + k_bias_idx, mask=mask, other=0.0)
    v_val = tl.load(qkv_ptr + v_src, mask=mask, other=0.0) + tl.load(bias_ptr + v_bias_idx, mask=mask, other=0.0)

    # dst index: [B, n_head, T, head_dim]
    dst = b * (n_head * T * head_dim) + nh * (T * head_dim) + t * head_dim + hd

    tl.store(q_ptr + dst, q_val, mask=mask)
    tl.store(k_ptr + dst, k_val, mask=mask)
    tl.store(v_ptr + dst, v_val, mask=mask)


# ============================================================
# Kernel 3: Fused attention scores: QK^T * scale + causal mask + softmax
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_T': 64}, num_warps=4),
        triton.Config({'BLOCK_T': 128}, num_warps=8),
        triton.Config({'BLOCK_T': 256}, num_warps=8),
        triton.Config({'BLOCK_T': 512}, num_warps=16),
    ],
    key=['T'],
)
@triton.jit
def attn_softmax_kernel(
    att_ptr, out_ptr,
    B, n_head, T,
    BLOCK_T: tl.constexpr,
):
    # att_ptr: [B, n_head, T, T] - raw attention scores (already scaled)
    # Apply causal mask and softmax over last dim
    # One program per (b, nh, t_row)
    pid = tl.program_id(0)
    total_rows = B * n_head * T
    if pid >= total_rows:
        return

    b = pid // (n_head * T)
    nh = (pid // T) % n_head
    t_row = pid % T

    row_ptr = att_ptr + (b * n_head * T * T + nh * T * T + t_row * T)
    out_row_ptr = out_ptr + (b * n_head * T * T + nh * T * T + t_row * T)

    # Find max (with causal mask: positions > t_row get -inf)
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, T, BLOCK_T):
        cols = off + tl.arange(0, BLOCK_T)
        mask = cols < T
        causal_mask = cols <= t_row
        val = tl.load(row_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        val = tl.where(causal_mask, val, float('-inf'))
        row_max = tl.maximum(row_max, tl.max(tl.where(mask, val, float('-inf')), axis=0))

    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, T, BLOCK_T):
        cols = off + tl.arange(0, BLOCK_T)
        mask = cols < T
        causal_mask = cols <= t_row
        val = tl.load(row_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        val = tl.where(causal_mask, val, float('-inf'))
        exp_val = tl.exp(val - row_max)
        exp_val = tl.where(mask, exp_val, 0.0)
        row_sum += tl.sum(exp_val, axis=0)

    # Store normalized
    for off in range(0, T, BLOCK_T):
        cols = off + tl.arange(0, BLOCK_T)
        mask = cols < T
        causal_mask = cols <= t_row
        val = tl.load(row_ptr + cols, mask=mask, other=float('-inf')).to(tl.float32)
        val = tl.where(causal_mask, val, float('-inf'))
        exp_val = tl.exp(val - row_max)
        softmax_val = exp_val / row_sum
        tl.store(out_row_ptr + cols, softmax_val.to(tl.float32), mask=mask)


# ============================================================
# Kernel 4: Transpose + reshape attn output [B, nh, T, hd] -> [B*T, C]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def transpose_reshape_kernel(
    src_ptr, dst_ptr,
    B, n_head, T, head_dim,
    total: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # src: [B, n_head, T, head_dim]
    # dst: [B*T, n_head*head_dim]
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # dst index: offs -> (b, t, nh, hd)
    hd = offs % head_dim
    nh = (offs // head_dim) % n_head
    t = (offs // (head_dim * n_head)) % T
    b = offs // (head_dim * n_head * T)

    # src index: [B, n_head, T, head_dim]
    src_idx = b * (n_head * T * head_dim) + nh * (T * head_dim) + t * head_dim + hd

    val = tl.load(src_ptr + src_idx, mask=mask, other=0.0)

    # dst index: [B*T, n_head*head_dim] = [B*T, C]
    C = n_head * head_dim
    dst_idx = (b * T + t) * C + nh * head_dim + hd

    tl.store(dst_ptr + dst_idx, val, mask=mask)


# ============================================================
# Kernel 5: Fused bias add + residual add
# ============================================================
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
def bias_add_residual_kernel(
    x_ptr, bias_ptr, residual_ptr, out_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias_idx = offs % 768
    b = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x + b + r
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


# ============================================================
# Kernel 6: Fused GELU activation (NewGELU) + bias add
# ============================================================
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
def gelu_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    N, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias_idx = offs % C
    b = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0).to(tl.float32)
    x = x + b

    # NewGELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    sqrt_2_over_pi = 0.7978845608028654
    x3 = x * x * x
    inner = sqrt_2_over_pi * (x + 0.044715 * x3)
    tanh_val = tl.math.tanh(inner)
    gelu = 0.5 * x * (1.0 + tanh_val)

    tl.store(out_ptr + offs, gelu.to(tl.float32), mask=mask)


# ============================================================
# Kernel 7: Fused bias add + residual for MLP output
# ============================================================
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
def mlp_bias_residual_kernel(
    x_ptr, bias_ptr, residual_ptr, out_ptr,
    N, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    bias_idx = offs % C
    b = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(residual_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = x + b + r
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


# ============================================================
# Tiled GEMM kernel (for matmul operations)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

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
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


# ============================================================
# Batched GEMM for attention: [B*nh, T, hd] x [B*nh, hd, T] -> [B*nh, T, T]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['T', 'head_dim'],
)
@triton.jit
def batched_qk_kernel(
    q_ptr, k_ptr, out_ptr,
    B_nh, T, head_dim,
    scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # q: [B_nh, T, head_dim], k: [B_nh, T, head_dim] (k already transposed logically)
    # out: [B_nh, T, T]
    batch_id = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    q_base = q_ptr + batch_id * T * head_dim
    k_base = k_ptr + batch_id * T * head_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(head_dim, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        q_ptrs = q_base + offs_m[:, None] * head_dim + k_offs[None, :]
        # k is transposed: [head_dim, T], so k[hd, t] = k_orig[t, hd]
        k_ptrs = k_base + offs_n[None, :] * head_dim + k_offs[:, None]

        q_mask = (offs_m[:, None] < T) & (k_offs[None, :] < head_dim)
        k_mask = (k_offs[:, None] < head_dim) & (offs_n[None, :] < T)

        q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)
        k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

        acc = tl.dot(q_tile, k_tile, acc=acc)

    acc = acc * scale

    out_base = out_ptr + batch_id * T * T
    out_ptrs = out_base + offs_m[:, None] * T + offs_n[None, :]
    out_mask = (offs_m[:, None] < T) & (offs_n[None, :] < T)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Batched GEMM for attention output: [B_nh, T, T] x [B_nh, T, hd] -> [B_nh, T, hd]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['T', 'head_dim'],
)
@triton.jit
def batched_av_kernel(
    att_ptr, v_ptr, out_ptr,
    B_nh, T, head_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # att: [B_nh, T, T], v: [B_nh, T, head_dim]
    # out: [B_nh, T, head_dim]
    batch_id = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    att_base = att_ptr + batch_id * T * T
    v_base = v_ptr + batch_id * T * head_dim

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(T, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        att_ptrs = att_base + offs_m[:, None] * T + k_offs[None, :]
        v_ptrs = v_base + k_offs[:, None] * head_dim + offs_n[None, :]

        att_mask = (offs_m[:, None] < T) & (k_offs[None, :] < T)
        v_mask = (k_offs[:, None] < T) & (offs_n[None, :] < head_dim)

        att_tile = tl.load(att_ptrs, mask=att_mask, other=0.0)
        v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)

        acc = tl.dot(att_tile, v_tile, acc=acc)

    out_base = out_ptr + batch_id * T * head_dim
    out_ptrs = out_base + offs_m[:, None] * head_dim + offs_n[None, :]
    out_mask = (offs_m[:, None] < T) & (offs_n[None, :] < head_dim)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    M = x.numel() // x.shape[-1]
    N = x.shape[-1]
    out = torch.empty_like(x)
    grid = (M,)
    layer_norm_kernel[grid](
        x, weight, bias, out,
        M, N, eps,
    )
    return out


def triton_matmul(a, b):
    # a: [M, K], b: [K, N]
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
    )
    return c


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
    n_head = 8
    head_dim = C // n_head  # 96
    B_nh = B * n_head

    # ---- LayerNorm 1 ----
    ln1_out = triton_layer_norm(x, p_ln_1_weight, p_ln_1_bias)

    # ---- Linear: QKV projection [B*T, C] x [C, 3C] -> [B*T, 3C] ----
    ln1_2d = ln1_out.reshape(B * T, C)
    # p_attn_c_attn_weight: [2304, 768] -> need [768, 2304] for matmul
    w_qkv = p_attn_c_attn_weight  # [2304, 768]
    qkv_mm = triton_matmul(ln1_2d, w_qkv.t().contiguous())  # [B*T, 2304]

    # ---- Split QKV + bias + reshape to [B, n_head, T, head_dim] ----
    q = torch.empty((B, n_head,