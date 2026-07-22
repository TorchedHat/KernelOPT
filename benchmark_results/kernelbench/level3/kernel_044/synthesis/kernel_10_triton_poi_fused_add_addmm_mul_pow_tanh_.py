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
        triton.Config({'BLOCK_SIZE': 768}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
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
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, out.to(tl.float32), mask=mask)


# ============================================================
# Kernel 2: GEMM using tl.dot (M x K) x (K x N) -> (M x N)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
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
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (mask_k[None, :]), other=0.0)
        b = tl.load(b_ptrs, mask=(mask_k[:, None]) & (offs_n[None, :] < N), other=0.0)
        
        acc = tl.dot(a, b, acc, allow_tf32=False)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


# ============================================================
# Kernel 3: Fused attention: add bias, mask, softmax
# Input: att [B, nh, T, T], causal_mask [1,1,T,T]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['T'],
)
@triton.jit
def attn_softmax_kernel(
    att_ptr, bias_ptr, out_ptr,
    B, nh, T, scale,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one row of the attention matrix
    row_idx = tl.program_id(0)
    b = row_idx // (nh * T)
    rem = row_idx % (nh * T)
    h = rem // T
    t = rem % T
    
    att_row = att_ptr + row_idx * T
    out_row = out_ptr + row_idx * T
    # bias is [1, 1, T, T], row t
    bias_row = bias_ptr + t * 1024  # max_seqlen=1024
    
    # Scale and find max (with masking)
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, T, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < T
        att_val = tl.load(att_row + cols, mask=mask, other=0.0).to(tl.float32)
        att_val = att_val * scale
        # Apply causal mask: bias==0 means masked
        b_val = tl.load(bias_row + cols, mask=mask, other=0.0).to(tl.float32)
        att_val = tl.where((b_val == 0.0) & mask, float('-inf'), att_val)
        att_val = tl.where(mask, att_val, float('-inf'))
        row_max = tl.maximum(row_max, tl.max(att_val, axis=0))
    
    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, T, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < T
        att_val = tl.load(att_row + cols, mask=mask, other=0.0).to(tl.float32)
        att_val = att_val * scale
        b_val = tl.load(bias_row + cols, mask=mask, other=0.0).to(tl.float32)
        att_val = tl.where((b_val == 0.0) & mask, float('-inf'), att_val)
        att_val = tl.where(mask, att_val, float('-inf'))
        exp_val = tl.exp(att_val - row_max)
        exp_val = tl.where(mask, exp_val, 0.0)
        row_sum += tl.sum(exp_val, axis=0)
    
    # Store normalized
    for off in range(0, T, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < T
        att_val = tl.load(att_row + cols, mask=mask, other=0.0).to(tl.float32)
        att_val = att_val * scale
        b_val = tl.load(bias_row + cols, mask=mask, other=0.0).to(tl.float32)
        att_val = tl.where((b_val == 0.0) & mask, float('-inf'), att_val)
        att_val = tl.where(mask, att_val, float('-inf'))
        out_val = tl.exp(att_val - row_max) / row_sum
        out_val = tl.where(mask, out_val, 0.0)
        tl.store(out_row + cols, out_val.to(tl.float32), mask=mask)


# ============================================================
# Kernel 4: Fused GELU (NewGELU)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
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
    out = 0.5 * x * (1.0 + triton.language.extra.libdevice.tanh(inner))
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


# ============================================================
# Kernel 5: Add residual
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
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


# ============================================================
# Kernel 6: Transpose [B, nh, T, hs] -> [B, T, nh, hs] (contiguous)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def transpose_1324_kernel(
    src_ptr, dst_ptr,
    B, nh, T, hs,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # src: [B, nh, T, hs] with strides [nh*T*hs, T*hs, hs, 1]
    # dst: [B, T, nh, hs] with strides [T*nh*hs, nh*hs, hs, 1]
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Decode dst index [B, T, nh, hs]
    hs_idx = offs % hs
    nh_idx = (offs // hs) % nh
    t_idx = (offs // (hs * nh)) % T
    b_idx = offs // (hs * nh * T)
    
    # Compute src index [B, nh, T, hs]
    src_idx = b_idx * (nh * T * hs) + nh_idx * (T * hs) + t_idx * hs + hs_idx
    
    val = tl.load(src_ptr + src_idx, mask=mask, other=0.0)
    tl.store(dst_ptr + offs, val, mask=mask)


# ============================================================
# Kernel 7: Rearrange QKV from [B, T, 2304] to [B, nh, T, hs]
# For Q: offset=0, K: offset=768, V: offset=1536
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def extract_qkv_kernel(
    qkv_ptr, bias_ptr, out_ptr,
    B, T, nh, hs, C,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Output: [B, nh, T, hs] contiguous
    # src: qkv[B, T, 3*C] at offset qkv_offset in last dim
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    
    # Decode output index [B, nh, T, hs]
    hs_idx = offs % hs
    t_idx = (offs // hs) % T
    nh_idx = (offs // (hs * T)) % nh
    b_idx = offs // (hs * T * nh)
    
    # src index in qkv: [B, T, 3*C] -> [b, t, qkv_offset + nh_idx*hs + hs_idx]
    src_col = qkv_offset + nh_idx * hs + hs_idx
    src_idx = b_idx * (T * 3 * C) + t_idx * (3 * C) + src_col
    
    # bias index
    bias_idx = src_col
    
    val = tl.load(qkv_ptr + src_idx, mask=mask, other=0.0).to(tl.float32)
    b_val = tl.load(bias_ptr + bias_idx, mask=mask, other=0.0).to(tl.float32)
    val = val + b_val
    tl.store(out_ptr + offs, val.to(tl.float32), mask=mask)


# ============================================================
# Batched matmul for attention: [B*nh, T, hs] x [B*nh, hs, T] -> [B*nh, T, T]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['T', 'hs'],
)
@triton.jit
def batched_qk_matmul_kernel(
    q_ptr, k_ptr, out_ptr,
    B_nh, T, hs,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # q: [B_nh, T, hs], k: [B_nh, hs, T] (k is already transposed)
    # out: [B_nh, T, T]
    batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    q_base = q_ptr + batch * T * hs
    k_base = k_ptr + batch * hs * T
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(hs, BLOCK_K)):
        k_off = k * BLOCK_K
        k_remaining = hs - k_off
        mask_k = offs_k < k_remaining
        
        # q[offs_m, k_off+offs_k]
        q_ptrs = q_base + offs_m[:, None] * hs + (k_off + offs_k[None, :])
        q_mask = (offs_m[:, None] < T) & (mask_k[None, :])
        a = tl.load(q_ptrs, mask=q_mask, other=0.0)
        
        # k[k_off+offs_k, offs_n] (k is [hs, T])
        k_ptrs = k_base + (k_off + offs_k[:, None]) * T + offs_n[None, :]
        k_mask = (mask_k[:, None]) & (offs_n[None, :] < T)
        b = tl.load(k_ptrs, mask=k_mask, other=0.0)
        
        acc = tl.dot(a, b, acc, allow_tf32=False)
    
    out_base = out_ptr + batch * T * T
    out_ptrs = out_base + offs_m[:, None] * T + offs_n[None, :]
    mask = (offs_m[:, None] < T) & (offs_n[None, :] < T)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


# ============================================================
# Batched matmul: att [B_nh, T, T] x v [B_nh, T, hs] -> [B_nh, T, hs]
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['T', 'hs'],
)
@triton.jit
def batched_av_matmul_kernel(
    att_ptr, v_ptr, out_ptr,
    B_nh, T, hs,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # att: [B_nh, T, T], v: [B_nh, T, hs]
    # out: [B_nh, T, hs]
    batch = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    att_base = att_ptr + batch * T * T
    v_base = v_ptr + batch * T * hs
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, tl.cdiv(T, BLOCK_K)):
        k_off = k * BLOCK_K
        k_remaining = T - k_off
        mask_k = offs_k < k_remaining
        
        att_ptrs = att_base + offs_m[:, None] * T + (k_off + offs_k[None, :])
        att_mask = (offs_m[:, None] < T) & (mask_k[None, :])
        a = tl.load(att_ptrs, mask=att_mask, other=0.0)
        
        v_ptrs = v_base + (k_off + offs_k[:, None]) * hs + offs_n[None, :]
        v_mask = (mask_k[:, None]) & (offs_n[None, :] < hs)
        b = tl.load(v_ptrs, mask=v_mask, other=0.0)
        
        acc = tl.dot(a, b, acc, allow_tf32=False)
    
    out_base = out_ptr + batch * T * hs
    out_ptrs = out_base + offs_m[:, None] * hs + offs_n[None, :]
    mask = (offs_m[:, None] < T) & (offs_n[None, :] < hs)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask)


def launch_layer_norm(x, weight, bias, eps=1e-5):
    B, T, C = x.shape
    M = B * T
    out = torch.empty_like(x)
    grid = (M,)
    layer_norm_kernel[grid](
        x, weight, bias, out,
        M, C, eps,
    )
    return out


def launch_matmul(a, b, bias=None):
    # a: [M, K], b: [K, N] (b is already transposed if needed)
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    out = torch.empty((M, N), device=a.device, dtype=torch.float32)
    has_bias = bias is not None
    if not has_bias:
        bias = torch.empty(1, device=a.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_kernel[grid](
        a, b, out, bias,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=has_bias,
    )
    return out


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
    hs = C // n_head  # 96
    
    # ---- LayerNorm 1 ----
    ln1_out = launch_layer_norm(x, p_ln_1_weight, p_ln_1_bias)
    
    # ---- QKV projection ----
    # ln1_out: [B, T, C] -> [B*T, C]
    ln1_flat = ln1_out.reshape(B * T, C)
    # p_attn_c_attn_weight: [2304, 768] -> need [768, 2304] for matmul
    qkv_weight = p_attn_c_attn_weight  # [2304, 768]
    # We want: [B*T, C] x [C, 2304] -> [B*T, 2304]
    qkv_flat = launch_matmul(ln1_flat, qkv_weight.t().contiguous(), p_attn_c_attn_bias)
    qkv = qkv_flat.reshape(B, T, 3 * C)
    
    # ---- Split Q, K, V and reshape to [B, nh, T, hs] ----
    # Q: qkv[:, :, 0:768], K: qkv[:, :, 768:1536], V: qkv[:, :, 1536:2304]
    N_qkv = B * n_head * T