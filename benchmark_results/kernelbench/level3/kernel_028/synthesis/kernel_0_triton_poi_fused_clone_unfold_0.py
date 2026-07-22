import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, eps,
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
    rstd = 1.0 / libdevice.sqrt(var + eps)
    
    # Normalize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, out.to(tl.float32), mask=mask)


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math

# ============================================================
# Fused LayerNorm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * N
    out_ptr = out_ptr + row * N

    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / libdevice.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = (x - mean) * rstd * w + b
        tl.store(out_ptr + cols, out.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    layer_norm_kernel[grid](x_2d, weight, bias, out, N, eps)
    return out.reshape(orig_shape)


# ============================================================
# Fused patch embedding: unfold + reshape (pointwise copy)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def unfold_patch_kernel(
    img_ptr, out_ptr,
    B, C, H, W, P,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    # Output shape: [B, num_patches, C*P*P]
    # num_patches = (H//P) * (W//P)
    # For each output element: batch, patch_idx, channel_pixel
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    nph = H // P
    npw = W // P
    num_patches = nph * npw
    patch_dim = C * P * P

    # Decode output index
    b = offs // (num_patches * patch_dim)
    rem = offs % (num_patches * patch_dim)
    patch_idx = rem // patch_dim
    cpix = rem % patch_dim

    ph = patch_idx // npw
    pw = patch_idx % npw

    c = cpix // (P * P)
    pix = cpix % (P * P)
    pi = pix // P
    pj = pix % P

    ih = ph * P + pi
    iw = pw * P + pj

    img_idx = b * (C * H * W) + c * (H * W) + ih * W + iw
    val = tl.load(img_ptr + img_idx, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


def triton_unfold_patches(img, patch_size):
    B, C, H, W = img.shape
    P = patch_size
    nph = H // P
    npw = W // P
    num_patches = nph * npw
    patch_dim = C * P * P
    numel = B * num_patches * patch_dim
    out = torch.empty(B, num_patches, patch_dim, device=img.device, dtype=img.dtype)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    unfold_patch_kernel[grid](img, out, B, C, H, W, P, numel)
    return out


# ============================================================
# GEMM kernel (M x K) @ (K x N) + bias -> (M x N)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
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
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask)


def triton_linear(x, weight, bias=None):
    # x: (M, K), weight: (N, K) -> out: (M, N)
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_bias_kernel[grid](
        x_2d, weight, bias if bias is not None else x_2d,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(1), weight.stride(0),
        HAS_BIAS=(bias is not None),
    )
    return out.reshape(*orig_shape[:-1], N)


# ============================================================
# Fused add + LayerNorm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def add_layer_norm_kernel(
    x_ptr, residual_ptr, weight_ptr, bias_ptr, out_ptr,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    x_ptr = x_ptr + row * N
    residual_ptr = residual_ptr + row * N
    out_ptr = out_ptr + row * N

    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x + r, axis=0)
    mean = mean / N

    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, (x + r) - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / libdevice.sqrt(var + eps)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        r = tl.load(residual_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(weight_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(bias_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        out = ((x + r) - mean) * rstd * w + b
        tl.store(out_ptr + cols, out.to(tl.float32), mask=mask)


def triton_add_layer_norm(x, residual, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    r_2d = residual.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)
    grid = (M,)
    add_layer_norm_kernel[grid](x_2d, r_2d, weight, bias, out, N, eps)
    return out.reshape(orig_shape)


# ============================================================
# Fused ReLU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def relu_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


def triton_relu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


# ============================================================
# Fused GELU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def gelu_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    out = x * cdf
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def triton_gelu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, numel)
    return out


# ============================================================
# Scaled dot-product attention using flash attention approach
# We'll use a simplified version with tl.dot
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64}, num_warps=8, num_stages=2),
    ],
    key=['SEQ_LEN', 'HEAD_DIM', 'BATCH_HEADS'],
)
@triton.jit
def flash_attn_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    SEQ_LEN, HEAD_DIM, BATCH_HEADS,
    scale,
    stride_qb, stride_qs, stride_qd,
    stride_kb, stride_ks, stride_kd,
    stride_vb, stride_vs, stride_vd,
    stride_ob, stride_os, stride_od,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one (batch_head, block_of_queries)
    bh = tl.program_id(0)
    block_m = tl.program_id(1)

    q_base = q_ptr + bh * stride_qb
    k_base = k_ptr + bh * stride_kb
    v_base = v_ptr + bh * stride_vb
    o_base = out_ptr + bh * stride_ob

    offs_m = block_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, HEAD_DIM)

    # Load Q block
    q_mask = offs_m[:, None] < SEQ_LEN
    q = tl.load(
        q_base + offs_m[:, None] * stride_qs + offs_d[None, :] * stride_qd,
        mask=q_mask & (offs_d[None, :] < HEAD_DIM),
        other=0.0
    ).to(tl.float32)

    # Online softmax accumulators
    m_i = tl.full((BLOCK_M,), float('-inf'), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)

    for n_start in range(0, SEQ_LEN, BLOCK_N):
        offs_n = n_start + tl.arange(0, BLOCK_N)
        kv_mask = offs_n < SEQ_LEN

        k = tl.load(
            k_base + offs_n[None, :] * stride_ks + offs_d[:, None] * stride_kd,
            mask=kv_mask[None, :] & (offs_d[:, None] < HEAD_DIM),
            other=0.0
        ).to(tl.float32)

        # QK^T
        qk = tl.dot(q, k)  # (BLOCK_M, BLOCK_N)
        qk = qk * scale
        # Mask out-of-bounds
        qk = tl.where(kv_mask[None, :], qk, float('-inf'))

        # Online softmax update
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(qk - m_new[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_base + offs_n[:, None] * stride_vs + offs_d[None, :] * stride_vd,
            mask=kv_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0
        ).to(tl.float32)

        acc = tl.dot(p.to(v.dtype), v, acc=acc)
        m_i = m_new

    # Normalize
    acc = acc / l_i[:, None]

    # Store output
    o_mask = offs_m[:, None] < SEQ_LEN
    tl.store(
        o_base + offs_m[:, None] * stride_os + offs_d[None, :] * stride_od,
        acc.to(tl.float32),
        mask=o_mask & (offs_d[None, :] < HEAD_DIM),
    )


def triton_attention(q, k, v, scale):
    # q, k, v: (batch_heads, seq_len, head_dim)
    BH, S, D = q.shape
    out = torch.empty_like(q)
    grid = lambda meta: (BH, triton.cdiv(S, meta['BLOCK_M']))
    flash_attn_kernel[grid](
        q, k, v, out,
        S, D, BH,
        scale,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
    )
    return out


# ============================================================
# Fused cat + add (cls_token concat + pos_embedding add)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def cat_add_kernel(
    cls_ptr, patches_ptr, pos_ptr, out_ptr,
    B, SEQ_LEN, DIM,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # Decode: out[b, s, d]
    b = offs // (SEQ_LEN * DIM)
    rem = offs % (SEQ_LEN * DIM)
    s = rem // DIM
    d = rem % DIM

    is_cls = s == 0
    # cls token: cls_ptr[0, 0, d]
    cls_val = tl.load(cls_ptr + d, mask=mask & is_cls, other=0.0)
    # patch: patches_ptr[b, s-1, d]
    patch_val = tl.load(patches_ptr + b * (SEQ_LEN - 1) * DIM + (s - 1) * DIM + d,
                        mask=mask & ~is_cls, other=0.0)
    # pos embedding: pos_ptr[0, s, d]
    pos_val = tl.load(pos_ptr + s * DIM + d, mask=mask, other=0.0)

    val = tl.where(is_cls, cls_val, patch_val) + pos_val
    tl.store(out_ptr + offs, val, mask=mask)


def triton_cat_add(cls_token, patches, pos_embedding):
    # cls_token: (1, 1, D), patches: (B, N, D), pos_embedding: (1, N+1, D)
    B, N, D = patches.shape
    SEQ_LEN = N + 1
    numel = B * SEQ_LEN * D
    out = torch.empty(B, SEQ_LEN, D, device=patches.device, dtype=patches.dtype)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    cat_add_kernel[grid](
        cls_token, patches, pos_embedding, out,
        B, SEQ_LEN, D, numel,
    )
    return out


# ============================================================
# Main transformer layer function
# ============================================================
def transformer_layer(x, in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
                      norm1_weight, norm1_bias, linear1_weight, linear1_bias,
                      linear2_weight, linear2_bias, norm2_weight, norm2_bias,
                      num_heads=8, head_dim=64):
    B, S, D = x.shape

    # Self-attention
    # QKV projection
    qkv = triton_linear(x, in_proj_weight, in_proj_bias)  # (B, S, 3*D)
    qkv = qkv.reshape(B, S, 3, num_heads, head_dim)
    q = qkv[:, :, 0].permute(0, 2, 1, 3).reshape(B * num_heads, S, head_dim).contiguous()
    k = qkv[:, :, 1].permute(0, 2, 1, 3).reshape(B * num_heads, S, head_dim).contiguous()
    v = qkv[:, :, 2].permute(0, 2, 1, 3).reshape(B * num_heads, S, head_dim).contiguous()

    scale = 1.0