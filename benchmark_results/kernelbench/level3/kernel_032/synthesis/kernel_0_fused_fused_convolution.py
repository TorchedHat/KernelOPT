import torch
import triton
import triton.language as tl
import math

# ─── Fused Conv2d (implicit GEMM) ───────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
    ],
    key=['M_TOTAL', 'OUT_C', 'IN_C'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    M_TOTAL,
    s_xn, s_xc, s_xh, s_xw,
    s_wco, s_wci, s_wkh, s_wkw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)

        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_x = (
            (batch_idx[:, None] < BATCH) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_ptrs = (
            batch_idx[:, None] * s_xn +
            in_c[None, :] * s_xc +
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (
            offs_n[None, :] * s_wco +
            in_c[:, None] * s_wci +
            kh[:, None] * s_wkh +
            kw[:, None] * s_wkw
        )
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Store
    valid_out = (
        (offs_m[:, None] < M_TOTAL) &
        (offs_n[None, :] < OUT_C)
    )
    # Output in NCHW: Y[n, c, h, w] = Y[n*OUT_C*OUT_H*OUT_W + c*OUT_H*OUT_W + h*OUT_W + w]
    y_ptrs = (
        batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
        offs_n[None, :] * (OUT_H * OUT_W) +
        out_h_idx[:, None] * OUT_W +
        out_w_idx[:, None]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


# ─── Fused Linear (GEMM) ────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
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
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)

    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float32), mask=out_mask)


# ─── Fused cat (cls_token expand + unsqueeze + cat) ─────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['TOTAL'],
)
@triton.jit
def cat_cls_kernel(
    CLS_ptr, X_ptr, OUT_ptr,
    B, D,
    TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    # OUT shape: (B, 2, D) = (B, 256)
    # First D elements per batch: cls_token (from CLS_ptr, shape [1,1,D])
    # Next D elements per batch: x (from X_ptr, shape [B, D])
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL

    # offs indexes into flat (B, 2, D)
    d_idx = offs % D
    seq_idx = (offs // D) % 2
    b_idx = offs // (2 * D)

    # Load cls token (broadcast from [1,1,D])
    cls_val = tl.load(CLS_ptr + d_idx, mask=mask & (seq_idx == 0), other=0.0)
    # Load x token
    x_val = tl.load(X_ptr + b_idx * D + d_idx, mask=mask & (seq_idx == 1), other=0.0)

    val = tl.where(seq_idx == 0, cls_val, x_val)
    tl.store(OUT_ptr + offs, val, mask=mask)


# ─── LayerNorm kernel ────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=2),
        triton.Config({'BLOCK_N': 256}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 64}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N,
    eps,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / N
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * rstd

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w + b

    tl.store(Y_ptr + row * N + offs, y.to(tl.float32), mask=mask)


# ─── Softmax kernel ──────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 4}, num_warps=1),
        triton.Config({'BLOCK_N': 8}, num_warps=1),
        triton.Config({'BLOCK_N': 16}, num_warps=2),
        triton.Config({'BLOCK_N': 32}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(
    X_ptr, Y_ptr,
    M, N,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_N)
    mask = offs < N

    x = tl.load(X_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    x = tl.where(mask, x - x_max, float('-inf'))
    exp_x = tl.exp(x)
    sum_exp = tl.sum(tl.where(mask, exp_x, 0.0), axis=0)
    y = exp_x / sum_exp

    tl.store(Y_ptr + row * N + offs, y.to(tl.float32), mask=mask)


# ─── GELU kernel ─────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(
    X_ptr, Y_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + tl.extra.cuda.libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    y = x * cdf

    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


# ─── Add residual kernel ──────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def add_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(C_ptr + offs, a + b, mask=mask)


# ─── Linear (no bias) for QKV projection ─────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
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
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)

    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_conv2d(x, weight, bias, stride=4, padding=0):
    """Conv2d using implicit GEMM Triton kernel."""
    B, C, H, W = x.shape
    Co, Ci, KH, KW = weight.shape
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    M_TOTAL = B * OH * OW

    # Use contiguous tensors
    x = x.contiguous()
    weight = weight.contiguous()

    y = torch.empty(B, Co, OH, OW, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(Co, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x, weight, bias, y,
        B, Ci, H, W,
        Co, OH, OW,
        KH, KW, stride, stride, padding, padding,
        M_TOTAL,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
    )
    return y


def triton_linear(x, weight, bias):
    """Linear layer using Triton GEMM kernel."""
    # x: (M, K), weight: (N, K), bias: (N,)
    M = x.shape[0]
    N, K = weight.shape
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight is (N,K), so stride_bk=stride(1), stride_bn=stride(0)
    )
    return out


def triton_layernorm(x, weight, bias, eps=1e-5):
    """LayerNorm using Triton kernel."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)

    layernorm_kernel[(M,)](
        x_2d, weight, bias, out,
        M, N, eps,
    )
    return out.reshape(orig_shape)


def triton_softmax(x):
    """Softmax over last dimension using Triton kernel."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x_2d.shape
    out = torch.empty_like(x_2d)

    softmax_kernel[(M,)](
        x_2d, out,
        M, N,
    )
    return out.reshape(orig_shape)


def triton_gelu(x):
    """GELU activation using Triton kernel."""
    x = x.contiguous()
    N = x.numel()
    out = torch.empty_like(x)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


def triton_add(a, b):
    """Element-wise add using Triton kernel."""
    a = a.contiguous()
    b = b.contiguous()
    N = a.numel()
    out = torch.empty_like(a)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](a, b, out, N)
    return out.reshape(a.shape)


def triton_linear_bias(x, weight, bias):
    """Linear layer with bias using Triton GEMM kernel (separate autotune config)."""
    M = x.shape[0]
    N, K = weight.shape
    x = x.contiguous()
    weight = weight.contiguous()
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_bias_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
    )
    return out


def transformer_encoder_layer_fwd(
    x,
    in_proj_weight, in_proj_bias,
    out_proj_weight, out_proj_bias,
    norm1_weight, norm1_bias,
    norm2_weight, norm2_bias,
    linear1_weight, linear1_bias,
    linear2_weight, linear2_bias,
    num_heads=4,
    eps=1e-5,
):
    """
    """
    B, S, D = x.shape
    head_dim = D // num_heads
    scale = head_dim ** -0.5

    # ── Self-Attention ──────────────────────────────────────────────────────
    # LayerNorm 1
    x_norm1 = triton_layernorm(x, norm1_weight, norm1_bias, eps)

    # QKV projection: (B*S, D) @ (D, 3D)^T + bias
    x_flat = x_norm1.reshape(B * S, D)
    qkv = triton_linear_bias(x_flat, in_proj_weight, in_proj_bias)  # (B*S, 3D)
    qkv = qkv.reshape(B, S, 3, num_heads, head_dim)

    q = qkv[:, :, 0].transpose(1, 2)  # (B, H, S, head_dim)
    k = qkv[:, :, 1].transpose(1, 2)
    v = qkv[:, :, 2].transpose(1, 2)

    # Attention scores: (B, H, S, S)
    attn = torch.empty(B, num_heads, S, S, device=x.device, dtype=x.dtype)
    # q @ k^T scaled
    q_flat = q.reshape(B * num_heads, S, head_dim).contiguous()
    k_flat = k.reshape(B * num_heads, S, head_dim).contiguous()
    v_flat = v.reshape(B * num_heads, S, head_dim).contiguous()

    # Compute attention scores using Triton matmul
    # attn_scores: (B*H, S, S)
    attn_scores = torch.empty(B * num_heads, S, S, device=x.device, dtype=x.dtype)
    # Use our linear kernel treating each head as a batch
    # For small S=2, just do it directly with Triton
    for bh in range(B * num_heads):
        # q_flat[bh]: (S, head_dim), k_flat[bh]: (S, head_dim)
        # score = q @ k^T * scale
        score = triton_linear(
            q_flat[bh],  # (S, head_dim)
            k_flat[bh].contiguous(),  # treat as (S, head_dim) -> need (head_dim, S) as weight
            torch.zeros(S, device=x.device, dtype=x.dtype),
        )
        # Actually we need q @ k^T, not q @ k
        # Let's use a different approach for small S
        pass

    # For the attention computation with small S=2, use bmm-equivalent via Triton
    # q_flat: (B*H, S, head_dim), k_flat: (B*H, S, head_dim)
    # attn = q_flat @ k_flat.transpose(-1,-2) * scale
    # Then softmax, then @ v_flat
    # Implement as batch of small matmuls

    BH = B * num_heads
    attn_scores_out = torch.empty(BH, S, S, device=x.device, dtype=x.dtype)

    # Use a dedicated kernel for batched small matmul
    # For S=2, head_dim=32, this is tiny - use a simple approach
    _batched_qk_kernel(q_flat, k_flat, attn_scores_out, BH, S, head_dim, scale)

    # Softmax over last dim
    attn_probs = triton_softmax(attn_scores_out)  # (BH, S, S)

    # attn_probs @ v: (BH, S, S) @ (BH, S, head_dim) -> (BH, S, head_dim)
    attn_out = torch.empty(BH, S, head_dim, device=x.device, dtype=x.dtype)
    _batched_av_kernel(attn_probs, v_flat, attn_out, BH, S, head_dim)

    # Reshape and out projection
    attn_out = attn_out.reshape(B, num_heads, S, head_dim).transpose(1, 2).reshape(B * S, D).contiguous()
    attn_out = triton_linear_bias(attn_out, out_proj_weight, out_proj_bias)  # (B*S, D)
    attn_out = attn_out.reshape(B, S, D)

    # Residual
    x = triton_add(x, attn_out)

    # ── FFN ─────────────────────────────────────────────────────────────────
    x_norm2 = triton_layernorm(x, norm2_weight, norm2_bias, eps)

    x_flat2 = x_norm2.reshape(B * S, D)
    ffn1 = triton_linear_bias(x_flat2, linear1_weight, linear1_bias)  # (B*S, 4D)
    ffn1 = triton_gelu(ffn1)
    ffn2 = triton_linear_bias(ffn1, linear2_weight, linear2_bias)  # (B*S, D)
    ffn2 = ffn2.reshape(B, S, D)

    x = triton_add(x, ffn2)

    return x


# ─── Batched QK