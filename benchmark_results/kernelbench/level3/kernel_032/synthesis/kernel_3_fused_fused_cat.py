import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─── Fused Conv2d (implicit GEMM) ───────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    M_TOTAL, TOTAL_K,
    s_xn, s_xc, s_xh, s_xw,
    s_wco, s_wci, s_wkh, s_wkw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_x = (
            (batch[:, None] < BATCH) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_ptrs = (X_ptr +
                  batch[:, None] * s_xn +
                  in_c[None, :] * s_xc +
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_wco +
                  in_c[:, None] * s_wci +
                  kh[:, None] * s_wkh +
                  kw[:, None] * s_wkw)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    # Output in NCHW: Y[n, c, h, w] = Y[n*OUT_C*OUT_H*OUT_W + c*OUT_H*OUT_W + h*OUT_W + w]
    out_idx = (batch[:, None] * OUT_C * OUT_H * OUT_W +
               offs_n[None, :] * OUT_H * OUT_W +
               out_h[:, None] * OUT_W +
               out_w[:, None])
    tl.store(Y_ptr + out_idx, acc.to(tl.float32), mask=mask_out)


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

        mask_a = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        mask_b = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask_c)


# ─── CLS token cat kernel ───────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['B', 'D'],
)
@triton.jit
def cat_cls_kernel(
    cls_ptr, x_ptr, out_ptr,
    B, D,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch item, one token (0=cls, 1=x)
    pid = tl.program_id(0)
    batch = pid // 2
    token = pid % 2

    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    if token == 0:
        # CLS token: shape [1, 1, D], broadcast over batch
        src = tl.load(cls_ptr + offs, mask=mask, other=0.0)
    else:
        src = tl.load(x_ptr + batch * D + offs, mask=mask, other=0.0)

    out_off = batch * 2 * D + token * D + offs
    tl.store(out_ptr + out_off, src, mask=mask)


# ─── LayerNorm kernel ────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['D'],
)
@triton.jit
def layernorm_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, D,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < D

    x = tl.load(x_ptr + row * D + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / D
    diff = tl.where(mask, x - mean, 0.0)
    var = tl.sum(diff * diff, axis=0) / D
    inv_std = 1.0 / libdevice.sqrt(var + eps)

    w = tl.load(w_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(b_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    out = diff * inv_std * w + b
    tl.store(out_ptr + row * D + offs, out.to(tl.float32), mask=mask)


# ─── Softmax kernel ──────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 8}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 16}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(
    x_ptr, out_ptr,
    M, N,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    x = tl.load(x_ptr + row * N + offs, mask=mask, other=float('-inf')).to(tl.float32)
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp = tl.sum(tl.where(mask, exp_x, 0.0), axis=0)
    out = exp_x / sum_exp
    tl.store(out_ptr + row * N + offs, out.to(tl.float32), mask=mask)


# ─── GELU kernel ─────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
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
    # GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    out = x * cdf
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


# ─── Add residual kernel ──────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
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


# ─── Attention QKV projection + scaled dot-product attention ─────────────────
# We'll implement multi-head attention in Triton

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
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
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn

        mask_a = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        mask_b = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask_c)


def triton_conv2d(x, weight, bias, stride=4, padding=0):
    """Conv2d using implicit GEMM Triton kernel."""
    B, C, H, W = x.shape
    Co, Ci, KH, KW = weight.shape
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1
    M_TOTAL = B * OH * OW
    TOTAL_K = Ci * KH * KW

    y = torch.empty(B, Co, OH, OW, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(Co, meta['BLOCK_N']),
    )

    conv2d_kernel[grid](
        x, weight, bias, y,
        B, Ci, H, W,
        Co, OH, OW,
        KH, KW, stride, stride, padding, padding,
        M_TOTAL, TOTAL_K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
    )
    return y


def triton_linear(x, weight, bias):
    """Linear layer using Triton GEMM."""
    # x: (M, K), weight: (N, K), bias: (N,)
    # out = x @ weight.T + bias
    M, K = x.shape
    N = weight.shape[0]

    out = torch.empty(M, N, device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight is (N, K), we want K stride then N stride
    )
    return out


def triton_layernorm(x, weight, bias, eps=1e-5):
    """LayerNorm using Triton."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    N, D = x_2d.shape

    out = torch.empty_like(x_2d)

    grid = (N,)
    layernorm_kernel[grid](
        x_2d, weight, bias, out,
        N, D,
        eps,
    )
    return out.reshape(orig_shape)


def triton_softmax(x):
    """Softmax over last dimension."""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape

    out = torch.empty_like(x_2d)
    grid = (M,)
    softmax_kernel[grid](x_2d, out, M, N)
    return out.reshape(orig_shape)


def triton_gelu(x):
    """GELU activation."""
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, out, N)
    return out


def triton_add(a, b):
    """Element-wise add."""
    out = torch.empty_like(a)
    N = a.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](a.reshape(-1), b.reshape(-1), out.reshape(-1), N)
    return out


def triton_matmul(A, B):
    """Matrix multiply A @ B."""
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty(M, N, device=A.device, dtype=A.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


def transformer_encoder_layer(
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

    # ── Self-attention ──────────────────────────────────────────────────────
    # LayerNorm 1
    x_norm = triton_layernorm(x, norm1_weight, norm1_bias, eps)

    # QKV projection: (B*S, D) @ (D, 3D)^T + bias
    x_flat = x_norm.reshape(B * S, D)
    # in_proj_weight: (3D, D), in_proj_bias: (3D,)
    qkv = triton_linear(x_flat, in_proj_weight, in_proj_bias)  # (B*S, 3D)
    qkv = qkv.reshape(B, S, 3, num_heads, head_dim)
    qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, S, head_dim)
    q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, S, head_dim)

    # Attention scores: (B, H, S, S)
    # q @ k^T * scale
    q_flat = q.reshape(B * num_heads, S, head_dim)
    k_flat = k.reshape(B * num_heads, S, head_dim)
    v_flat = v.reshape(B * num_heads, S, head_dim)

    # Compute attention scores for each head
    attn_scores = torch.empty(B * num_heads, S, S, device=x.device, dtype=x.dtype)
    for bh in range(B * num_heads):
        scores = triton_matmul(q_flat[bh], k_flat[bh].t()) * scale
        attn_scores[bh] = scores

    # Softmax
    attn_weights = triton_softmax(attn_scores)  # (B*H, S, S)

    # Weighted sum of values
    attn_out = torch.empty(B * num_heads, S, head_dim, device=x.device, dtype=x.dtype)
    for bh in range(B * num_heads):
        attn_out[bh] = triton_matmul(attn_weights[bh], v_flat[bh])

    # Reshape and project
    attn_out = attn_out.reshape(B, num_heads, S, head_dim)
    attn_out = attn_out.permute(0, 2, 1, 3).reshape(B * S, D)

    # Output projection
    attn_out = triton_linear(attn_out, out_proj_weight, out_proj_bias)
    attn_out = attn_out.reshape(B, S, D)

    # Residual
    x = triton_add(x, attn_out)

    # ── FFN ─────────────────────────────────────────────────────────────────
    x_norm2 = triton_layernorm(x, norm2_weight, norm2_bias, eps)

    # FFN: linear1 -> GELU -> linear2
    x_flat2 = x_norm2.reshape(B * S, D)
    ffn_hidden = triton_linear(x_flat2, linear1_weight, linear1_bias)  # (B*S, 4D)
    ffn_hidden = triton_gelu(ffn_hidden)
    ffn_out = triton_linear(ffn_hidden, linear2_weight, linear2_bias)  # (B*S, D)
    ffn_out = ffn_out.reshape(B, S, D)

    # Residual
    x = triton_add(x, ffn_out)

    return x


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_linear_proj_weight, p_linear_proj_bias,
    p_cls_token,
    p_transformer_layers_0_self_attn_in_proj_bias,
    p_transformer_layers_0_self_attn_in_proj_weight,
    p_transformer_layers_0_self_attn_out_proj_weight,
    p_transformer_layers_0_self_attn_out_proj_bias,
    p_transformer_layers_0_norm1_weight,
    p_transformer_layers_0_norm1_bias,
    p_transformer_layers_0_norm2_weight,
    p_transformer_layers_0_norm2_bias,
    p_transformer_layers_0_linear1_weight,
    p_transformer_layers_0_linear1_bias,
    p_transformer_layers_0_linear2_weight,
    p_transformer_layers_0_linear2_bias,
    p_transformer_layers_1_self_attn_in_proj_bias,
    p_transformer_layers_1_self_attn_in_proj_weight,
    p_transformer_layers_1_self_attn_out_proj_weight,
    p_transformer_layers_1_self_attn_out_proj_bias,
    p_transformer_layers_1_norm1_weight,
    p_transformer_layers_1_norm1_bias,
    p_transformer_layers_1_norm2_weight,
    p_transformer_layers_1_norm2_bias,
    p_transformer_layers_1_linear1_weight,
    p_transformer_layers_1_linear1_bias,
    p_transformer_layers_1_linear2_weight,
    p_transformer_layers_1_linear2_bias,
    p_transformer_layers_2_self_attn_in_proj_bias,
    p_transformer_layers_2_self_attn_in_proj_weight,
    p_transformer_layers_2_self_attn_out_proj_weight,
    p_transformer_layers_2_self_attn_out_proj_bias,
    p_transformer_layers_2_norm1_weight,
    p_transformer_layers_2_norm1_bias,
    p_transformer_layers_2_norm2_weight,
    p_transformer_layers_2_norm2_bias,
    p_transformer_layers_2_linear1_weight,
    p_transformer_layers_2_linear1_bias,
    p_transformer_layers_2_linear2_weight,
    p_transformer_layers_2_linear2_bias,
    p_transformer_layers_3_self_attn_in_proj_bias,
    p_transformer_layers_3_self_attn_in_proj_weight,