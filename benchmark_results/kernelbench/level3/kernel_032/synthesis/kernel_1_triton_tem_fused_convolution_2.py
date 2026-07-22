import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton Kernels
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
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

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KERNEL_W
        kh_idx = tmp_k // KERNEL_W

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            (offs_m[:, None] < M) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < K)
        )

        x_ptrs = (
            batch_idx[:, None] * stride_xn +
            in_c_idx[None, :] * stride_xc +
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_ptrs = (
            offs_n[None, :] * stride_wco +
            in_c_idx[:, None] * stride_wci +
            kh_idx[:, None] * stride_wkh +
            kw_idx[:, None] * stride_wkw
        )
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Store output: Y is (BATCH, OUT_C, OUT_H, OUT_W) in NCHW
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = (
        batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
        offs_n[None, :] * (OUT_H * OUT_W) +
        out_h_idx[:, None] * OUT_W +
        out_w_idx[:, None]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 256}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
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
    """General linear (matmul + bias) kernel: C = A @ B^T + bias"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
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
    diff = x - mean
    var = tl.sum(diff * diff, axis=0) / N
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = diff * inv_std

    w = tl.load(W_ptr + offs, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = x_norm * w + b

    tl.store(Y_ptr + row * N + offs, y.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_stages=2, num_warps=8),
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
    x_shifted = x - x_max
    exp_x = tl.exp(x_shifted)
    sum_exp = tl.sum(exp_x, axis=0)
    y = exp_x / sum_exp

    tl.store(Y_ptr + row * N + offs, y.to(tl.float32), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=8),
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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=8),
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


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=4, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """General matmul kernel: C = A @ B"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak, mask=a_mask, other=0.0)
        b = tl.load(B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=out_mask)


# ============================================================
# Helper functions
# ============================================================

def triton_conv2d(x, weight, bias, stride=4, padding=0):
    """Conv2d using implicit GEMM Triton kernel"""
    B, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1

    M = B * OH * OW
    N = C_out
    K = C_in * KH * KW

    y = torch.empty((B, C_out, OH, OW), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x, weight, bias, y,
        B, C_in, H, W,
        C_out, OH, OW,
        KH, KW,
        stride, stride,
        padding, padding,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
    )
    return y


def triton_linear(x, weight, bias):
    """Linear layer: y = x @ weight.T + bias"""
    # x: (M, K), weight: (N, K), bias: (N,)
    # We want y = x @ weight.T, so B = weight.T: (K, N)
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, K = x_2d.shape
    N = weight.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    # weight is (N, K), we want to compute x @ weight.T
    # So A = x (M, K), B = weight (N, K) with stride_bk=1, stride_bn=K
    linear_kernel[grid](
        x_2d, weight, bias, y,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.stride(1), weight.stride(0),  # B is weight, access as (K, N): stride_bk=stride(1), stride_bn=stride(0)
    )

    out_shape = orig_shape[:-1] + (N,)
    return y.reshape(out_shape)


def triton_layernorm(x, weight, bias, eps=1e-5):
    """LayerNorm over last dimension"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape

    y = torch.empty_like(x_2d)

    grid = (M,)
    layernorm_kernel[grid](
        x_2d, weight, bias, y,
        M, N, eps,
    )
    return y.reshape(orig_shape)


def triton_softmax(x):
    """Softmax over last dimension"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1])
    M, N = x_2d.shape

    y = torch.empty_like(x_2d)
    grid = (M,)
    softmax_kernel[grid](x_2d, y, M, N)
    return y.reshape(orig_shape)


def triton_gelu(x):
    """GELU activation"""
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, y, N)
    return y


def triton_add(a, b):
    """Element-wise addition"""
    c = torch.empty_like(a)
    N = a.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    add_kernel[grid](a, b, c, N)
    return c


def triton_matmul(a, b):
    """Matrix multiplication: C = A @ B"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
    )
    return c


def triton_transformer_layer(
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

    # --- Self-attention ---
    # LayerNorm 1
    x_norm = triton_layernorm(x, norm1_weight, norm1_bias, eps)

    # QKV projection: (B*S, 3*D)
    x_flat = x_norm.reshape(B * S, D)
    qkv = triton_linear(x_flat, in_proj_weight, in_proj_bias)  # (B*S, 3*D)
    qkv = qkv.reshape(B, S, 3, num_heads, head_dim)

    q = qkv[:, :, 0, :, :].permute(0, 2, 1, 3).contiguous()  # (B, H, S, head_dim)
    k = qkv[:, :, 1, :, :].permute(0, 2, 1, 3).contiguous()
    v = qkv[:, :, 2, :, :].permute(0, 2, 1, 3).contiguous()

    # Attention scores: (B, H, S, S)
    # Reshape for batched matmul
    q_r = q.reshape(B * num_heads, S, head_dim)
    k_r = k.reshape(B * num_heads, S, head_dim)
    v_r = v.reshape(B * num_heads, S, head_dim)

    # attn = q @ k^T * scale
    attn_scores = torch.empty(B * num_heads, S, S, device=x.device, dtype=x.dtype)
    for i in range(B * num_heads):
        attn_scores[i] = triton_matmul(q_r[i], k_r[i].t()) * scale

    # Softmax
    attn_weights = triton_softmax(attn_scores)  # (B*H, S, S)

    # attn_out = attn_weights @ v
    attn_out = torch.empty(B * num_heads, S, head_dim, device=x.device, dtype=x.dtype)
    for i in range(B * num_heads):
        attn_out[i] = triton_matmul(attn_weights[i], v_r[i])

    # Reshape: (B, S, D)
    attn_out = attn_out.reshape(B, num_heads, S, head_dim).permute(0, 2, 1, 3).contiguous()
    attn_out = attn_out.reshape(B * S, D)

    # Out projection
    attn_out = triton_linear(attn_out, out_proj_weight, out_proj_bias)
    attn_out = attn_out.reshape(B, S, D)

    # Residual
    x = triton_add(x, attn_out)

    # --- FFN ---
    x_norm2 = triton_layernorm(x, norm2_weight, norm2_bias, eps)

    x_flat2 = x_norm2.reshape(B * S, D)
    ffn_hidden = triton_linear(x_flat2, linear1_weight, linear1_bias)
    ffn_hidden = triton_gelu(ffn_hidden)
    ffn_out = triton_linear(ffn_hidden, linear2_weight, linear2_bias)
    ffn_out = ffn_out.reshape(B, S, D)

    x = triton_add(x, ffn_out)

    return x


# ============================================================
# Main kernel function
# ============================================================

def kernel_function(
    p_conv1_weight,
    p_conv1_bias,
    p_linear_proj_weight,
    p_linear_proj_bias,
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
    p_transformer_layers_3_self_attn_out_proj_weight,
    p_transformer_layers_3_self_attn_out_proj_bias,
    p_transformer_layers_3_norm1_weight,
    p_transformer_layers_3_norm1_bias,
    p_transformer_layers_3_norm2_weight,
    p_transformer_layers_3_norm2_bias,
    p_transformer_layers_3_linear1_weight,
    p_transformer_layers_3_linear1_bias,
    p_transformer_layers_3_linear2_weight,
    p_transformer_layers_3_linear2_bias,
    p_transformer_layers_4_self_attn_in_proj_bias,
    p_transformer_layers_4_self_attn_in_proj_weight,