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
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(a, b, acc, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def triton_matmul_bias(A, B, bias=None):
    """A: (M, K), B: (K, N) or (N, K) transposed"""
    assert A.ndim == 2 and B.ndim == 2
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.empty((M, N), device=A.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    matmul_bias_kernel[grid](
        A, B, bias if bias is not None else A,
        C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
        HAS_BIAS=(bias is not None),
    )
    return C


# Conv2d as implicit GEMM
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'OUT_C', 'IN_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, bias_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (out_c, kh, kw, in_c)
    s_wo, s_wkh, s_wkw, s_wi,
    # strides for Y (NHWC layout)
    s_yn, s_yh, s_yw, s_yc,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M_TOTAL
    mask_n = offs_n < OUT_C

    # Decode m -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOTAL_K

        # Decode k -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (mask_m[:, None] & mask_k[None, :] &
                 (ih >= 0) & (ih < IN_H) &
                 (iw >= 0) & (iw < IN_W))

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * s_xn +
                  in_c[None, :] * s_xc)
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Weight: (out_c, kh, kw, in_c) -> index [offs_n, kh, kw, in_c]
        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_wo +
                  kh[:, None] * s_wkh +
                  kw[:, None] * s_wkw +
                  in_c[:, None] * s_wi)
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
        acc = acc + bias[None, :]

    # Store output (NHWC)
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * s_yn +
              out_h_idx[:, None] * s_yh +
              out_w_idx[:, None] * s_yw +
              offs_n[None, :] * s_yc)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def triton_conv2d(x, weight, bias, stride=4, padding=0):
    """
    """
    N, C, H, W = x.shape
    out_c, in_c, kh, kw = weight.shape
    out_h = (H + 2 * padding - kh) // stride + 1
    out_w = (W + 2 * padding - kw) // stride + 1

    # Convert to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # Weight: (out_c, kh, kw, in_c)
    w_conv = weight.permute(0, 2, 3, 1).contiguous()

    # Output in NHWC
    y_nhwc = torch.empty((N, out_h, out_w, out_c), device=x.device, dtype=torch.float32)

    M_TOTAL = N * out_h * out_w

    grid = lambda meta: (triton.cdiv(M_TOTAL, meta['BLOCK_M']), triton.cdiv(out_c, meta['BLOCK_N']))

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_conv, bias if bias is not None else x_nhwc,
        y_nhwc,
        N, C, H, W,
        out_c, out_h, out_w,
        kh, kw,
        stride, stride,
        padding, padding,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w_conv.stride(0), w_conv.stride(1), w_conv.stride(2), w_conv.stride(3),
        y_nhwc.stride(0), y_nhwc.stride(1), y_nhwc.stride(2), y_nhwc.stride(3),
        HAS_BIAS=(bias is not None),
    )

    # Convert back to NCHW
    return y_nhwc.permute(0, 3, 1, 2).contiguous()


# LayerNorm kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layernorm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N,
    stride_m,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    X_row = X_ptr + row * stride_m
    Y_row = Y_ptr + row * stride_m

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / tl.sqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y_row + cols, y.to(tl.float32), mask=mask)


def triton_layernorm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x_2d.shape
    y_2d = torch.empty_like(x_2d)

    grid = (M,)
    layernorm_kernel[grid](
        x_2d, weight, bias, y_2d,
        M, N,
        x_2d.stride(0),
        eps,
    )
    return y_2d.reshape(orig_shape)


# Softmax kernel for attention
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(
    X_ptr, Y_ptr,
    M, N,
    stride_m,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return

    X_row = X_ptr + row * stride_m
    Y_row = Y_ptr + row * stride_m

    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_sum += tl.sum(tl.exp(tl.where(mask, x - row_max, float('-inf'))), axis=0)

    # Normalize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        y = tl.exp(tl.where(mask, x - row_max, float('-inf'))) / row_sum
        tl.store(Y_row + cols, y.to(tl.float32), mask=mask)


def triton_softmax(x):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, N = x_2d.shape
    y_2d = torch.empty_like(x_2d)

    grid = (M,)
    softmax_kernel[grid](
        x_2d, y_2d,
        M, N,
        x_2d.stride(0),
    )
    return y_2d.reshape(orig_shape)


# GELU activation kernel
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


def triton_gelu(x):
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, y, N)
    return y


# ============================================================
# Transformer Encoder Layer (manual implementation)
# ============================================================

def triton_linear(x, weight, bias=None):
    """x: (..., K), weight: (N, K) -> (..., N)"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1]).contiguous()
    M, K = x_2d.shape
    N = weight.shape[0]

    # weight is (N, K), we need (K, N) for matmul
    weight_t = weight.t().contiguous()

    out = triton_matmul_bias(x_2d, weight_t, bias)
    return out.reshape(*orig_shape[:-1], N)


def transformer_encoder_layer_fwd(
    x,
    in_proj_weight, in_proj_bias,
    out_proj_weight, out_proj_bias,
    norm1_weight, norm1_bias,
    norm2_weight, norm2_bias,
    linear1_weight, linear1_bias,
    linear2_weight, linear2_bias,
    num_heads,
    eps=1e-5,
):
    """
    """
    B, S, D = x.shape
    head_dim = D // num_heads
    scale = head_dim ** -0.5

    # Self-attention with pre-norm
    residual = x
    x_norm = triton_layernorm(x, norm1_weight, norm1_bias, eps)

    # QKV projection
    # in_proj_weight: (3*D, D), in_proj_bias: (3*D,)
    qkv = triton_linear(x_norm, in_proj_weight, in_proj_bias)  # (B, S, 3*D)

    # Split Q, K, V
    q = qkv[..., :D]       # (B, S, D)
    k = qkv[..., D:2*D]    # (B, S, D)
    v = qkv[..., 2*D:]     # (B, S, D)

    # Reshape for multi-head attention
    q = q.reshape(B, S, num_heads, head_dim).permute(0, 2, 1, 3)  # (B, H, S, head_dim)
    k = k.reshape(B, S, num_heads, head_dim).permute(0, 2, 1, 3)
    v = v.reshape(B, S, num_heads, head_dim).permute(0, 2, 1, 3)

    # Attention scores
    # (B, H, S, head_dim) x (B, H, head_dim, S) -> (B, H, S, S)
    q_2d = q.reshape(B * num_heads, S, head_dim).contiguous()
    k_2d = k.reshape(B * num_heads, S, head_dim).contiguous()
    v_2d = v.reshape(B * num_heads, S, head_dim).contiguous()

    # attn = q @ k^T * scale
    # Use triton matmul for each batch*head
    BH = B * num_heads
    attn_scores = torch.empty(BH, S, S, device=x.device, dtype=torch.float32)
    for bh in range(BH):
        attn_scores[bh] = triton_matmul_bias(q_2d[bh], k_2d[bh].t().contiguous(), None) * scale

    attn_weights = triton_softmax(attn_scores)  # (BH, S, S)

    # attn_weights @ v
    attn_out = torch.empty(BH, S, head_dim, device=x.device, dtype=torch.float32)
    for bh in range(BH):
        attn_out[bh] = triton_matmul_bias(attn_weights[bh], v_2d[bh], None)

    # Reshape back
    attn_out = attn_out.reshape(B, num_heads, S, head_dim).permute(0, 2, 1, 3).reshape(B, S, D).contiguous()

    # Output projection
    attn_out = triton_linear(attn_out, out_proj_weight, out_proj_bias)

    # Residual connection
    x = residual + attn_out

    # FFN with pre-norm
    residual = x
    x_norm = triton_layernorm(x, norm2_weight, norm2_bias, eps)

    # FFN
    ffn_out = triton_linear(x_norm, linear1_weight, linear1_bias)
    ffn_out = triton_gelu(ffn_out)
    ffn_out = triton_linear(ffn_out, linear2_weight, linear2_bias)

    x = residual + ffn_out

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
    p_transformer_layers_4_self_attn_out_proj_weight,
    p_transformer_layers_4_self_attn_out_proj_bias,
    p_transformer_layers_4_norm1_weight,
    p_transformer_layers_4_norm1_bias,
    p_transformer_layers_4_norm2_weight,
    p_transformer_layers_4_norm2_bias,
    p_transformer_layers_4_linear1_weight,
    p_transformer_layers_4_linear1_bias,
    p_transformer_layers_4_linear2_weight,
    p_transformer_layers_4_linear2_bias,
    p_transformer_layers_5_self_attn_in_proj_bias,
    p_transformer_layers_5_self_attn_in_proj_weight,
    p_transformer_layers_5_self_attn_out_proj_weight,
    p_transformer_layers_5_self_attn_out_proj_bias,
    p_transformer_layers_5_norm1_weight,
    p_transformer_layers_5_norm1_bias,
    p_transformer_layers_5_norm2_weight,