import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math
import torch.nn.functional as F


# ============================================================
# Fused LayerNorm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    X_row = X_ptr + row * N
    Y_row = Y_ptr + row * N

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
    rstd = libdevice.rsqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(Y_row + cols, y.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y = torch.empty_like(x_2d)
    grid = (M,)
    layer_norm_kernel[grid](
        x_2d, weight, bias, y,
        M, N, eps,
    )
    return y.reshape(orig_shape)


# ============================================================
# Fused GELU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def gelu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    cdf = 0.5 * (1.0 + libdevice.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))
    y = x * cdf
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


def triton_gelu(x):
    N = x.numel()
    y = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, y, N)
    return y


# ============================================================
# Fused Softmax kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(X_ptr, Y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    X_row = X_ptr + row * N
    Y_row = Y_ptr + row * N

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
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)

    # Normalize
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < N
        x = tl.load(X_row + cols, mask=mask, other=float('-inf')).to(tl.float32)
        y = tl.exp(x - row_max) / row_sum
        tl.store(Y_row + cols, y.to(tl.float32), mask=mask)


def triton_softmax(x, dim=-1):
    # Reshape so last dim is the reduction dim
    if dim != -1 and dim != x.ndim - 1:
        x = x.transpose(dim, -1).contiguous()
        orig_shape = x.shape
        x_2d = x.reshape(-1, x.shape[-1])
        M, N = x_2d.shape
        y = torch.empty_like(x_2d)
        grid = (M,)
        softmax_kernel[grid](x_2d, y, M, N)
        return y.reshape(orig_shape).transpose(dim, -1).contiguous()
    else:
        orig_shape = x.shape
        x_2d = x.reshape(-1, orig_shape[-1])
        M, N = x_2d.shape
        y = torch.empty_like(x_2d)
        grid = (M,)
        softmax_kernel[grid](x_2d, y, M, N)
        return y.reshape(orig_shape)


# ============================================================
# Triton Linear (matmul) kernel - for when we need it
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
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

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, triton.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)
    C_ptrs = C_ptr + offs_m_out[:, None] * stride_cm + offs_n_out[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)


def triton_matmul(A, B):
    """A: (M, K), B: (K, N) -> (M, N)"""
    assert A.ndim == 2 and B.ndim == 2
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    C = torch.empty((M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_kernel[grid](
        A, B, C,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        C.stride(0), C.stride(1),
    )
    return C


def triton_linear(x, weight, bias=None):
    """x: (..., in_features), weight: (out_features, in_features)"""
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    # weight is (out, in), we need x @ weight.T
    w_t = weight.t().contiguous()
    out = triton_matmul(x_2d, w_t)
    if bias is not None:
        out = out + bias
    return out.reshape(*orig_shape[:-1], weight.shape[0])


# ============================================================
# Triton Conv2d kernel (implicit GEMM)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M_TOTAL, TOTAL_K,
    s_xn, s_xc, s_xh, s_xw,
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
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K) &
            (offs_m[:, None] < M_TOTAL)
        )

        x_ptrs = X_ptr + batch[:, None] * s_xn + in_c[None, :] * s_xc + ih * s_xh + iw * s_xw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Weight: (OUT_C, IN_C, KH, KW) -> index: oc * IN_C*KH*KW + kh*KW*IN_C + kw*IN_C + ic
        # But stored as (OUT_C, IN_C, KH, KW) with strides (IN_C*KH*KW, KH*KW, KW, 1)
        # w_idx for (kh, kw, in_c) -> kh * KW * IN_C + kw * IN_C + in_c
        w_k_idx = kh[:, None] * KERNEL_W * IN_C + kw[:, None] * IN_C + in_c[:, None]
        w_ptrs = W_ptr + offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) + w_k_idx
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias and store
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Output stored as (BATCH, OUT_C, OUT_H, OUT_W)
    # index: batch * OUT_C*OUT_H*OUT_W + oc * OUT_H*OUT_W + oh * OUT_W + ow
    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr +
              batch[:, None] * (OUT_C * OUT_H * OUT_W) +
              offs_n[None, :] * (OUT_H * OUT_W) +
              out_h[:, None] * OUT_W +
              out_w[:, None])
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_conv2d(x, weight, bias, stride=4, padding=0):
    """x: (N, C, H, W), weight: (OC, IC, KH, KW)"""
    N, IC, H, W = x.shape
    OC, IC2, KH, KW = weight.shape
    assert IC == IC2
    SH = SW = stride if isinstance(stride, int) else stride[0]
    PH = PW = padding if isinstance(padding, int) else padding[0]
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    # Reorder weight to (OC, KH*KW*IC) for easier indexing
    # Actually keep as (OC, IC, KH, KW) and compute index in kernel
    # But we need contiguous weight in (OC, IC, KH, KW) order
    w_cont = weight.contiguous()
    # Reshape to (OC, KH*KW*IC) - reorder to (OC, KH, KW, IC) first
    w_reordered = w_cont.permute(0, 2, 3, 1).contiguous().reshape(OC, -1)

    x_cont = x.contiguous()
    y = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    M_TOTAL = N * OH * OW
    TOTAL_K = IC * KH * KW

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    conv2d_kernel[grid](
        x_cont, w_reordered, bias, y,
        N, IC, H, W,
        OC, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        M_TOTAL, TOTAL_K,
        x_cont.stride(0), x_cont.stride(1), x_cont.stride(2), x_cont.stride(3),
    )
    return y


# ============================================================
# Triton batched matmul for attention
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['B', 'M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    A_ptr, B_ptr, C_ptr,
    B_batch, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_cb, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_b = tl.program_id(2)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + pid_b * stride_ab + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + pid_b * stride_bb + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, triton.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        k_mask = offs_k < k_remaining
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    C_ptrs = C_ptr + pid_b * stride_cb + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)


def triton_bmm(A, B):
    """A: (batch, M, K), B: (batch, K, N) -> (batch, M, N)"""
    assert A.ndim == 3 and B.ndim == 3
    batch, M, K = A.shape
    batch2, K2, N = B.shape
    assert batch == batch2 and K == K2
    C = torch.empty((batch, M, N), device=A.device, dtype=A.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
        batch,
    )
    bmm_kernel[grid](
        A, B, C,
        batch, M, N, K,
        A.stride(0), A.stride(1), A.stride(2),
        B.stride(0), B.stride(1), B.stride(2),
        C.stride(0), C.stride(1), C.stride(2),
    )
    return C


# ============================================================
# Helper: vector norm
# ============================================================
def triton_vector_norm(x, dim=-1, keepdim=True):
    return torch.linalg.vector_norm(x, dim=dim, keepdim=keepdim)


# ============================================================
# Swin Transformer V2 components
# ============================================================

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def compute_relative_position_bias(
    relative_coords_table,
    relative_position_index,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    num_heads,
):
    # relative_coords_table: (1, 2*Wh-1, 2*Ww-1, 2)
    # MLP: Linear(2, 512) -> ReLU -> Linear(512, num_heads)
    table = relative_coords_table  # (1, 13, 13, 2)
    # fc1
    t = triton_linear(table, cpb_mlp_0_weight, cpb_mlp_0_bias)  # (1, 13, 13, 512)
    # relu
    t = torch.clamp(t, min=0.0)
    # fc2 (no bias)
    t = triton_linear(t, cpb_mlp_2_weight)  # (1, 13, 13, num_heads)
    t = t.view(-1, num_heads)  # (169, num_heads)

    # index
    idx = relative_position_index.view(-1)  # (49*49,)
    relative_position_bias_table = t[idx].view(49, 49, num_heads)
    relative_position_bias = relative_position_bias_table.permute(2, 0, 1).contiguous()
    # sigmoid * 16
    relative_position_bias = torch.sigmoid(relative_position_bias) * 16.0
    return relative_position_bias  # (num_heads, 49, 49)


def swin_window_attention(
    q_bias, v_bias,
    qkv_weight,
    logit_scale,
    relative_coords_table,
    relative_position_index,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
    proj_weight, proj_bias,
    num_heads,
    attn_mask=None,
):
    B_, N, C = x_windows.shape
    head_dim = C // num_heads

    # Build qkv bias
    k_bias = torch.zeros(C, device=q_bias.device, dtype=q_bias.dtype)
    qkv_bias = torch.cat([q_bias, k_bias, v_bias])  # (3*C,)

    # QKV projection
    qkv = triton_linear(x_windows, qkv_weight, qkv_bias)  # (B_, N, 3*C)
    qkv = qkv.reshape(B_, N, 3, num_heads, head