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

# ============================================================
# Fused LayerNorm kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
        triton.Config({'BLOCK_N': 1024}, num_warps=16),
    ],
    key=['N'],
)
@triton.jit
def layer_norm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, eps,
    stride_xm, stride_xn,
    stride_ym, stride_yn,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    if row >= M:
        return
    x_base = X_ptr + row * stride_xm
    y_base = Y_ptr + row * stride_ym

    # Compute mean
    mean = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_base + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        mean += tl.sum(x, axis=0)
    mean = mean / N

    # Compute variance
    var = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_base + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        diff = tl.where(mask, x - mean, 0.0)
        var += tl.sum(diff * diff, axis=0)
    var = var / N
    rstd = 1.0 / libdevice.sqrt(var + eps)

    # Normalize and apply weight/bias
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_base + cols * stride_xn, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(W_ptr + cols, mask=mask, other=1.0).to(tl.float32)
        b = tl.load(B_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        y = (x - mean) * rstd * w + b
        tl.store(y_base + cols * stride_yn, y.to(tl.float32), mask=mask)


def triton_layer_norm(x, weight, bias, eps=1e-5):
    orig_shape = x.shape
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y_2d = torch.empty_like(x_2d)
    grid = (M,)
    layer_norm_kernel[grid](
        x_2d, weight, bias, y_2d,
        M, N, eps,
        x_2d.stride(0), x_2d.stride(1),
        y_2d.stride(0), y_2d.stride(1),
    )
    return y_2d.reshape(orig_shape)


# ============================================================
# Fused GELU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
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
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, y, N)
    return y


# ============================================================
# Fused Softmax kernel (row-wise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 64}, num_warps=2),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def softmax_kernel(X_ptr, Y_ptr, M, N, stride_m, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    x_base = X_ptr + row * stride_m
    y_base = Y_ptr + row * stride_m

    # Find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_base + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_max = tl.maximum(row_max, tl.max(x, axis=0))

    # Compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_base + cols, mask=mask, other=float('-inf')).to(tl.float32)
        row_sum += tl.sum(tl.exp(x - row_max), axis=0)

    # Normalize
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_base + cols, mask=mask, other=float('-inf')).to(tl.float32)
        y = tl.exp(x - row_max) / row_sum
        tl.store(y_base + cols, y.to(tl.float32), mask=mask)


def triton_softmax(x, dim=-1):
    # x: arbitrary shape, softmax over last dim
    orig_shape = x.shape
    if dim != -1 and dim != len(orig_shape) - 1:
        # Move dim to last
        x = x.transpose(dim, -1).contiguous()
        result = triton_softmax(x, dim=-1)
        return result.transpose(dim, -1).contiguous()
    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y_2d = torch.empty_like(x_2d)
    grid = (M,)
    softmax_kernel[grid](x_2d, y_2d, M, N, x_2d.stride(0))
    return y_2d.reshape(orig_shape)


# ============================================================
# GEMM kernel (tiled matrix multiplication)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = tl.minimum(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
        b = tl.load(B_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc += bias[None, :]

    mask_m = offs_m < M
    mask_n = offs_n < N
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def triton_linear(x, weight, bias=None):
    """x: (..., K), weight: (N, K) -> (..., N)"""
    orig_shape = x.shape
    K = orig_shape[-1]
    N = weight.shape[0]
    x_2d = x.reshape(-1, K)
    M = x_2d.shape[0]
    y_2d = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # weight is (N, K), we need A(M,K) x B(K,N)
    # B = weight.T
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    gemm_kernel[grid](
        x_2d, weight.T.contiguous(), y_2d,
        M, N, K,
        x_2d.stride(0), x_2d.stride(1),
        weight.T.contiguous().stride(0), weight.T.contiguous().stride(1),
        y_2d.stride(0), y_2d.stride(1),
        HAS_BIAS=(bias is not None),
    )
    return y_2d.reshape(orig_shape[:-1] + (N,))


# ============================================================
# Conv2d kernel (implicit GEMM)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    TOTAL_K = IN_C * KH * KW

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K) &
            (offs_m[:, None] < M_TOTAL)
        )

        x_ptrs = X_ptr + (
            batch[:, None] * stride_xn +
            in_c[None, :] * stride_xc +
        )
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        w_ptrs = W_ptr + (
            offs_n[None, :] * stride_wco +
            in_c[:, None] * stride_wci +
            kh[:, None] * stride_wkh +
            kw[:, None] * stride_wkw
        )
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0).to(tl.float32)
        acc += bias[None, :]

    y_ptrs = Y_ptr + (
        batch[:, None] * stride_yn +
        offs_n[None, :] * stride_yc +
        out_h[:, None] * stride_yh +
        out_w[:, None] * stride_yw
    )
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_conv2d(x, weight, bias, stride=4, padding=0):
    N, C, H, W = x.shape
    Co, Ci, KH, KW = weight.shape
    SH = SW = stride
    PH = PW = padding
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    y = torch.empty((N, Co, OH, OW), device=x.device, dtype=x.dtype)

    M_TOTAL = N * OH * OW
    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(Co, meta['BLOCK_N']),
    )

    conv2d_kernel[grid](
        x, weight, bias if bias is not None else x, y,
        N, Ci, H, W,
        Co, OH, OW,
        KH, KW, SH, SW, PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        HAS_BIAS=(bias is not None),
    )
    return y


# ============================================================
# Sigmoid kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def sigmoid_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = 1.0 / (1.0 + tl.exp(-x))
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


def triton_sigmoid(x):
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, y, N)
    return y


# ============================================================
# ReLU kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def relu_kernel(X_ptr, Y_ptr, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offs, y.to(tl.float32), mask=mask)


def triton_relu(x):
    y = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, y, N)
    return y


# ============================================================
# Vector norm kernel (L2 norm along last dim)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 32}, num_warps=2),
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def vector_norm_kernel(X_ptr, Y_ptr, M, N, stride_m, BLOCK_N: tl.constexpr):
    row = tl.program_id(0)
    if row >= M:
        return
    x_base = X_ptr + row * stride_m
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x * x, axis=0)
    norm = libdevice.sqrt(acc)
    tl.store(Y_ptr + row, norm)


def triton_vector_norm(x, dim=-1, keepdim=True):
    """L2 norm along dim"""
    orig_shape = x.shape
    if dim < 0:
        dim = len(orig_shape) + dim
    # Move dim to last
    if dim != len(orig_shape) - 1:
        perm = list(range(len(orig_shape)))
        perm.append(perm.pop(dim))
        x = x.permute(*perm).contiguous()
        result = triton_vector_norm(x, dim=-1, keepdim=False)
        # Restore
        inv_perm = [0] * len(perm)
        for i, p in enumerate(perm):
            inv_perm[p] = i
        if keepdim:
            result = result.unsqueeze(-1)
            result = result.permute(*inv_perm)
        return result

    x_2d = x.reshape(-1, orig_shape[-1])
    M, N = x_2d.shape
    y = torch.empty(M, device=x.device, dtype=x.dtype)
    grid = (M,)
    vector_norm_kernel[grid](x_2d, y, M, N, x_2d.stride(0))
    if keepdim:
        return y.reshape(orig_shape[:-1] + (1,))
    return y.reshape(orig_shape[:-1])


# ============================================================
# Adaptive avg pool 1d kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['C', 'L'],
)
@triton.jit
def adaptive_avg_pool1d_kernel(X_ptr, Y_ptr, N, C, L, BLOCK_SIZE: tl.constexpr):
    # Output size is 1, so just average over L
    pid = tl.program_id(0)
    nc = pid  # index into N*C
    if nc >= N * C:
        return
    n = nc // C
    c = nc % C
    x_base = X_ptr + n * C * L + c * L
    acc = tl.zeros((), dtype=tl.float32)
    for off in range(0, L, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        mask = cols < L
        x = tl.load(x_base + cols, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(tl.where(mask, x, 0.0), axis=0)
    tl.store(Y_ptr + nc, (acc / L).to(tl.float32))


def triton_adaptive_avg_pool1d(x):
    """x: (N, C, L) -> (N, C, 1)"""
    N, C, L = x.shape
    y = torch.empty((N, C, 1), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    adaptive_avg_pool1d_kernel[grid](x, y, N, C, L)
    return y


# ============================================================
# Swin Transformer V2 forward pass
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
    relative_coords_table, relative_position_index,
    cpb_mlp_0_weight, cpb_mlp_0_bias,
    cpb_mlp_2_weight,
):
    # relative_coords_table: (1, 13, 13, 2)
    # relative_position_index: (49, 49)
    rct = relative_coords_table  # (1, 13, 13