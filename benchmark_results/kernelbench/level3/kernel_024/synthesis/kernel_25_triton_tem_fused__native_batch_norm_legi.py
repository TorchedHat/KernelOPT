import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Fused BatchNorm + optional ReLU (pointwise, per-channel params)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=4),
        triton.Config({'BLOCK': 128},  num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    do_relu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # channel index
    c = (offs // HW) % C
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c, mask=mask, other=0.0).to(tl.float32)
    w    = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c, mask=mask, other=0.0).to(tl.float32)
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    if do_relu:
        out = tl.maximum(out, 0.0)
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5, relu=True):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * HW
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK']),)
    bn_relu_kernel[grid](
        x, out, running_mean, running_var, weight, bias,
        numel, C, HW, eps, relu,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: 1x1 Convolution as GEMM
# Input:  (N, Ci, H, W)  ->  reshape to (N*H*W, Ci)
# Weight: (Co, Ci)
# Output: (N, Co, H, W)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 16,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    X_ptr, W_ptr, Out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        mask_x = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        mask_w = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc = tl.dot(x, w, acc)
    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def conv1x1(x, weight):
    # x: (N, Ci, H, W), weight: (Co, Ci, 1, 1)
    N, Ci, H, W = x.shape
    Co = weight.shape[0]
    M = N * H * W
    K = Ci
    # Reshape x to (M, K) - need contiguous layout
    x_2d = x.permute(0, 2, 3, 1).contiguous().reshape(M, K)
    w_2d = weight.reshape(Co, Ci)  # (N=Co, K=Ci)
    out_2d = torch.empty((M, Co), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(Co, meta['BLOCK_N']))
    conv1x1_kernel[grid](
        x_2d, w_2d, out_2d,
        M, Co, K,
        x_2d.stride(0), x_2d.stride(1),
        w_2d.stride(0), w_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
    )
    # out_2d: (M, Co) -> (N, H, W, Co) -> (N, Co, H, W)
    return out_2d.reshape(N, H, W, Co).permute(0, 3, 1, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Standard Conv2d (3x3, stride, padding) - implicit GEMM
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M_total', 'OUT_C', 'IN_C', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    M_total, TOTAL_K,
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

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (
            (offs_m[:, None] < M_total) &
            (batch_idx[:, None] < BATCH) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )
        x_ptrs = X_ptr + batch_idx[:, None] * (IN_C * IN_H * IN_W) + in_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Weight: (OUT_C, IN_C, KH, KW) -> index [n, in_c, kh, kw]
        w_valid = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) + in_c[:, None] * (KERNEL_H * KERNEL_W) + kh[:, None] * KERNEL_W + kw[:, None]
        w_tile = tl.load(w_ptrs, mask=w_valid, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    out_mask = (offs_m[:, None] < M_total) & (offs_n[None, :] < OUT_C)
    out_ptrs = Out_ptr + batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h_idx[:, None] * OUT_W + out_w_idx[:, None]
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def conv2d_triton(x, weight, stride=1, padding=0):
    N, Ci, H, W = x.shape
    Co, _, KH, KW = weight.shape
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    SH, SW = stride
    PH, PW = padding
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1
    M_total = N * OH * OW
    TOTAL_K = Ci * KH * KW
    out = torch.empty((N, Co, OH, OW), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M_total, meta['BLOCK_M']), triton.cdiv(Co, meta['BLOCK_N']))
    conv2d_kernel[grid](
        x, weight, out,
        N, Ci, H, W,
        Co, OH, OW,
        KH, KW, SH, SW, PH, PW,
        M_total, TOTAL_K,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 4: Depthwise Conv2d (groups=C)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_C': 32}, num_warps=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 16}, num_warps=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 16}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_C': 16}, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_C': 16}, num_warps=2),
    ],
    key=['M_total', 'C', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    M_total,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    mask_m = offs_m < M_total
    mask_c = offs_c < C

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for kh in tl.static_range(KERNEL_H):
        for kw in tl.static_range(KERNEL_W):
            ih = out_h_idx[:, None] * STRIDE_H + kh - PAD_H
            iw = out_w_idx[:, None] * STRIDE_W + kw - PAD_W
            valid = mask_m[:, None] & mask_c[None, :] & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + batch_idx[:, None] * (C * IN_H * IN_W) + offs_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
            x_vals = tl.load(x_ptrs, mask=valid, other=0.0)
            # Weight: (C, 1, KH, KW)
            w_ptrs = W_ptr + offs_c[None, :] * (KERNEL_H * KERNEL_W) + kh * KERNEL_W + kw
            w_vals = tl.load(w_ptrs, mask=mask_c[None, :], other=0.0)
            acc += x_vals * w_vals

    out_mask = mask_m[:, None] & mask_c[None, :]
    out_ptrs = Out_ptr + batch_idx[:, None] * (C * OUT_H * OUT_W) + offs_c[None, :] * (OUT_H * OUT_W) + out_h_idx[:, None] * OUT_W + out_w_idx[:, None]
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def depthwise_conv2d_triton(x, weight, stride=1, padding=1):
    N, C, H, W = x.shape
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    SH, SW = stride
    PH, PW = padding
    KH, KW = weight.shape[2], weight.shape[3]
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1
    M_total = N * OH * OW
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M_total, meta['BLOCK_M']), triton.cdiv(C, meta['BLOCK_C']))
    depthwise_conv2d_kernel[grid](
        x, weight.reshape(C, KH * KW), out,
        N, C, H, W, OH, OW,
        KH, KW, SH, SW, PH, PW,
        M_total,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 5: Adaptive Average Pool 2D -> (1,1)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 128}, num_warps=2),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_C': 8,  'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_C': 4,  'BLOCK_HW': 1024},num_warps=8),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 512}, num_warps=8),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    X_ptr, Out_ptr,
    N, C, HW,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for hw_start in range(0, HW, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask_hw = offs_hw < HW
        x_ptrs = X_ptr + pid_n * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
        x_vals = tl.load(x_ptrs, mask=mask_c[:, None] & mask_hw[None, :], other=0.0)
        acc += tl.sum(x_vals, axis=1)

    out = acc / HW
    out_ptrs = Out_ptr + pid_n * C + offs_c
    tl.store(out_ptrs, out.to(tl.float32), mask=mask_c)


def adaptive_avg_pool_2d(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = lambda meta: (N, triton.cdiv(C, meta['BLOCK_C']))
    adaptive_avg_pool_kernel[grid](x, out.reshape(N, C), N, C, HW)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 6: Linear layer (GEMM: (M, K) x (K, N) + bias)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=2, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 4,  'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 2,  'BLOCK_N': 512, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 512, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = W_ptr + offs_n[None, :] * K + offs_k[:, None]
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)
        acc = tl.dot(x, w, acc)
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc += bias[None, :]
    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def linear_triton(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, weight, bias, out, M, N, K)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 7: Sigmoid pointwise
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=2),
        triton.Config({'BLOCK': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, numel, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)