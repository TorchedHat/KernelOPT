import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ============================================================
# Fused BatchNorm + ReLU pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C,
    eps,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    # channel index: layout is (N, C, H, W) so channel = (offset // HW) % C
    # But we pass C and numel=N*C*H*W, HW = numel // (N*C) -- we need channel idx
    # We'll compute channel from offset: channel = (offset // (numel // (N*C))) % C
    # Actually simpler: we store channel stride info
    # For NCHW: element i has channel = (i // hw) % C where hw = numel/N/C
    # We pass spatial_size = H*W*N... let's just use C directly
    # channel index = (offset // spatial) % C, but we don't have spatial here
    # Instead, pass the stride: channel_stride = H*W
    # channel = (offset // channel_stride) % C
    # We'll pass channel_stride as a parameter
    # Actually let's restructure - pass channel_stride
    # For now use: channel = (offset // 1) % C -- NO
    # We need to pass HW separately. Let me use a different approach:
    # offset layout: offset = n * C * HW + c * HW + hw
    # so c = (offset // HW) % C
    # We'll pass HW as parameter
    # This kernel signature needs HW. Let me add it.
    pass

# Rewrite with HW parameter
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_relu_kernel_v2(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, HW, C,
    eps: tl.constexpr,
    apply_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

    c = (offsets // HW) % C
    c_mask = mask

    mean = tl.load(mean_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c, mask=c_mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c, mask=c_mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c, mask=c_mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b

    if apply_relu:
        out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Fused Sigmoid pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def sigmoid_kernel(
    x_ptr, out_ptr, numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# ReLU pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def relu_kernel(
    x_ptr, out_ptr, numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Linear (GEMM) kernel using tl.dot
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc)

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
        acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Implicit GEMM Conv2d kernel (for 1x1 convolutions)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """1x1 convolution as GEMM: x(M, K) @ w(K, N) -> out(M, N)
    where M = N*H*W, K = C_in, N = C_out
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Depthwise Conv2d kernel (groups = C_in = C_out)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 16}, num_warps=8),
    ],
    key=['M', 'C'],
)
@triton.jit
def depthwise_conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    # Decode m -> (n, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    n = offs_m // (OUT_H * OUT_W)

    m_mask = offs_m < M
    c_mask = offs_c < C

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W

            valid = m_mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)

            # x[n, c, ih, iw] -> x_ptr + n*C*IN_H*IN_W + c*IN_H*IN_W + ih*IN_W + iw
            x_ptrs = (x_ptr
            x_mask = valid[:, None] & c_mask[None, :]
            x_val = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            # w[c, 1, kh, kw] -> w_ptr + c*KH*KW + kh*KW + kw
            w_ptrs = w_ptr + offs_c * (KH * KW) + kh * KW + kw
            w_val = tl.load(w_ptrs, mask=c_mask, other=0.0).to(tl.float32)

            acc += x_val * w_val[None, :]

    # Store out[n, c, oh, ow]
    out_ptrs = (out_ptr
    out_mask = m_mask[:, None] & c_mask[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Adaptive Average Pool 2d -> (1,1) kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 128}, num_warps=4),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    """Adaptive avg pool to (1,1): output shape (N, C, 1, 1)"""
    pid = tl.program_id(0)  # over N*C
    nc = pid
    n = nc // C
    c = nc % C

    if n >= N:
        return

    acc = tl.zeros((), dtype=tl.float32)
    count = 0

    for hw_start in range(0, HW, BLOCK_HW):
        offs_hw = hw_start + tl.arange(0, BLOCK_HW)
        mask = offs_hw < HW
        x_ptrs = x_ptr + n * C * HW + c * HW + offs_hw
        x_val = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(x_val, axis=0)

    out = acc / HW
    tl.store(out_ptr + n * C + c, out)


# ============================================================
# Helper functions
# ============================================================
def apply_bn_relu(x, mean, var, weight, bias, eps=1e-5, apply_relu=True):
    """Apply batch norm + optional relu using Triton kernel"""
    out = torch.empty_like(x)
    numel = x.numel()
    N, C, H, W = x.shape
    HW = H * W

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel_v2[grid](
        x, mean, var, weight, bias,
        out,
        numel, HW, C,
        eps,
        apply_relu,
    )
    return out


def apply_sigmoid(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


def apply_relu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def do_conv1x1(x, w):
    """1x1 conv: x(N,C_in,H,W), w(C_out,C_in,1,1) -> out(N,C_out,H,W)"""
    N, C_in, H, W = x.shape
    C_out = w.shape[0]
    M = N * H * W
    K = C_in

    # Reshape x to (M, K) and w to (K, N_out)
    x_2d = x.permute(0, 2, 3, 1).contiguous().view(M, K)  # (M, K)
    w_2d = w.view(C_out, K)  # (C_out, K)

    out_2d = torch.empty((M, C_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )
    conv1x1_kernel[grid](
        x_2d, w_2d, out_2d,
        M, C_out, K,
        x_2d.stride(0), x_2d.stride(1),
        w_2d.stride(0), w_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
    )

    return out_2d.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def do_depthwise_conv2d(x, w, stride=1, padding=1):
    """Depthwise conv2d: groups = C_in = C_out"""
    N, C, IN_H, IN_W = x.shape
    KH, KW = w.shape[2], w.shape[3]
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1
    M = N * OUT_H * OUT_W

    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C, meta['BLOCK_C']),
    )
    depthwise_conv2d_kernel[grid](
        x, w.view(C, KH * KW).contiguous().view(C, 1, KH, KW).view(C, KH, KW).contiguous(),
        out,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW,
        stride, stride,
        padding, padding,
        M,
    )
    return out


def do_adaptive_avg_pool2d_1x1(x):
    """Adaptive avg pool to (1,1)"""
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)

    # Use a flat output buffer
    out_flat = out.view(N * C)

    grid = (N * C,)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x.contiguous(), out_flat,
        N, C, HW,
    )
    return out


def do_linear(x, w, bias):
    """Linear: x(M, K) @ w(N, K).T + bias(N) -> out(M, N)"""
    M, K = x.shape
    N = w.shape[0]

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_kernel[grid](
        x, w, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
    )
    return out


# ============================================================
# Standard 3x3 conv using implicit GEMM
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K', 'IN_H', 'IN_W'],
)
@triton.jit
def conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    N_batch, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """3x3 conv as implicit GEMM
    M = N_batch * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * 3 * 3
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode m -> (batch, out_h, out_w)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    nb = offs_m // (OUT_H * OUT_W)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * 9  # 3x3 kernel

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode k -> (in_c, kh, kw)
        in_c = offs_k // 9
        kh = (offs_k % 9) // 3
        kw = offs_k % 3

        ih = oh[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m < M
        valid_k = offs_k < TOTAL_K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        x_mask = valid_m[:, None] & valid_k[None, :] & valid_spatial

        x_ptrs = (x_ptr
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # w[out_c, in_c, kh, kw] -> w_ptr + out_c * IN_C*9 + in_c*9 + kh*3 + kw
        w_ptrs = (w_ptr
        w_mask = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Store output: out[nb, out_c, oh, ow]
    out_ptrs = (out_ptr
    out_mask = (offs_m[:, None] < M) & (offs_n[None