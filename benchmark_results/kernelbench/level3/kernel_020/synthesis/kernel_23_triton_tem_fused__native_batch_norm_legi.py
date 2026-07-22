import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused BatchNorm + HardTanh (ReLU6) Triton kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['C', 'numel'],
)
@triton.jit
def bn_relu6_kernel(
    numel,       # total elements = N*C*H*W
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # For NCHW layout: channel index = (offset // (H*W)) % C
    # But we don't know H*W here. We pass spatial = numel // (N*C) but
    # it's easier to compute channel from offset given stride info.
    # We'll pass HW = numel / (N*C) as a separate param... 
    # Actually let's just use: channel = (offset // spatial) % C
    # We need spatial = HW. Let's add it as a param.
    # For now, restructure: see bn_relu6_kernel_v2 below.
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    # channel index: offset // spatial % C  -- but we need spatial
    # This kernel signature needs spatial. Let's use a different approach.
    # We'll compute c = (offset // spatial) % C
    # But BLOCK_SIZE is constexpr and spatial is runtime... 
    # Just pass spatial as a param.
    tl.store(x_ptr + offsets, x, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['C', 'spatial'],
)
@triton.jit
def fused_bn_hardtanh_kernel(
    x_ptr,
    mean_ptr,
    var_ptr,
    weight_ptr,
    bias_ptr,
    C: tl.constexpr,
    apply_relu6,  # 1 = apply hardtanh(0,6), 0 = no activation
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index for NCHW layout
    # offset = n * C * spatial + c * spatial + hw
    # c = (offset // spatial) % C
    c_idx = (offsets // spatial) % C

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std * w + b

    # HardTanh(0, 6) = ReLU6
    if apply_relu6:
        x_norm = tl.maximum(x_norm, 0.0)
        x_norm = tl.minimum(x_norm, 6.0)

    tl.store(x_ptr + offsets, x_norm, mask=mask)


# ============================================================
# Depthwise Conv2d Triton kernel (groups == in_channels)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 16}, num_warps=8),
    ],
    key=['C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def depthwise_conv2d_kernel(
    N, C, H, W,
    OUT_H, OUT_W,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    # Each program handles BLOCK_M spatial positions and BLOCK_C channels
    spatial_total = N * OUT_H * OUT_W
    m_start = pid_m * BLOCK_M
    c_start = pid_c * BLOCK_C

    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_c = c_start + tl.arange(0, BLOCK_C)

    mask_m = offs_m < spatial_total
    mask_c = offs_c < C

    # Decode spatial index
    n_idx = offs_m // (OUT_H * OUT_W)
    rem = offs_m % (OUT_H * OUT_W)
    oh_idx = rem // OUT_W
    ow_idx = rem % OUT_W

    # Accumulate over kernel
    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for kh in tl.static_range(KERNEL_H):
        for kw in tl.static_range(KERNEL_W):
            ih = oh_idx * STRIDE_H + kh - PAD_H
            iw = ow_idx * STRIDE_W + kw - PAD_W

            valid_h = (ih >= 0) & (ih < H)
            valid_w = (iw >= 0) & (iw < W)
            valid_spatial = mask_m & valid_h & valid_w

            # Load input: x[n, c, ih, iw]
            # x_ptr offset = n*C*H*W + c*H*W + ih*W + iw
            x_offset = (n_idx[:, None] * C * H * W +
                        offs_c[None, :] * H * W +
                        ih[:, None] * W +
                        iw[:, None])
            x_mask = valid_spatial[:, None] & mask_c[None, :]
            x_val = tl.load(x_ptr + x_offset, mask=x_mask, other=0.0)

            # Load weight: w[c, 0, kh, kw]
            # w_ptr offset = c * KH * KW + kh * KW + kw
            w_offset = offs_c * KERNEL_H * KERNEL_W + kh * KERNEL_W + kw
            w_val = tl.load(w_ptr + w_offset, mask=mask_c, other=0.0)

            acc += x_val * w_val[None, :]

    # Store output: out[n, c, oh, ow]
    out_offset = (n_idx[:, None] * C * OUT_H * OUT_W +
                  offs_c[None, :] * OUT_H * OUT_W +
                  oh_idx[:, None] * OUT_W +
                  ow_idx[:, None])
    out_mask = mask_m[:, None] & mask_c[None, :]
    tl.store(out_ptr + out_offset, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Pointwise Conv2d (1x1) via matrix multiply
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def pointwise_conv2d_kernel(
    x_ptr,    # (M, K) = (N*H*W, C_in)
    w_ptr,    # (K, N) = (C_in, C_out)  [transposed weight]
    out_ptr,  # (M, N) = (N*H*W, C_out)
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        mask_k = k_offs < K

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Standard Conv2d (3x3, stride 2) via implicit GEMM
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M_total', 'OUT_C', 'IN_C', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M_total
    mask_n = offs_n < OUT_C

    # Decode m -> (n, oh, ow)
    n_idx = offs_m // (OUT_H * OUT_W)
    rem = offs_m % (OUT_H * OUT_W)
    oh_idx = rem // OUT_W
    ow_idx = rem % OUT_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_base in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_base + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOTAL_K

        # Decode k -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kh = (offs_k // IN_C) % KERNEL_H
        kw = offs_k // (IN_C * KERNEL_H)

        ih = oh_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = ow_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (mask_m[:, None] &
                 (ih >= 0) & (ih < IN_H) &
                 (iw >= 0) & (iw < IN_W) &
                 mask_k[None, :])

        x_offset = (n_idx[:, None] * IN_C * IN_H * IN_W +
                    in_c[None, :] * IN_H * IN_W +
        x_tile = tl.load(x_ptr + x_offset, mask=valid, other=0.0)

        # Weight: w[out_c, in_c, kh, kw]
        # offset = out_c * IN_C*KH*KW + in_c * KH*KW + kh * KW + kw
        w_offset = (offs_n[None, :] * IN_C * KERNEL_H * KERNEL_W +
                    in_c[:, None] * KERNEL_H * KERNEL_W +
                    kh[:, None] * KERNEL_W +
                    kw[:, None])
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptr + w_offset, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    out_offset = (n_idx[:, None] * OUT_C * OUT_H * OUT_W +
                  offs_n[None, :] * OUT_H * OUT_W +
                  oh_idx[:, None] * OUT_W +
                  ow_idx[:, None])
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptr + out_offset, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Adaptive Average Pooling (global) + flatten
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=4),
        triton.Config({'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 256}, num_warps=8),
    ],
    key=['C', 'spatial'],
)
@triton.jit
def global_avg_pool_kernel(
    N, C, spatial,  # spatial = H*W
    BLOCK_C: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    mask_c = offs_c < C

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for s in range(spatial):
        x_offset = pid_n * C * spatial + offs_c * spatial + s
        x_val = tl.load(x_ptr + x_offset, mask=mask_c, other=0.0)
        acc += x_val

    acc = acc / spatial

    out_offset = pid_n * C + offs_c
    tl.store(out_ptr + out_offset, acc.to(tl.float32), mask=mask_c)


# ============================================================
# Linear layer (GEMM): out = x @ W.T + b
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        mask_k = k_offs < K

        x_ptrs = x_ptr + offs_m[:, None] * K + k_offs[None, :]
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # w is (N, K), we want w[offs_n, k_offs]
        w_ptrs = w_ptr + offs_n[None, :] * K + k_offs[:, None]
        # Actually: w_ptrs shape should be (BLOCK_K, BLOCK_N)
        w_ptrs2 = w_ptr + k_offs[:, None] * N + offs_n[None, :]
        # w[n, k] -> offset = n*K + k
        # We want (BLOCK_K, BLOCK_N): w[k_offs[i], offs_n[j]] = w_ptr + k_offs[i]*... 
        # w is (N, K): w[n, k] = w_ptr + n*K + k
        # We want tile (BLOCK_K, BLOCK_N): entry [i,j] = w[offs_n[j], k_offs[i]]
        w_ptrs3 = w_ptr + offs_n[None, :] * K + k_offs[:, None]
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs3, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    b_vals = tl.load(b_ptr + offs_n, mask=mask_n, other=0.0)
    acc += b_vals[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Helper functions
# ============================================================

def apply_bn_relu6(x, mean, var, weight, bias, apply_relu6=True, eps=1e-5):
    """Apply batch norm + optional ReLU6 in-place using Triton."""
    N, C, H, W = x.shape
    numel = x.numel()
    spatial = H * W

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    fused_bn_hardtanh_kernel[grid](
        x, mean, var, weight, bias,
        C, spatial, numel,
        1 if apply_relu6 else 0,
        eps,
    )
    return x


def do_pointwise_conv2d(x, weight):
    """1x1 convolution: x is (N, C_in, H, W), weight is (C_out, C_in, 1, 1)."""
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    M = N * H * W

    # Reshape x to (M, C_in) and weight to (C_in, C_out)
    x_2d = x.permute(0, 2, 3, 1).contiguous().view(M, C_in)
    w_2d = weight.view(C_out, C_in)  # (C_out, C_in)

    out_2d = torch.empty((M, C_out), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )
    pointwise_conv2d_kernel[grid](
        x_2d, w_2d, out_2d,
        M, C_out, C_in,
        x_2d.stride(0), x_2d.stride(1),
        1, 1,  # w_2d is (C_out, C_in), we want x @ w.T so stride_wk=1, stride_wn=C_in... 
        # Actually w_2d[k, n] means we need w transposed
        # Let's just transpose w
        out_2d.stride(0), out_2d.stride(1),
    )
    # This is wrong - let me fix the weight layout
    return out_2d.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def do_pointwise_conv2d_v2(x, weight):
    """1x1 convolution: x is (N, C_in, H, W), weight is (C_out, C_in, 1, 1)."""
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    M = N * H * W

    # x_2d: (M, C_in) in row-major
    x_2d = x.permute(0, 2, 3, 1).reshape(M, C_in)
    # w_2d: (C_in, C_out) for x @ w
    w_2d = weight.view(C_out, C_in).t().contiguous()  # (C_in, C_out)

    out_2d = torch.empty((M, C_out), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )
    pointwise_conv2d_kernel[grid](
        x_2d, w_2d, out_2d,
        M, C_out, C_in,
        x_2d.stride(0), x_2d.stride(1),
        w_2d.stride(0), w_2d.stride(1),
        out_2d.stride(0), out_2d.stride(1),
    )
    return out_2d.view(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def do_depthwise_conv2d(x, weight, stride=1, padding=1):
    """Depthwise conv2d."""
    N, C, H, W = x.shape
    KH, KW = weight.shape[2], weight.shape[3]
    OH = (H + 2 * padding - KH) // stride + 1
    OW = (W + 2 * padding - KW) // stride + 1