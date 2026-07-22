import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: fused bias + ReLU (pointwise epilogue)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def fused_bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Triton kernel: fused bias add (no ReLU)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def fused_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Triton kernel: max pool 2D (3x3, stride=2, padding=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'OH', 'OW', 'IH', 'IW'],
)
@triton.jit
def max_pool2d_3x3_s2_p1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh * 2 + kh - 1
            iw = ow * 2 + kw - 1
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid & (val > max_val), val, max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# ============================================================
# Triton kernel: max pool 2D (3x3, stride=1, padding=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'OH', 'OW', 'IH', 'IW'],
)
@triton.jit
def max_pool2d_3x3_s1_p1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh + kh - 1
            iw = ow + kw - 1
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid & (val > max_val), val, max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# ============================================================
# Triton kernel: adaptive avg pool 2D (output 1x1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'IH', 'IW'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    c = offsets % C
    n = offsets // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    hw = IH * IW
    for h in range(IH):
        for w in range(IW):
            idx = n * (C * IH * IW) + c * (IH * IW) + h * IW + w
            val = tl.load(inp_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    avg = acc / (hw)
    tl.store(out_ptr + offsets, avg, mask=mask)


# ============================================================
# Triton GEMM kernel for linear layer
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)

    offs_m2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (offs_m2[:, None] < M) & (offs_n2[None, :] < N)

    b = tl.load(b_ptr + offs_n2, mask=offs_n2 < N, other=0.0)
    acc = acc + b[None, :]

    out_ptrs = out_ptr + offs_m2[:, None] * N + offs_n2[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Triton implicit GEMM convolution kernel
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
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KH', 'KW'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    has_bias: tl.constexpr,
    apply_relu: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    n = tmp // OUT_H

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        ic = k_offs % IN_C
        kw_idx = (k_offs // IN_C) % KW
        kh_idx = k_offs // (IN_C * KW)

        ih = oh[:, None] * stride_h + kh_idx[None, :] - pad_h
        iw = ow[:, None] * stride_w + kw_idx[None, :] - pad_w

        valid_x = (
            (offs_m[:, None] < M) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (k_offs[None, :] < TOTAL_K)
        )

        x_idx = n[:, None] * (IN_C * IN_H * IN_W) + ic[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(x_ptr + x_idx, mask=valid_x, other=0.0)

        # weight: [OUT_C, IN_C, KH, KW] -> index [oc, ic, kh, kw]
        # w_idx for each (k, oc): oc * IN_C*KH*KW + ic * KH*KW + kh * KW + kw
        w_idx = offs_n[None, :] * TOTAL_K + ic[:, None] * (KH * KW) + kh_idx[:, None] * KW + kw_idx[:, None]
        valid_w = (k_offs[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptr + w_idx, mask=valid_w, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    if has_bias:
        b = tl.load(bias_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + b[None, :]

    if apply_relu:
        acc = tl.where(acc > 0.0, acc, 0.0)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    # output layout: [BATCH, OUT_C, OUT_H, OUT_W] -> n * OUT_C*OUT_H*OUT_W + oc * OUT_H*OUT_W + oh * OUT_W + ow
    out_idx = n[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + oh[:, None] * OUT_W + ow[:, None]
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), apply_relu=False):
    """Triton-based conv2d with optional bias and ReLU."""
    N, C, H, W = x.shape
    OC, IC, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OH = (H + 2*PH - KH) // SH + 1
    OW = (W + 2*PW - KW) // SW + 1

    out = torch.empty(N, OC, OH, OW, device=x.device, dtype=x.dtype)

    M = N * OH * OW
    has_bias = bias is not None

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    # Flatten weight to [OC, IC*KH*KW]
    w_flat = weight.reshape(OC, -1)

    if not has_bias:
        bias_dummy = torch.zeros(OC, device=x.device, dtype=x.dtype)
    else:
        bias_dummy = bias

    conv2d_kernel[grid](
        x, w_flat, bias_dummy, out,
        N, C, H, W,
        OC, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        has_bias,
        apply_relu,
        BATCH=N, IN_C=C, IN_H=H, IN_W=W, OUT_C=OC, OUT_H=OH, OUT_W=OW,
    )
    return out


def triton_max_pool2d_s2(x):
    """Max pool 3x3, stride=2, padding=1."""
    N, C, H, W = x.shape
    OH = (H + 2*1 - 3) // 2 + 1
    OW = (W + 2*1 - 3) // 2 + 1
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    max_pool2d_3x3_s2_p1_kernel[grid](
        x, out,
        N, C, H, W, OH, OW,
    )
    return out


def triton_max_pool2d_s1(x):
    """Max pool 3x3, stride=1, padding=1."""
    N, C, H, W = x.shape
    OH = H
    OW = W
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    max_pool2d_3x3_s1_p1_kernel[grid](
        x, out,
        N, C, H, W, OH, OW,
    )
    return out


def triton_adaptive_avg_pool2d(x):
    """Adaptive avg pool to 1x1."""
    N, C, H, W = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    total = N * C
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x, out,
        N, C, H, W,
    )
    return out


def triton_linear(x, weight, bias):
    """Linear layer: x @ weight.T + bias."""
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


def inception_module(x,
                     w_1x1, b_1x1,
                     w_r3, b_r3, w_3x3, b_3x3,
                     w_r5, b_r5, w_5x5, b_5x5,
    """Inception module with 4 branches."""
    # Branch 1: 1x1 conv + relu
    b1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), apply_relu=True)

    # Branch 2: 1x1 reduce + 3x3 conv + relu
    b2 = triton_conv2d(x, w_r3, b_r3, stride=(1,1), padding=(0,0), apply_relu=True)
    b2 = triton_conv2d(b2, w_3x3, b_3x3, stride=(1,1), padding=(1,1), apply_relu=True)

    # Branch 3: 1x1 reduce + 5x5 conv + relu
    b3 = triton_conv2d(x, w_r5, b_r5, stride=(1,1), padding=(0,0), apply_relu=True)
    b3 = triton_conv2d(b3, w_5x5, b_5x5, stride=(1,1), padding=(2,2), apply_relu=True)

    # Branch 4: max pool + 1x1 conv + relu
    b4 = triton_max_pool2d_s1(x)
    b4 = triton_conv2d(b4, w_pool, b_pool, stride=(1,1), padding=(0,0), apply_relu=True)

    return torch.cat([b1, b2, b3, b4], dim=1)


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_conv3_weight, p_conv3_bias,
    p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
    p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
    p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
    p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
    p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
    p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
    p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
    p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
    p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
    p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
    p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
    p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
    p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
    p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
    p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
    p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
    p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
    p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
    p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
    p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
    p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
    p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
    p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
    p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
    p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
    p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias,
    p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
    p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
    p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
    p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
    p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
    p_inception4d_branch_pool_1_weight, p_inception4d_branch_pool_1_bias,
    p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
    p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
    p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
    p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
    p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
    p_inception4e_branch_pool_1_weight, p_inception4e_branch_pool_1_bias,
    p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
    p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
    p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
    p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
    p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
    p_inception5a_branch_pool_1_weight, p_inception5a_branch_pool_1_bias,
    p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
    p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,