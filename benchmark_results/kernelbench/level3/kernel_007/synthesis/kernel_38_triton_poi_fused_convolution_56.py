import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused bias add + ReLU pointwise kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + bias
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(
    x_ptr, out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# Max pool 2D kernel: kernel=3, stride=2, padding=1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW', 'IH', 'IW'],
)
@triton.jit
def maxpool2d_k3s2p1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
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

    ih_base = oh * 2 - 1
    iw_base = ow * 2 - 1

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for ki in range(3):
        for kj in range(3):
            ih = ih_base + ki
            iw = iw_base + kj
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * stride_n + c * stride_c + ih * stride_h + iw * stride_w
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid & (val > max_val), val, max_val)

    out_idx = n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow * out_stride_w
    tl.store(out_ptr + out_idx, max_val, mask=mask)


# Max pool 2D kernel: kernel=3, stride=1, padding=1 (for inception branch_pool)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW', 'IH', 'IW'],
)
@triton.jit
def maxpool2d_k3s1p1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
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

    ih_base = oh - 1
    iw_base = ow - 1

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for ki in range(3):
        for kj in range(3):
            ih = ih_base + ki
            iw = iw_base + kj
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * stride_n + c * stride_c + ih * stride_h + iw * stride_w
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid & (val > max_val), val, max_val)

    out_idx = n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow * out_stride_w
    tl.store(out_ptr + out_idx, max_val, mask=mask)


# Adaptive avg pool 2D: output 1x1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'IH', 'IW'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    c = offsets % C
    n = offsets // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = IH * IW

    for h in range(IH):
        for w in range(IW):
            idx = n * stride_n + c * stride_c + h * stride_h + w * stride_w
            val = tl.load(inp_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    avg = acc / count
    tl.store(out_ptr + offsets, avg, mask=mask)


# Linear kernel (matrix-vector / matrix-matrix using tl.dot)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4),
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

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x, w, acc)

    offs_m2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b = tl.load(b_ptr + offs_n2, mask=offs_n2 < N, other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m2[:, None] < M) & (offs_n2[None, :] < N)
    tl.store(out_ptr + offs_m2[:, None] * N + offs_n2[None, :], acc, mask=out_mask)


# Implicit GEMM convolution kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # Input strides (NCHW)
    s_xn, s_xc, s_xh, s_xw,
    # Weight strides (NCHW: out_c, in_c, kh, kw)
    s_woc, s_wic, s_wkh, s_wkw,
    DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode m -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode k -> (in_c, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c = tmp_k // KERNEL_H

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        x_mask = valid_m & valid_k & valid_spatial

        x_ptrs = X_ptr + batch[:, None] * s_xn + in_c[None, :] * s_xc + ih * s_xh + iw * s_xw
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Weight: [BLOCK_K, BLOCK_N] -> w[out_c, in_c, kh, kw]
        w_ptrs = W_ptr + offs_n[None, :] * s_woc + in_c[:, None] * s_wic + kh[:, None] * s_wkh + kw[:, None] * s_wkw
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + b[None, :]

    if DO_RELU:
        acc = tl.where(acc > 0.0, acc, 0.0)

    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    # Store in NCHW: Y[batch, out_c, out_h, out_w]
    y_ptrs = Y_ptr + batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(y_ptrs, acc, mask=out_mask)


def triton_conv2d(x, w, b, stride=(1,1), padding=(0,0), do_relu=False):
    """Run conv2d using Triton implicit GEMM kernel."""
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = w.shape
    STRIDE_H, STRIDE_W = stride
    PAD_H, PAD_W = padding

    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    y = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Make contiguous
    x = x.contiguous()
    w = w.contiguous()

    s_xn, s_xc, s_xh, s_xw = x.stride()
    s_woc, s_wic, s_wkh, s_wkw = w.stride()

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_relu_kernel[grid](
        x, w, b, y,
        BATCH, IN_C, IN_H, IN_W, OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xc, s_xh, s_xw,
        s_woc, s_wic, s_wkh, s_wkw,
        do_relu,
    )
    return y


def triton_maxpool2d_s2(x):
    """Max pool 2D with kernel=3, stride=2, padding=1."""
    BATCH, C, IH, IW = x.shape
    OH = (IH + 2 * 1 - 3) // 2 + 1
    OW = (IW + 2 * 1 - 3) // 2 + 1

    x = x.contiguous()
    out = torch.empty((BATCH, C, OH, OW), device=x.device, dtype=x.dtype)

    s_n, s_c, s_h, s_w = x.stride()
    os_n, os_c, os_h, os_w = out.stride()

    total = BATCH * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    maxpool2d_k3s2p1_kernel[grid](
        x, out,
        BATCH, C, IH, IW, OH, OW,
        s_n, s_c, s_h, s_w,
        os_n, os_c, os_h, os_w,
    )
    return out


def triton_maxpool2d_s1(x):
    """Max pool 2D with kernel=3, stride=1, padding=1."""
    BATCH, C, IH, IW = x.shape
    OH = IH
    OW = IW

    x = x.contiguous()
    out = torch.empty((BATCH, C, OH, OW), device=x.device, dtype=x.dtype)

    s_n, s_c, s_h, s_w = x.stride()
    os_n, os_c, os_h, os_w = out.stride()

    total = BATCH * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    maxpool2d_k3s1p1_kernel[grid](
        x, out,
        BATCH, C, IH, IW, OH, OW,
        s_n, s_c, s_h, s_w,
        os_n, os_c, os_h, os_w,
    )
    return out


def triton_adaptive_avg_pool2d_1x1(x):
    """Adaptive avg pool 2D to 1x1."""
    BATCH, C, IH, IW = x.shape
    x = x.contiguous()
    out = torch.empty((BATCH, C, 1, 1), device=x.device, dtype=x.dtype)

    s_n, s_c, s_h, s_w = x.stride()

    total = BATCH * C
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    adaptive_avg_pool2d_1x1_kernel[grid](
        x, out,
        BATCH, C, IH, IW,
        s_n, s_c, s_h, s_w,
    )
    return out


def triton_linear(x, w, b):
    """Linear layer: x @ w.T + b"""
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2

    x = x.contiguous()
    w = w.contiguous()

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_kernel[grid](
        x, w, b, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
    )
    return out


def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    """Inception module forward pass."""
    # Branch 1x1
    branch1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), do_relu=True)

    # Branch 3x3
    branch3_r = triton_conv2d(x, w_3x3_r, b_3x3_r, stride=(1,1), padding=(0,0), do_relu=True)
    branch3 = triton_conv2d(branch3_r, w_3x3, b_3x3, stride=(1,1), padding=(1,1), do_relu=True)

    # Branch 5x5
    branch5_r = triton_conv2d(x, w_5x5_r, b_5x5_r, stride=(1,1), padding=(0,0), do_relu=True)
    branch5 = triton_conv2d(branch5_r, w_5x5, b_5x5, stride=(1,1), padding=(2,2), do_relu=True)

    # Branch pool
    pooled = triton_maxpool2d_s1(x)
    branch_pool = triton_conv2d(pooled, w_pool, b_pool, stride=(1,1), padding=(0,0), do_relu=True)

    return torch.cat([branch1, branch3, branch5, branch_pool], dim=1)


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