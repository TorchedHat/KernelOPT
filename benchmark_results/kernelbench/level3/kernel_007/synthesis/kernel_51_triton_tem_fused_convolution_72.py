import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: bias add + ReLU (fused epilogue)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
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
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Triton kernel: bias add (no ReLU)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
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
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Triton kernel: max pool 2d (3x3, stride=2, padding=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def max_pool2d_s2_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh * 2 - 1 + kh
            iw = ow * 2 - 1 + kw
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# ============================================================
# Triton kernel: max pool 2d (3x3, stride=1, padding=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def max_pool2d_s1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh - 1 + kh
            iw = ow - 1 + kw
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# ============================================================
# Triton kernel: adaptive avg pool 2d (output 1x1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 64, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 32, 'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256, 'BLOCK_HW': 32}, num_warps=8),
    ],
    key=['NC', 'HW'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    inp_ptr, out_ptr,
    NC, HW,
    BLOCK_N: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    pid = tl.program_id(0)
    nc_offsets = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    nc_mask = nc_offsets < NC

    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    for hw_start in range(0, HW, BLOCK_HW):
        hw_offsets = hw_start + tl.arange(0, BLOCK_HW)
        hw_mask = hw_offsets < HW
        idx = nc_offsets[:, None] * HW + hw_offsets[None, :]
        full_mask = nc_mask[:, None] & hw_mask[None, :]
        val = tl.load(inp_ptr + idx, mask=full_mask, other=0.0)
        acc += tl.sum(val, axis=1)

    avg = acc / HW
    tl.store(out_ptr + nc_offsets, avg, mask=nc_mask)


# ============================================================
# Triton kernel: linear (GEMM) with bias
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc += tl.dot(x, w)

    offs_m_out = offs_m
    offs_n_out = offs_n
    b = tl.load(b_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)
    tl.store(out_ptr + offs_m_out[:, None] * N + offs_n_out[None, :], acc, mask=out_mask)


# ============================================================
# Implicit GEMM Conv2d kernel (general)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    nb = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        ih = oh[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_x = (
            (nb[:, None] < BATCH) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_idx = nb[:, None] * (IN_C * IN_H * IN_W) + in_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=valid_x, other=0.0)

        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_idx = offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) + in_c[:, None] * (KERNEL_H * KERNEL_W) + kh[:, None] * KERNEL_W + kw[:, None]
        w_tile = tl.load(W_ptr + w_idx, mask=valid_w, other=0.0)

        acc += tl.dot(x_tile, w_tile)

    if HAS_BIAS:
        b = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + b[None, :]

    if RELU:
        acc = tl.where(acc > 0.0, acc, 0.0)

    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    out_idx = nb[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + oh[:, None] * OUT_W + ow[:, None]
    tl.store(Y_ptr + out_idx, acc, mask=out_mask)


def triton_conv2d(x, weight, bias, stride, padding, relu=False):
    N, C, H, W = x.shape
    OC, IC, KH, KW = weight.shape
    SH, SW = stride if isinstance(stride, (list, tuple)) else (stride, stride)
    PH, PW = padding if isinstance(padding, (list, tuple)) else (padding, padding)
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    y = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    M_TOTAL = N * OH * OW
    HAS_BIAS = bias is not None

    if not HAS_BIAS:
        bias_ptr = x  # dummy, won't be used
    else:
        bias_ptr = bias

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    conv2d_kernel[grid](
        x, weight, bias_ptr, y,
        N, C, H, W,
        OC, OH, OW,
        SH, SW,
        PH, PW,
        KH, KW,
        HAS_BIAS, relu,
    )
    return y


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    OH = (H + 2 * padding - kernel_size) // stride + 1
    OW = (W + 2 * padding - kernel_size) // stride + 1
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)

    if stride == 2:
        grid = lambda meta: (triton.cdiv(N * C * OH * OW, meta['BLOCK_SIZE']),)
        max_pool2d_s2_kernel[grid](x, out, N, C, H, W, OH, OW)
    else:
        grid = lambda meta: (triton.cdiv(N * C * OH * OW, meta['BLOCK_SIZE']),)
        max_pool2d_s1_kernel[grid](x, out, N, C, H, W, OH, OW)
    return out


def triton_adaptive_avg_pool2d_1x1(x):
    N, C, H, W = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    NC = N * C
    HW = H * W
    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_N']),)
    adaptive_avg_pool2d_1x1_kernel[grid](x, out, NC, HW)
    return out


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
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
    # branch 1x1
    b1 = triton_conv2d(x, w_1x1, b_1x1, stride=1, padding=0, relu=True)

    # branch 3x3
    r3 = triton_conv2d(x, w_r3, b_r3, stride=1, padding=0, relu=True)
    b3 = triton_conv2d(r3, w_3x3, b_3x3, stride=1, padding=1, relu=True)

    # branch 5x5
    r5 = triton_conv2d(x, w_r5, b_r5, stride=1, padding=0, relu=True)
    b5 = triton_conv2d(r5, w_5x5, b_5x5, stride=1, padding=2, relu=True)

    # branch pool
    pool = triton_max_pool2d(x, kernel_size=3, stride=1, padding=1)
    bp = triton_conv2d(pool, w_pool, b_pool, stride=1, padding=0, relu=True)

    return torch.cat([b1, b3, b5, bp], dim=1)


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
    p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
    p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
    p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
    p_inception5b_branch_pool_1_weight, p_inception5b_branch_pool_1_bias,
    p_fc_weight, p_fc_bias,
    x,
):
    # conv1: 3->64, 7x7, stride=2, pad=3
    x = triton_conv2d(x, p_conv1_weight, p_conv1_bias, stride=2, padding=3, relu=True)
    # maxpool1: 3x3, stride=2, pad=1
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)
    # conv2: 64->64, 1x1
    x = triton_conv2d(x, p_conv2_weight, p_conv2_bias, stride=1, padding=0, relu=True)
    # conv3: 64->192, 3x3, pad=1
    x = triton_conv2d(x, p_conv3_weight, p_conv3_bias, stride=1, padding=1, relu=True)
    # maxpool2: 3x3, stride=2, pad=1
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)

    # inception3a: in=192, out_1x1=64, r3=96, out3=128, r5=16, out5=32, pool=32
    x = inception_module(x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,