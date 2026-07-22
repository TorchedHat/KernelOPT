import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: bias + relu (pointwise epilogue after conv)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def bias_relu_kernel(x_ptr, bias_ptr, out_ptr, numel, C, apply_relu: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c_idx = offs % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x = x + b
    if apply_relu:
        x = tl.maximum(x, tl.zeros_like(x))
    tl.store(out_ptr + offs, x, mask=mask)


# ============================================================
# Triton kernel: max pool 2d (3x3, stride=2, padding=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def maxpool2d_kernel(
    inp_ptr, out_ptr,
    N, C, H, W, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    numel,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # decode: offs -> (n, c, oh, ow)
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            idx = n * (C * H * W) + c * (H * W) + ih * W + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offs, max_val, mask=mask)


# ============================================================
# Triton kernel: max pool 2d (3x3, stride=1, padding=1) - for inception branch_pool
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def maxpool2d_s1_kernel(
    inp_ptr, out_ptr,
    N, C, H, W,
    numel,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    ow = offs % W
    tmp = offs // W
    oh = tmp % H
    tmp2 = tmp // H
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            ih = oh + kh - 1
            iw = ow + kw - 1
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            idx = n * (C * H * W) + c * (H * W) + ih * W + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offs, max_val, mask=mask)


# ============================================================
# Triton kernel: adaptive avg pool 2d (output 1x1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=2),
    ],
    key=['NC'],
)
@triton.jit
def adaptive_avg_pool_1x1_kernel(
    inp_ptr, out_ptr,
    NC, HW,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < NC

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(HW):
        val = tl.load(inp_ptr + offs * HW + i, mask=mask, other=0.0)
        acc = acc + val
    acc = acc / HW

    tl.store(out_ptr + offs, acc, mask=mask)


# ============================================================
# Triton kernel: linear (GEMM) with bias
# M x K @ K x N + N -> M x N
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :], mask=w_mask, other=0.0)
        acc = tl.dot(x, w, acc)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ============================================================
# Implicit GEMM Conv2d kernel (general)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    N, C_in, H_in, W_in,
    C_out, H_out, W_out,
    KH, KW,
    SH, SW,
    PH, PW,
    M, N_out, K,
    apply_relu: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # decode m -> (batch, out_h, out_w)
    ow = offs_m % W_out
    tmp = offs_m // W_out
    oh = tmp % H_out
    bn = tmp // H_out

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # decode k -> (c_in, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        c_in_idx = tmp_k // KH

        ih = oh[:, None] * SH + kh_idx[None, :] - PH
        iw = ow[:, None] * SW + kw_idx[None, :] - PW

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < H_in)
        valid_iw = (iw >= 0) & (iw < W_in)
        valid_bn = bn[:, None] < N

        x_mask = valid_m & valid_k & valid_ih & valid_iw & valid_bn
        x_idx = bn[:, None] * (C_in * H_in * W_in) + c_in_idx[None, :] * (H_in * W_in) + ih * W_in + iw
        x = tl.load(X_ptr + x_idx, mask=x_mask, other=0.0)

        # weight: [C_out, C_in, KH, KW] -> index [c_out, c_in, kh, kw]
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N_out)
        w_idx = offs_n[None, :] * K + offs_k[:, None]
        w = tl.load(W_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc)

    # add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N_out, other=0.0)
    acc = acc + b[None, :]

    if apply_relu:
        acc = tl.maximum(acc, tl.zeros_like(acc))

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_out)
    # output layout: [N, C_out, H_out, W_out]
    out_idx = bn[:, None] * (C_out * H_out * W_out) + offs_n[None, :] * (H_out * W_out) + oh[:, None] * W_out + ow[:, None]
    tl.store(Out_ptr + out_idx, acc, mask=out_mask)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), apply_relu=False):
    """
    """
    N, C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    H_out = (H_in + 2 * PH - KH) // SH + 1
    W_out = (W_in + 2 * PW - KW) // SW + 1

    M = N * H_out * W_out
    N_out = C_out
    K = C_in * KH * KW

    # Reshape weight to [C_out, K] for easier indexing
    w_flat = weight.reshape(C_out, K)

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_out, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x, w_flat, bias, out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW,
        SH, SW,
        PH, PW,
        M, N_out, K,
        apply_relu,
    )
    return out


def triton_maxpool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    OH = (H + 2 * padding - kernel_size) // stride + 1
    OW = (W + 2 * padding - kernel_size) // stride + 1
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    numel = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, out,
        N, C, H, W, OH, OW,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
        numel,
    )
    return out


def triton_maxpool2d_s1(x):
    """3x3 maxpool with stride=1, padding=1 (same spatial size)"""
    N, C, H, W = x.shape
    out = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)
    numel = N * C * H * W
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    maxpool2d_s1_kernel[grid](
        x, out,
        N, C, H, W,
        numel,
    )
    return out


def triton_adaptive_avg_pool_1x1(x):
    N, C, H, W = x.shape
    NC = N * C
    HW = H * W
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_SIZE']),)
    adaptive_avg_pool_1x1_kernel[grid](
        x, out,
        NC, HW,
    )
    return out


def triton_linear(x, weight, bias):
    """x: [M, K], weight: [N, K], bias: [N] -> [M, N]"""
    M, K = x.shape
    N = weight.shape[0]
    # weight is [N, K], we need [K, N] for matmul
    w_t = weight.t().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_kernel[grid](
        x, w_t, bias, out,
        M, N, K,
    )
    return out


def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    # branch1x1
    branch1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), apply_relu=True)

    # branch3x3
    branch3_r = triton_conv2d(x, w_3x3_r, b_3x3_r, stride=(1,1), padding=(0,0), apply_relu=True)
    branch3 = triton_conv2d(branch3_r, w_3x3, b_3x3, stride=(1,1), padding=(1,1), apply_relu=True)

    # branch5x5
    branch5_r = triton_conv2d(x, w_5x5_r, b_5x5_r, stride=(1,1), padding=(0,0), apply_relu=True)
    branch5 = triton_conv2d(branch5_r, w_5x5, b_5x5, stride=(1,1), padding=(2,2), apply_relu=True)

    # branch_pool: maxpool(3x3,s=1,p=1) then conv1x1
    pooled = triton_maxpool2d_s1(x)
    branch_pool = triton_conv2d(pooled, w_pool, b_pool, stride=(1,1), padding=(0,0), apply_relu=True)

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
):
    # conv1: 7x7, stride=2, padding=3
    x = triton_conv2d(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), apply_relu=True)
    # maxpool1: 3x3, stride=2, padding=1
    x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=1)

    # conv2: 1x1
    x = triton_conv2d(x, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), apply_relu=True)
    # conv3: 3x3, padding=1
    x = triton_conv2d(x, p_conv3_weight, p_conv3_bias, stride=(1,1), padding=(1,1), apply_relu=True)
    # maxpool2: 3x3, stride=2, padding=1
    x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=1)

    # inception3a
    x = inception_module(x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,

    # inception3b
    x = inception_module(x,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
        p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
        p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
        p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,

    # maxpool3: 3x3, stride=2, padding=1
    x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=1)

    # inception4a
    x = inception_module(x,
        p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
        p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,