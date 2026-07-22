import torch
import triton
import triton.language as tl
import torch.nn.functional as F

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    M, N_out, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    apply_relu: tl.constexpr,
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

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kh_idx = tmp_k // KW
        kw_idx = tmp_k % KW

        ih = out_h_idx[:, None] * SH + kh_idx[None, :] - PH
        iw = out_w_idx[:, None] * SW + kw_idx[None, :] - PW

        valid_x = (
            (batch_idx[:, None] < N) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < K)
        )

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * stride_xn +
                  in_c_idx[None, :] * stride_xc +
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = (offs_n[None, :] < OUT_C) & (offs_k[:, None] < K)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * stride_wco +
                  in_c_idx[:, None] * stride_wci +
                  kh_idx[:, None] * stride_wkh +
                  kw_idx[:, None] * stride_wkw)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    if apply_relu:
        acc = tl.maximum(acc, 0.0)

    mask_out = (
        (batch_idx[:, None] < N) &
        (out_h_idx[:, None] < OUT_H) &
        (out_w_idx[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )

    y_ptrs = (Y_ptr +
              batch_idx[:, None] * stride_yn +
              offs_n[None, :] * stride_yc +
              out_h_idx[:, None] * stride_yh +
              out_w_idx[:, None] * stride_yw)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), apply_relu=False):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1

    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_relu_kernel[grid](
        x, weight, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        M, OUT_C, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        apply_relu,
    )
    return y


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def relu_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


def triton_relu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def maxpool2d_kernel(
    x_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    KH, KW, SH, SW, PH, PW,
    stride_n, stride_c, stride_h, stride_w,
    out_stride_n, out_stride_c, out_stride_h, out_stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * OH * OW
    mask = offs < total

    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            ptrs = x_ptr + n * stride_n + c * stride_c + ih * stride_h + iw * stride_w
            val = tl.load(ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    out_ptrs = out_ptr + n * out_stride_n + c * out_stride_c + oh * out_stride_h + ow * out_stride_w
    tl.store(out_ptrs, max_val, mask=mask)


def triton_maxpool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, IH, IW = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    OH = (IH + 2*PH - KH) // SH + 1
    OW = (IW + 2*PW - KW) // SW + 1

    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, out,
        N, C, IH, IW, OH, OW,
        KH, KW, SH, SW, PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, IH, IW,
    numel,
    stride_n, stride_c, stride_h, stride_w,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    c = offs % C
    n = offs // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = IH * IW

    for h in range(IH):
        for w in range(IW):
            ptrs = x_ptr + n * stride_n + c * stride_c + h * stride_h + w * stride_w
            val = tl.load(ptrs, mask=mask, other=0.0)
            acc = acc + val

    acc = acc / count
    tl.store(out_ptr + offs, acc, mask=mask)


def triton_adaptive_avg_pool2d(x):
    N, C, IH, IW = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    numel = N * C
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool_kernel[grid](
        x, out,
        N, C, IH, IW,
        numel,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
    )
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc)

    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


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
                     w_r3x3, b_r3x3, w_3x3, b_3x3,
                     w_r5x5, b_r5x5, w_5x5, b_5x5,
    branch1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), apply_relu=True)

    r3 = triton_conv2d(x, w_r3x3, b_r3x3, stride=(1,1), padding=(0,0), apply_relu=True)
    branch2 = triton_conv2d(r3, w_3x3, b_3x3, stride=(1,1), padding=(1,1), apply_relu=True)

    r5 = triton_conv2d(x, w_r5x5, b_r5x5, stride=(1,1), padding=(0,0), apply_relu=True)
    branch3 = triton_conv2d(r5, w_5x5, b_5x5, stride=(1,1), padding=(2,2), apply_relu=True)

    pooled = triton_maxpool2d(x, kernel_size=3, stride=1, padding=1)
    branch4 = triton_conv2d(pooled, w_pool, b_pool, stride=(1,1), padding=(0,0), apply_relu=True)

    return torch.cat([branch1, branch2, branch3, branch4], dim=1)


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
    # conv1 + relu + maxpool
    x = triton_conv2d(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), apply_relu=True)
    x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=1)

    # conv2 + relu
    x = triton_conv2d(x, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), apply_relu=True)
    # conv3 + relu
    x = triton_conv2d(x, p_conv3_weight, p_conv3_bias, stride=(1,1), padding=(1,1), apply_relu=True)
    x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=1)

    # inception3a
    x = inception_module(x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
        p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    )

    # inception3b
    x = inception_module(x,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
        p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
        p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
        p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
        p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    )
    x = triton_maxpool2d(x, kernel_size=3, stride=2, padding=1)

    # inception4a
    x = inception_module(x,
        p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
        p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
        p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
        p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
        p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
        p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    )

    # inception4b
    x = inception_module(x,
        p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
        p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
        p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
        p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
        p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
        p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    )

    # inception4c
    x = inception_module(x,
        p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
        p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
        p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
        p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,