import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
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
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        ic_idx = tmp_k // KH

        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            (batch_idx[:, None] < N) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (ic_idx[None, :] < IN_C) &
            (offs_k[None, :] < K)
        )

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * stride_xn +
                  ic_idx[None, :] * stride_xc +
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = (
            (offs_n[:, None] < OUT_C) &
            (offs_k[None, :] < K)
        )
        w_ptrs = (W_ptr +
                  offs_n[:, None] * stride_wco +
                  ic_idx[None, :] * stride_wci +
                  kh_idx[None, :] * stride_wkh +
                  kw_idx[None, :] * stride_wkw)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU
    if apply_relu:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store
    valid_out = (
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
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_out)


def triton_conv2d_relu(x, weight, bias, stride=(1,1), padding=(0,0), apply_relu=True):
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

    conv2d_bias_relu_kernel[grid](
        x, weight, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
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
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def max_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW,
    SH, SW,
    PH, PW,
    ceil_mode: tl.constexpr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # Decode: offs -> (n, c, oh, ow)
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    y_ptrs = Y_ptr + n * (C * OUT_H * OUT_W) + c * (OUT_H * OUT_W) + oh * OUT_W + ow
    tl.store(y_ptrs, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=0, ceil_mode=True):
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding

    if ceil_mode:
        OUT_H = math.ceil((IN_H + 2*PH - KH) / SH) + 1
        OUT_W = math.ceil((IN_W + 2*PW - KW) / SW) + 1
    else:
        OUT_H = (IN_H + 2*PH - KH) // SH + 1
        OUT_W = (IN_W + 2*PW - KW) // SW + 1

    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = N * C * OUT_H * OUT_W

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W,
        OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        ceil_mode,
        numel,
    )
    return y


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    # Compute input range for this output position
    ih_start = (oh * IN_H) // OUT_H
    ih_end = ((oh + 1) * IN_H + OUT_H - 1) // OUT_H
    iw_start = (ow * IN_W) // OUT_W
    iw_end = ((ow + 1) * IN_W + OUT_W - 1) // OUT_W

    ih_start = tl.where(mask, ih_start, 0)
    ih_end = tl.where(mask, ih_end, 1)
    iw_start = tl.where(mask, iw_start, 0)
    iw_end = tl.where(mask, iw_end, 1)

    sum_val = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # For OUT_H=OUT_W=1, this reduces to global average
    # We need to handle variable ranges - use a fixed loop bound
    for ih in range(IN_H):
        for iw in range(IN_W):
            valid = mask & (ih >= ih_start) & (ih < ih_end) & (iw >= iw_start) & (iw < iw_end)
            x_ptrs = X_ptr + n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptrs, mask=valid, other=0.0)
            sum_val = sum_val + tl.where(valid, val, tl.zeros((BLOCK_SIZE,), dtype=tl.float32))
            count = count + tl.where(valid, tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32), tl.zeros((BLOCK_SIZE,), dtype=tl.float32))

    avg = sum_val / tl.maximum(count, tl.full((BLOCK_SIZE,), 1.0, dtype=tl.float32))
    y_ptrs = Y_ptr + n * (C * OUT_H * OUT_W) + c * (OUT_H * OUT_W) + oh * OUT_W + ow
    tl.store(y_ptrs, avg, mask=mask)