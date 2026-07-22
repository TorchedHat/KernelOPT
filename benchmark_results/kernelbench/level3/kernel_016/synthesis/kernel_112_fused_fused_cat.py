import torch
import triton
import triton.language as tl
import torch.nn.functional as F

import torch
import triton
import triton.language as tl

# Fused BatchNorm + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['C', 'HW'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    w = tl.load(weight_ptr + c)
    b = tl.load(bias_ptr + c)

    inv_std = 1.0 / tl.sqrt(var + eps)

    base = n * C * HW + c * HW
    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        x_norm = (x - mean) * inv_std * w + b
        x_relu = tl.maximum(x_norm, 0.0)
        tl.store(out_ptr + base + idx, x_relu, mask=mask)


def bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        N, C, HW,
        eps,
        C=C, HW=HW,
    )
    return out


# Fused BN + ReLU for output (no relu)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['C', 'HW'],
)
@triton.jit
def bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    mean = tl.load(mean_ptr + c)
    var = tl.load(var_ptr + c)
    w = tl.load(weight_ptr + c)
    b = tl.load(bias_ptr + c)

    inv_std = 1.0 / tl.sqrt(var + eps)

    base = n * C * HW + c * HW
    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        x = x.to(tl.float32)
        x_norm = (x - mean) * inv_std * w + b
        tl.store(out_ptr + base + idx, x_norm, mask=mask)


def bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty_like(x)
    grid = (N * C,)
    bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        N, C, HW,
        eps,
        C=C, HW=HW,
    )
    return out


# Adaptive avg pool + flatten kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['C', 'HW'],
)
@triton.jit
def adaptive_avg_pool_flatten_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * HW + c * HW
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = 0
    for off in range(0, HW, BLOCK_SIZE):
        idx = off + tl.arange(0, BLOCK_SIZE)
        mask = idx < HW
        x = tl.load(x_ptr + base + idx, mask=mask, other=0.0)
        acc = acc + tl.where(mask, x.to(tl.float32), 0.0)
        count += BLOCK_SIZE

    total = tl.sum(acc, axis=0)
    avg = total / HW
    tl.store(out_ptr + n * C + c, avg)


def adaptive_avg_pool_flatten(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, device=x.device, dtype=x.dtype)
    grid = (N * C,)
    adaptive_avg_pool_flatten_kernel[grid](
        x, out,
        N, C, HW,
        C=C, HW=HW,
    )
    return out


# Linear kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 16, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=w_mask, other=0.0)

        acc = tl.dot(x, tl.trans(w), acc)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = (triton.cdiv(M, 16), triton.cdiv(N, 16))

    # Use autotune
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        M=M, N=N, K=K,
    )
    return out


# Max pool 2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['C', 'OUT_HW'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    N, C,
    IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    nc = pid // OUT_HW
    hw_off = pid % OUT_HW
    # Actually use flat indexing
    # pid covers N*C*OUT_H*OUT_W elements in blocks
    # Let's do one output element per thread in a block
    OUT_HW = OUT_H * OUT_W
    total = N * C * OUT_HW
    base_idx = pid * BLOCK_SIZE
    idx = base_idx + tl.arange(0, BLOCK_SIZE)
    mask = idx < total

    out_w = idx % OUT_W
    tmp = idx // OUT_W
    out_h = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    for kh in range(KH):
        for kw in range(KW):
            ih = out_h * SH + kh - PH
            iw = out_w * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_idx = n * C * IN_H * IN_W + c * IN_H * IN_W + ih * IN_W + iw
            x = tl.load(x_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, x)

    tl.store(out_ptr + idx, max_val, mask=mask)


def max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    OUT_H = (H + 2 * padding - kernel_size) // stride + 1
    OUT_W = (W + 2 * padding - kernel_size) // stride + 1
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    total = N * C * OUT_H * OUT_W
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    max_pool2d_kernel[grid](
        x, out,
        N, C,
        H, W,
        OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride, padding, padding,
        C=C, OUT_HW=OUT_H * OUT_W,
    )
    return out


# Avg pool 2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['C', 'OUT_HW'],
)
@triton.jit
def avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C,
    IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW, SH, SW,
    OUT_HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OUT_HW
    base_idx = pid * BLOCK_SIZE
    idx = base_idx + tl.arange(0, BLOCK_SIZE)
    mask = idx < total

    out_w = idx % OUT_W
    tmp = idx // OUT_W
    out_h = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = KH * KW
    for kh in range(KH):
        for kw in range(KW):
            ih = out_h * SH + kh
            iw = out_w * SW + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_idx = n * C * IN_H * IN_W + c * IN_H * IN_W + ih * IN_W + iw
            x = tl.load(x_ptr + x_idx, mask=valid, other=0.0)
            acc = acc + x.to(tl.float32)

    tl.store(out_ptr + idx, (acc / count).to(tl.float32), mask=mask)


def avg_pool2d(x, kernel_size=2, stride=2):
    N, C, H, W = x.shape
    OUT_H = (H - kernel_size) // stride + 1
    OUT_W = (W - kernel_size) // stride + 1
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    total = N * C * OUT_H * OUT_W
    BLOCK_SIZE = 256
    grid = (triton.cdiv(total, BLOCK_SIZE),)
    avg_pool2d_kernel[grid](
        x, out,
        N, C,
        H, W,
        OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride,
        OUT_HW=OUT_H * OUT_W,
        C=C,
    )
    return out


# Conv2d using implicit GEMM in Triton
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    SH, SW, PH, PW,
    M, N_out, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KW
        kh = tmp_k // KW

        ih = out_h_idx[:, None] * SH + kh[None, :] - PH
        iw = out_w_idx[:, None] * SW + kw[None, :] - PW

        valid_x = (
            (offs_m[:, None] < M) &
            (batch_idx[:, None] < N) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < K)
        )

        x_ptrs = (
            batch_idx[:, None] * IN_C * IN_H * IN_W +
            in_c[None, :] * IN_H * IN_W +
        )
        x_tile = tl.load(x_ptr + x_ptrs, mask=valid_x, other=0.0)

        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (
            offs_n[None, :] * IN_C * KH * KW +
            in_c[:, None] * KH * KW +
            kh[:, None] * KW +
            kw[:, None]
        )
        w_tile = tl.load(w_ptr + w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    out_ptrs = (
        batch_idx[:, None] * OUT_C * OUT_H * OUT_W +
        offs_n[None, :] * OUT_H * OUT_W +
        out_h_idx[:, None] * OUT_W +
        out_w_idx[:, None]
    )
    tl.store(out_ptr + out_ptrs, acc, mask=out_mask)


def conv2d(x, weight, stride=1, padding=0):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    if isinstance(stride, int):
        SH = SW = stride
    else:
        SH, SW = stride
    if isinstance(padding, int):
        PH = PW = padding
    else:
        PH, PW = padding

    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1

    M = N * OUT_H * OUT_W
    N_out = OUT_C
    K = IN_C * KH * KW

    out = torch.empty(N, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_out, meta['BLOCK_N']))

    conv2d_kernel[grid](
        x, weight, out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW, PH, PW,
        M, N_out, K,
        M=M, N=N_out, K=K,
    )
    return out


def dense_layer(x, bn_w, bn_b, bn_mean, bn_var, conv_w):
    """BN + ReLU + Conv2d (3x3, pad=1) + dropout(0.0)"""
    x = bn_relu(x, bn_w, bn_b, bn_mean, bn_var)
    x = conv2d(x, conv_w, stride=1, padding=1)
    # dropout with p=0.0 is identity
    return x


def kernel_function(
    p_features_0_weight,
    p_features_1_weight,
    p_features_1_bias,
    p_dense_blocks_0_layers_0_0_weight,
    p_dense_blocks_0_layers_0_0_bias,
    p_dense_blocks_0_layers_0_2_weight,
    p_dense_blocks_0_layers_1_0_weight,
    p_dense_blocks_0_layers_1_0_bias,
    p_dense_blocks_0_layers_1_2_weight,
    p_dense_blocks_0_layers_2_0_weight,
    p_dense_blocks_0_layers_2_0_bias,
    p_dense_blocks_0_layers_2_2_weight,
    p_dense_blocks_0_layers_3_0_weight,
    p_dense_blocks_0_layers_3_0_bias,
    p_dense_blocks_0_layers_3_2_weight,
    p_dense_blocks_0_layers_4_0_weight,
    p_dense_blocks_0_layers_4_0_bias,
    p_dense_blocks_0_layers_4_2_weight,
    p_dense_blocks_0_layers_5_0_weight,
    p_dense_blocks_0_layers_5_0_bias,
    p_dense_blocks_0_layers_5_2_weight,
    p_transition_layers_0_transition_0_weight,
    p_transition_layers_0_transition_0_bias,
    p_transition_layers_0_transition_2_weight,
    p_dense_blocks_1_layers_0_0_weight,
    p_dense_blocks_1_layers_0_0_bias,
    p_dense_blocks_1_layers_0_2_weight,
    p_dense_blocks_1_layers_1_0_weight,
    p_dense_blocks_1_layers_1_0_bias,
    p_dense_blocks_1_layers_1_2_weight,
    p_dense_blocks_1_layers_2_0_weight,
    p_dense_blocks_1_layers_2_0_bias,
    p_dense_blocks_1_layers_2_2_weight,
    p_dense_blocks_1_layers_3_0_weight,
    p_dense_blocks_1_layers_3_0_bias,
    p_dense_blocks_1_layers_3_2_weight,
    p_dense_blocks_1_layers_4_0_weight,
    p_dense_blocks_1_layers_4_0_bias,
    p_dense_blocks_1_layers_4_2_weight,
    p_dense_blocks_1_layers_5_0_weight,
    p_dense_blocks_1_layers_5_0_bias,
    p_dense_blocks_1_layers_5_2_weight,
    p_dense_blocks_1_layers_6_0_weight,
    p_dense_blocks_1_layers_6_0_bias,
    p_dense_blocks_1_layers_6_2_weight,
    p_dense_blocks_1_layers_7_0_weight,
    p_dense_blocks_1_layers_7_0_bias,
    p_dense_blocks_1_layers_7_2_weight,
    p_dense_blocks_1_layers_8_0_weight,
    p_dense_blocks_1_layers_8_0_bias,
    p_dense_blocks_1_layers_8_2_weight,
    p_dense_blocks_1_layers_9_0_weight,
    p_dense_blocks_1_layers_9_0_bias,
    p_dense_blocks_1_layers_9_2_weight,
    p_dense_blocks_1_layers_10_0_weight,
    p_dense_blocks_1_layers_10_0_bias,
    p_dense_blocks_1_layers_10_2_weight,
    p_dense_blocks_1_layers_11_0_weight,
    p_dense_blocks_1_layers_11_0_bias,
    p_dense_blocks_1_layers_11_2_weight,
    p_transition_layers_1_transition_0_weight,
    p_transition_layers_1_transition_0_bias,
    p_transition_layers_1_transition_2_weight,
    p_dense_blocks_2_layers_0_0_weight,
    p_dense_blocks_2_layers_0_0_bias,
    p_dense_blocks_2_layers_0_2_weight,
    p_dense_blocks_2_layers_1_0_weight,
    p_dense_blocks_2_layers_1_0_bias,
    p_dense_blocks_2_layers_1_2_weight,
    p_dense_blocks_2_layers_2_0_weight,
    p_dense_blocks_2_layers_2_0_bias,
    p_dense_blocks_2_layers_2_2_weight,
    p_dense_blocks_2_layers_3_0_weight,
    p_dense_blocks_2_layers_3_0_bias,
    p_dense_blocks_2_layers_3_2_weight,
    p_dense_blocks_2_layers_4_0_weight,
    p_dense_blocks_2_layers_4_0_bias,
    p_dense_blocks_2_layers_4_2_weight,
    p_dense_blocks_2_layers_5_0_weight,
    p_dense_blocks_2_layers_5_0_bias,
    p_dense_blocks_2_layers_5_2_weight,
    p_dense_blocks_2_layers_6_0_weight,
    p_dense_blocks_2_layers_6_0_bias,
    p_dense_blocks_2_layers_6_2_weight,
    p_dense_blocks_2_layers_7_0_weight,
    p_dense_blocks_2_layers_7_0_bias,
    p_dense_blocks_2_layers_7_2_weight,
    p_dense_blocks_2_layers_8_0_weight,
    p_dense_blocks_2_layers_8_0_bias,
    p_dense_blocks_2_layers_8_2_weight,
    p_dense_blocks_2_layers_9_0_weight,
    p_dense_blocks_2_layers_9_0_bias,
    p_dense_blocks_2_layers_9_2_weight,
    p_dense_blocks_2_layers_10_0_weight,
    p_dense_blocks_2_layers_10_0_bias,
    p_dense_blocks_2_layers_10_2_weight,
    p_dense_blocks_2_layers_11_0_weight,
    p_dense_blocks_2_layers_11_0_bias,