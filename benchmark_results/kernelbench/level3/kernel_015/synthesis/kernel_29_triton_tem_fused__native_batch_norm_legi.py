import torch
import triton
import triton.language as tl
import math

# Fused BatchNorm + ReLU kernel
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
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_elements, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Compute channel index
    c = (offsets // HW) % C
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    
    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)
    
    tl.store(out_ptr + offsets, out, mask=mask)


import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
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


import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements', 'C'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_elements, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements', 'C'],
)
@triton.jit
def fused_bn_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    n_elements, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    x_norm = (x - mean) * inv_std
    out = x_norm * w + b

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
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

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    out_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    b = tl.load(b_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + b[None, :]

    out_ptrs = out_ptr + offs_m_out[:, None] * N + offs_n_out[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    n_elements = N * C * HW
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        n_elements, C, HW,
        eps,
    )
    return out


def triton_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    n_elements = N * C * HW
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bn_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        n_elements, C, HW,
        eps,
    )
    return out


def triton_linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


# Implicit GEMM convolution kernel for 3x3 conv
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_3x3_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M_TOTAL, TOTAL_K,
    s_xn, s_xc, s_xh, s_xw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # K layout: in_c + IN_C * (kw + KW * kh)  => kh*KW*IN_C + kw*IN_C + in_c
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % 3
        kh = offs_k // (IN_C * 3)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K) &
            (batch[:, None] < BATCH)
        )

        x_ptrs = X_ptr + batch[:, None] * s_xn + in_c[None, :] * s_xc + ih * s_xh + iw * s_xw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Weight layout: [OUT_C, IN_C, KH, KW] => w[oc, in_c, kh, kw]
        # offs_n: out channel, offs_k: in_c + IN_C*(kw + KW*kh)
        w_ptrs = W_ptr + offs_n[None, :] * (IN_C * 9) + in_c[:, None] + kh[:, None] * (IN_C * 3) + kw[:, None] * IN_C
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    # Output layout: [BATCH, OUT_C, OUT_H, OUT_W]
    y_ptrs = Y_ptr + batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(y_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'IN_C'],
)
@triton.jit
def conv2d_1x1_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    M_TOTAL,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, IN_C, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)

        valid_m = (offs_m < M_TOTAL) & (batch < BATCH)
        valid_k = offs_k < IN_C

        x_ptrs = X_ptr + batch[:, None] * (IN_C * IN_H * IN_W) + offs_k[None, :] * (IN_H * IN_W) + out_h[:, None] * IN_W + out_w[:, None]
        x_mask = valid_m[:, None] & valid_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = W_ptr + offs_n[None, :] * IN_C + offs_k[:, None]
        w_mask = (offs_k[:, None] < IN_C) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(y_ptrs, acc, mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_7x7_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M_TOTAL, TOTAL_K,
    s_xn, s_xc, s_xh, s_xw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % 7
        kh = offs_k // (IN_C * 7)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K) &
            (batch[:, None] < BATCH)
        )

        x_ptrs = X_ptr + batch[:, None] * s_xn + in_c[None, :] * s_xc + ih * s_xh + iw * s_xw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        w_ptrs = W_ptr + offs_n[None, :] * (IN_C * 49) + in_c[:, None] + kh[:, None] * (IN_C * 7) + kw[:, None] * IN_C
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(y_ptrs, acc, mask=out_mask)


def triton_conv2d_7x7(x, weight, stride=2, padding=3):
    N, C, H, W = x.shape
    OUT_C = weight.shape[0]
    OUT_H = (H + 2 * padding - 7) // stride + 1
    OUT_W = (W + 2 * padding - 7) // stride + 1
    M_TOTAL = N * OUT_H * OUT_W
    TOTAL_K = C * 7 * 7

    # Use channels-last for input
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # Weight: [OUT_C, IN_C, 7, 7] -> keep as is but reorder to [OUT_C, KH, KW, IN_C]
    w_reorder = weight.permute(0, 2, 3, 1).contiguous()  # [OUT_C, 7, 7, IN_C]

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M_TOTAL, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    conv2d_7x7_kernel[grid](
        x_nhwc, w_reorder, y,
        N, C, H, W,
        OUT_C, OUT_H, OUT_W,
        stride, stride,
        padding, padding,
        M_TOTAL, TOTAL_K,
        x_nhwc.stride(0), x_nhwc.stride(3), x_nhwc.stride(1), x_nhwc.stride(2),
    )
    return y


def triton_conv2d_3x3(x, weight, stride=1, padding=1):
    N, C, H, W = x.shape
    OUT_C = weight.shape[0]
    OUT_H = (H + 2 * padding - 3) // stride + 1
    OUT_W = (W + 2 * padding - 3) // stride + 1
    M_TOTAL = N * OUT_H * OUT_W
    TOTAL_K = C * 3 * 3

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    w_reorder = weight.permute(0, 2, 3, 1).contiguous()  # [OUT_C, 3, 3, IN_C]

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M_TOTAL, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    conv2d_3x3_kernel[grid](
        x_nhwc, w_reorder, y,
        N, C, H, W,
        OUT_C, OUT_H, OUT_W,
        stride, stride,
        padding, padding,
        M_TOTAL, TOTAL_K,
        x_nhwc.stride(0), x_nhwc.stride(3), x_nhwc.stride(1), x_nhwc.stride(2),
    )
    return y


def triton_conv2d_1x1(x, weight):
    N, C, H, W = x.shape
    OUT_C = weight.shape[0]
    M_TOTAL = N * H * W

    y = torch.empty((N, OUT_C, H, W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M_TOTAL, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    conv2d_1x1_kernel[grid](
        x, weight, y,
        N, C, H, W,
        OUT_C, H, W,
        M_TOTAL,
    )
    return y


def max_pool2d(x, kernel_size=3, stride=2, padding=1):
    return torch.nn.functional.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)


def avg_pool2d(x, kernel_size=2, stride=2):
    return torch.nn.functional.avg_pool2d(x, kernel_size=kernel_size, stride=stride)


def adaptive_avg_pool2d(x, output_size=(1, 1)):
    return torch.nn.functional.adaptive_avg_pool2d(x, output_size)


def kernel_function(
    p_features_0_weight,
    p_features_1_weight,
    p_features_1_bias,
    p_dense_blocks_0_layers_0_0_weight,
    p_dense_blocks_0_layers_0_0_bias,
    p_dense_blocks_0_layers_0_2_weight,
    p_dense_blocks_0_layers_1_0_weight,
    p_dense_blocks_0_layers_1_0_bias,