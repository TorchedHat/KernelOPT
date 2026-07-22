import torch
import triton
import triton.language as tl

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc=acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask_c)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, spatial,
    eps: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # channel index
    c_idx = (offs // spatial) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(var_ptr + c_idx, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    if APPLY_RELU:
        out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def sigmoid_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['numel'],
)
@triton.jit
def relu_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5, apply_relu=True):
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    spatial = x.numel() // (x.shape[0] * x.shape[1])
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out, mean, var, weight, bias,
        numel, C, spatial,
        eps=eps,
        APPLY_RELU=apply_relu,
    )
    return out


def triton_sigmoid(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


def triton_relu(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def triton_linear(x, weight, bias=None):
    # x: (M, K), weight: (N, K) -> out: (M, N)
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    matmul_kernel[grid](
        x, weight, out, bias if bias is not None else x,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
    )
    return out


def conv2d_triton_impl(x, weight, bias=None, stride=1, padding=0, groups=1):
    """Implement conv2d using Triton implicit GEMM."""
    N, C_in, H, W = x.shape
    C_out, C_in_per_group, KH, KW = weight.shape
    
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding

    OUT_H = (H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (W + 2 * pad_w - KW) // stride_w + 1

    # Use PyTorch's conv2d since implementing full conv2d in Triton for all cases
    # is extremely complex and the spec says to use library calls for conv
    # But since it's banned, we implement via im2col + matmul approach
    # Actually let's implement it properly with Triton GEMM
    
    # For simplicity with groups (depthwise), handle separately
    if groups == C_in and groups == C_out:
        # Depthwise convolution - use specialized kernel
        return _depthwise_conv2d(x, weight, bias, stride_h, stride_w, pad_h, pad_w)
    
    # Standard conv2d via implicit GEMM
    return _conv2d_gemm(x, weight, bias, stride_h, stride_w, pad_h, pad_w, groups)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, C_in, H, W_dim, C_out, OUT_H, OUT_W,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    M, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wkh, stride_wkw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: m = batch * OUT_H * OUT_W + oh * OUT_W + ow
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K index: k = kh * KW * C_in + kw * C_in + c_in
        c_in_idx = offs_k % C_in
        tmp_k = offs_k // C_in
        kw_idx = tmp_k % KW
        kh_idx = tmp_k // KW

        # Input coordinates
        ih = out_h_idx[:, None] * stride_h + kh_idx[None, :] - pad_h
        iw = out_w_idx[:, None] * stride_w + kw_idx[None, :] - pad_w

        valid_m = offs_m < M
        valid_k = offs_k < K
        valid_ih = (ih >= 0) & (ih < H)
        valid_iw = (iw >= 0) & (iw < W_dim)
        valid_batch = batch_idx < N

        x_mask = valid_m[:, None] & valid_k[None, :] & valid_ih & valid_iw & valid_batch[:, None]

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Weight: [C_out, C_in, KH, KW] -> index [n_out, k]
        # k = kh * KW * C_in + kw * C_in + c_in
        w_ptrs = (W_ptr
        w_mask = valid_k[:, None] & (offs_n[None, :] < C_out)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Store output: [N, C_out, OUT_H, OUT_W]
    out_ptrs = (Out_ptr
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < C_out)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def _conv2d_gemm(x, weight, bias, stride_h, stride_w, pad_h, pad_w, groups=1):
    N, C_in, H, W = x.shape
    C_out, C_in_per_group, KH, KW = weight.shape
    OUT_H = (H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (W + 2 * pad_w - KW) // stride_w + 1
    M = N * OUT_H * OUT_W
    K = C_in_per_group * KH * KW

    out = torch.empty((N, C_out, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv2d_implicit_gemm_kernel[grid](
        x, weight, out,
        N, C_in, H, W, C_out, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        M, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
    )

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 64}, num_warps=8),
    ],
    key=['M', 'C', 'KH', 'KW'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, C, H, W_dim, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_h, stride_w,
    pad_h, pad_w,
    M,
    stride_xn, stride_xc, stride_xh, stride_xw,
    BLOCK_M: tl.constexpr, BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h_idx[:, None] * stride_h + kh - pad_h
            iw = out_w_idx[:, None] * stride_w + kw - pad_w

            valid_m = offs_m < M
            valid_c = offs_c < C
            valid_ih = (ih >= 0) & (ih < H)
            valid_iw = (iw >= 0) & (iw < W_dim)
            valid_batch = batch_idx < N

            x_mask = valid_m[:, None] & valid_c[None, :] & valid_ih & valid_iw & valid_batch[:, None]

            x_ptrs = (X_ptr
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Weight for depthwise: [C, 1, KH, KW]
            w_ptrs = W_ptr + offs_c * (KH * KW) + kh * KW + kw
            w_tile = tl.load(w_ptrs, mask=valid_c, other=0.0)

            acc = acc + x_tile * w_tile[None, :]

    out_ptrs = (Out_ptr
    out_mask = (offs_m[:, None] < M) & (offs_c[None, :] < C)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def _depthwise_conv2d(x, weight, bias, stride_h, stride_w, pad_h, pad_w):
    N, C, H, W = x.shape
    KH = weight.shape[2]
    KW = weight.shape[3]
    OUT_H = (H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (W + 2 * pad_w - KW) // stride_w + 1
    M = N * OUT_H * OUT_W

    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C, meta['BLOCK_C']))
    depthwise_conv2d_kernel[grid](
        x, weight, out,
        N, C, H, W, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        M,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
    )

    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out


def adaptive_avg_pool2d_1x1(x):
    """Adaptive average pool to (1,1) - just mean over spatial dims."""
    return x.mean(dim=(2, 3), keepdim=True)


def mbconv_block(x, layers_weights, expand_ratio, stride, in_channels, out_channels,
                 bn_eps=1e-5):
    """
    """
    expanded_channels = in_channels * expand_ratio

    idx = 0
    if expand_ratio != 1:
        # Expansion conv
        w0 = layers_weights[idx]; idx += 1
        w1 = layers_weights[idx]; idx += 1  # bn weight
        w2 = layers_weights[idx]; idx += 1  # bn bias
        rm = layers_weights[idx]; idx += 1  # running mean
        rv = layers_weights[idx]; idx += 1  # running var
        x = _conv2d_gemm(x, w0, None, 1, 1, 0, 0)
        x = triton_bn_relu(x, rm, rv, w1, w2, eps=bn_eps, apply_relu=True)

    # Depthwise conv
    dw_w = layers_weights[idx]; idx += 1
    dw_bn_w = layers_weights[idx]; idx += 1
    dw_bn_b = layers_weights[idx]; idx += 1
    dw_rm = layers_weights[idx]; idx += 1
    dw_rv = layers_weights[idx]; idx += 1
    x = _depthwise_conv2d(x, dw_w, None, stride, stride, 1, 1)
    x = triton_bn_relu(x, dw_rm, dw_rv, dw_bn_w, dw_bn_b, eps=bn_eps, apply_relu=True)

    # SE block
    se_pool = adaptive_avg_pool2d_1x1(x)
    se_w1 = layers_weights[idx]; idx += 1
    se_w2 = layers_weights[idx]; idx += 1
    se_squeezed = _conv2d_gemm(se_pool, se_w1, None, 1, 1, 0, 0)
    se_squeezed = triton_relu(se_squeezed)
    se_excited = _conv2d_gemm(se_squeezed, se_w2, None, 1, 1, 0, 0)
    se_excited = triton_sigmoid(se_excited)
    x = x * se_excited

    # Output conv + BN (no relu)
    out_w = layers_weights[idx]; idx += 1
    out_bn_w = layers_weights[idx]; idx += 1
    out_bn_b = layers_weights[idx]; idx += 1
    out_rm = layers_weights[idx]; idx += 1
    out_rv = layers_weights[idx]; idx += 1
    x = _conv2d_gemm(x, out_w, None, 1, 1, 0, 0)
    x = triton_bn_relu(x, out_rm, out_rv, out_bn_w, out_bn_b, eps=bn_eps, apply_relu=False)

    return x


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_mbconv1_0_weight, p_mbconv1_1_weight, p_mbconv1_1_bias,
    p_mbconv1_3_weight, p_mbconv1_4_weight, p_mbconv1_4_bias,
    p_mbconv1_7_weight, p_mbconv1_9_weight,
    p_mbconv1_11_weight, p_mbconv1_12_weight, p_mbconv1_12_bias,
    p_mbconv2_0_weight, p_mbconv2_1_weight, p_mbconv2_1_bias,
    p_mbconv2_3_weight, p_mbconv2_4_weight, p_mbconv2_4_bias,
    p_mbconv2_7_weight, p_mbconv2_9_weight,
    p_mbconv2_11_weight, p_mbconv2_12_weight, p_mbconv2_12_bias,
    p_mbconv3_0_weight, p_mbconv3_1_weight, p_mbconv3_1_bias,
    p_mbconv3_3_weight, p_mbconv3_4_weight, p_mbconv3_4_bias,
    p_mbconv3_7_weight, p_mbconv3_9_weight,
    p_mbconv3_11_weight, p_mbconv3_12_weight, p_mbconv3_12_bias,
    p_mbconv4_0_weight, p_mbconv4_1_weight, p_mbconv4_1_bias,
    p_mbconv4_3_weight, p_mbconv4_4_weight, p_mbconv4_4_bias,
    p_mbconv4_7_weight, p_mbconv4_9_weight,
    p_mbconv4_11_weight, p_mbconv4_12_weight, p_mbconv4_12_bias,
    p_mbconv5_0_weight, p_mbconv5_1_weight, p_mbconv5_1_bias,