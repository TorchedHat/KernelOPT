import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ============================================================
# Fused BatchNorm + ReLU pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    activation: tl.constexpr,  # 0=none, 1=relu, 2=relu6(hardtanh 0..6)
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index: layout is (N, C, H, W) -> channel = (offset // HW) % C
    c_idx = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var = tl.load(var_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)

    # BN: (x - mean) / sqrt(var + eps) * weight + bias
    inv_std = 1.0 / libdevice.sqrt(var + eps)
    out = (x - mean) * inv_std * w + b

    # Activation
    if activation == 1:
        out = tl.maximum(out, 0.0)
    elif activation == 2:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ============================================================
# Fused Linear (GEMM) + bias kernel using tl.dot
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Implicit GEMM Conv2d kernel (for 1x1 convolutions)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """1x1 convolution as GEMM: M=N*H*W, N=out_channels, K=in_channels"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = x_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Weight: (out_channels, in_channels) -> (N, K)
        w_ptrs = w_ptr + offs_n[None, :] * K + offs_k[:, None]
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


# ============================================================
# Depthwise Conv2d kernel (3x3, groups=channels)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW', 'KH', 'KW', 'stride_h', 'stride_w'],
)
@triton.jit
def depthwise_conv_kernel(
    x_ptr, w_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Depthwise convolution: groups=C, weight shape (C, 1, KH, KW)"""
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    # Decode output index
    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp = tmp // OH
    c = tmp % C
    n = tmp // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * stride_h + kh - pad_h
            iw = ow * stride_w + kw - pad_w
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)

            x_idx = n * C * IH * IW + c * IH * IW + ih * IW + iw
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0).to(tl.float32)

            # Weight: (C, 1, KH, KW) -> index: c * KH * KW + kh * KW + kw
            w_idx = c * KH * KW + kh * KW + kw
            w_val = tl.load(w_ptr + w_idx, mask=mask, other=0.0).to(tl.float32)

            acc = acc + x_val * w_val

    tl.store(out_ptr + offsets, acc, mask=mask)


# ============================================================
# Standard Conv2d kernel (3x3, stride, padding) using implicit GEMM
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'IN_H', 'IN_W', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv3x3_kernel(
    x_ptr, w_ptr, out_ptr,
    BATCH, IN_C, OUT_C, IN_H, IN_W, OUT_H, OUT_W,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """3x3 conv as implicit GEMM. M=BATCH*OUT_H*OUT_W, N=OUT_C, K=IN_C*3*3"""
    KH = 3
    KW = 3
    TOTAL_K = IN_C * KH * KW

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (n, oh, ow)
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    n = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (ic, kh, kw)
        kw = offs_k % KW
        tmp_k = offs_k // KW
        kh = tmp_k % KH
        ic = tmp_k // KH

        ih = oh[:, None] * stride_h + kh[None, :] - pad_h
        iw = ow[:, None] * stride_w + kw[None, :] - pad_w

        valid_m = offs_m[:, None] < (BATCH * OUT_H * OUT_W)
        valid_k = offs_k[None, :] < TOTAL_K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        valid_n_idx = n[:, None] < BATCH

        x_idx = n[:, None] * IN_C * IN_H * IN_W + ic[None, :] * IN_H * IN_W + ih * IN_W + iw
        x_mask = valid_m & valid_k & valid_spatial & valid_n_idx
        x_tile = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        # Weight: (OUT_C, IN_C, KH, KW) -> (OUT_C, K)
        # w[oc, ic, kh, kw] = w_ptr[oc * IN_C*KH*KW + ic*KH*KW + kh*KW + kw]
        w_idx = offs_n[None, :] * TOTAL_K + offs_k[:, None]
        w_mask = (offs_n[None, :] < OUT_C) & (offs_k[:, None] < TOTAL_K)
        w_tile = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Store: output layout (BATCH, OUT_C, OUT_H, OUT_W)
    out_idx = n[:, None] * OUT_C * OUT_H * OUT_W + offs_n[None, :] * OUT_H * OUT_W + oh[:, None] * OUT_W + ow[:, None]
    out_mask = (offs_m[:, None] < BATCH * OUT_H * OUT_W) & (offs_n[None, :] < OUT_C) & (n[:, None] < BATCH)
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


# ============================================================
# Adaptive Average Pool 2d (global average pool to 1x1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 32}, num_warps=2),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_HW: tl.constexpr,
):
    """Global average pool: output shape (N, C, 1, 1)"""
    pid = tl.program_id(0)  # one program per (n, c) pair
    nc = pid
    n = nc // C
    c = nc % C

    base = n * C * HW + c * HW

    acc = tl.zeros([1], dtype=tl.float32)
    for hw_start in range(0, HW, BLOCK_HW):
        offs = hw_start + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    out_val = acc / HW
    tl.store(out_ptr + nc, out_val)


def apply_bn(x, weight, bias, running_mean, running_var, eps=1e-5, activation=0):
    """Apply batch norm + optional activation using Triton kernel"""
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps,
        activation,
    )
    return out


def do_conv1x1(x, weight):
    """1x1 convolution as GEMM"""
    N, C_in, H, W = x.shape
    C_out = weight.shape[0]
    M = N * H * W
    K = C_in

    # Reshape x to (M, K) - need contiguous NHWC-like layout
    # x is (N, C, H, W), we need (N*H*W, C)
    x_reshaped = x.permute(0, 2, 3, 1).contiguous().reshape(M, K)
    # weight is (C_out, C_in) for 1x1
    w_reshaped = weight.reshape(C_out, K)

    out = torch.empty((M, C_out), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv1x1_kernel[grid](
        x_reshaped, w_reshaped, out,
        M, C_out, K,
    )
    # Reshape back to (N, C_out, H, W)
    return out.reshape(N, H, W, C_out).permute(0, 3, 1, 2).contiguous()


def do_depthwise_conv(x, weight, stride=1, padding=1):
    """Depthwise 3x3 convolution"""
    N, C, IH, IW = x.shape
    KH, KW = weight.shape[2], weight.shape[3]
    OH = (IH + 2 * padding - KH) // stride + 1
    OW = (IW + 2 * padding - KW) // stride + 1

    # Make x contiguous in NCHW
    x = x.contiguous()
    # Weight: (C, 1, KH, KW) -> (C, KH, KW)
    w = weight.reshape(C, KH, KW).contiguous()

    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    depthwise_conv_kernel[grid](
        x, w, out,
        N, C, IH, IW, OH, OW,
        KH, KW,
        stride, stride,
        padding, padding,
    )
    return out


def do_conv3x3(x, weight, stride=2, padding=1):
    """Standard 3x3 convolution"""
    N, C_in, IH, IW = x.shape
    C_out = weight.shape[0]
    OH = (IH + 2 * padding - 3) // stride + 1
    OW = (IW + 2 * padding - 3) // stride + 1

    x = x.contiguous()
    w = weight.contiguous()

    out = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)
    M = N * OH * OW
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(C_out, meta['BLOCK_N']))
    conv3x3_kernel[grid](
        x, w, out,
        N, C_in, C_out, IH, IW, OH, OW,
        stride, stride,
        padding, padding,
    )
    return out


def do_linear(x, weight, bias):
    """Linear layer: x @ weight.T + bias"""
    M, K = x.shape
    N = weight.shape[0]

    x = x.contiguous()
    w = weight.contiguous()

    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, w, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
    )
    return out


def do_adaptive_avg_pool(x):
    """Global average pool to (N, C, 1, 1)"""
    N, C, H, W = x.shape
    HW = H * W
    x = x.contiguous()
    out = torch.empty((N, C), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    adaptive_avg_pool_kernel[grid](
        x, out,
        N, C, HW,
    )
    return out.reshape(N, C, 1, 1)


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_mbconv1_0_weight, p_mbconv1_1_weight, p_mbconv1_1_bias,
    p_mbconv1_3_weight, p_mbconv1_4_weight, p_mbconv1_4_bias,
    p_mbconv1_6_weight, p_mbconv1_7_weight, p_mbconv1_7_bias,
    p_mbconv2_0_weight, p_mbconv2_1_weight, p_mbconv2_1_bias,
    p_mbconv2_3_weight, p_mbconv2_4_weight, p_mbconv2_4_bias,
    p_mbconv2_6_weight, p_mbconv2_7_weight, p_mbconv2_7_bias,
    p_mbconv3_0_weight, p_mbconv3_1_weight, p_mbconv3_1_bias,
    p_mbconv3_3_weight, p_mbconv3_4_weight, p_mbconv3_4_bias,
    p_mbconv3_6_weight, p_mbconv3_7_weight, p_mbconv3_7_bias,
    p_mbconv4_0_weight, p_mbconv4_1_weight, p_mbconv4_1_bias,
    p_mbconv4_3_weight, p_mbconv4_4_weight, p_mbconv4_4_bias,
    p_mbconv4_6_weight, p_mbconv4_7_weight, p_mbconv4_7_bias,
    p_mbconv5_0_weight, p_mbconv5_1_weight, p_mbconv5_1_bias,
    p_mbconv5_3_weight, p_mbconv5_4_weight, p_mbconv5_4_bias,
    p_mbconv5_6_weight, p_mbconv5_7_weight, p_mbconv5_7_bias,
    p_mbconv6_0_weight, p_mbconv6_1_weight, p_mbconv6_1_bias,
    p_mbconv6_3_weight, p_mbconv6_4_weight, p_mbconv6_4_bias,
    p_mbconv6_6_weight, p_mbconv6_7_weight, p_mbconv6_7_bias,
    p_mbconv7_0_weight, p_mbconv7_1_weight, p_mbconv7_1_bias,
    p_mbconv7_3_weight, p_mbconv7_4_weight, p_mbconv7_4_bias,
    p_mbconv7_6_weight, p_mbconv7_7_weight, p_mbconv7_7_bias,
    p_conv2_weight, p_bn2_weight, p_bn2_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_mbconv1_1_running_mean, b_mbconv1_1_running_var,
    b_mbconv1_4_running_mean, b_mbconv1_4_running_var,
    b_mbconv1_7_running_mean, b_mbconv1_7_running_var,
    b_mbconv2_1_running_mean, b_mbconv2_1_running_var,
    b_mbconv2_4_running_mean, b_mbconv2_4_running_var,
    b_mbconv2_7_running_mean, b_mbconv2_7_running_var,
    b_mbconv3_1_running_mean, b_mbconv3_1_running_var,
    b_mbconv3_4_running_mean, b_mbconv3_4_running_var,
    b_mbconv3_7_running_mean, b_mbconv3_7_running_var,
    b_mbconv4_1_running_mean, b_mbconv4_1_running_var,
    b_mbconv4_4_running_mean, b_mbconv4_4_running_var,
    b_mbconv4_7_running_mean, b_mbconv4_7_running_var,
    b_mbconv5_1_running_mean, b_mbconv5_1_running_var,
    b_mbconv5_4_running_mean, b_mbconv5_4_running_var,
    b_mbconv5_7_running_mean, b_mbconv5_7_running_var,
    b_mbconv6_1_running_mean, b_mbconv6_1_running_var,
    b_mbconv6_4_running_mean, b_mbconv6_4_running_var,
    b_mbconv6_7_running_mean, b_mbconv6_7_running_var,
    b_mbconv7_1_running_mean, b_mbconv7_1_running_var,
    b_mbconv7_4_running_mean, b_mbconv7_4_running_var,
    b_mbconv7_7_running_mean, b_mbconv7_7_running_var,
    b_bn2_running_mean, b_bn2_running_var,
    x,
):
    # Initial conv: 3x3, stride=2, padding=1
    x = do_conv3x3(x, p_conv1_weight, stride=2, padding=1)
    x = apply_bn(x, p_bn1_weight, p_bn1_bias, b_bn1_running_mean, b_bn1_running_var, activation=1)

    # MBConv1: expand=1, in=32, out=16, stride=1
    # expand conv (1x1)
    x = do_conv1x1(x, p_mbconv1_0_weight)
    x = apply_bn(x, p_mbconv1_1_weight, p_mbconv1_1_bias, b_mbconv1_1_running_mean, b_mbconv1_1_running_var, activation=2)