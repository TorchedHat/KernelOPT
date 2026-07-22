import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Fused BatchNorm (inference) + ReLU pointwise kernel
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # channel index: layout is (N, C, H, W) contiguous -> channel = (offset // HW) % C
    c_idx = (offsets // HW) % C

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var  = tl.load(var_ptr  + c_idx, mask=mask, other=1.0)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.where(out > 0.0, out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](x, out, mean, var, weight, bias, numel, C, HW, eps)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# General Conv2d via implicit GEMM in Triton
# Layout: input NCHW, weight OIHW
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    # tensor dims
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides for X (NCHW)
    sXn, sXc, sXh, sXw,
    # strides for W (OIHW)
    sWo, sWi, sWh, sWw,
    # strides for Out (NCHW)
    sOn, sOc, sOh, sOw,
    # GEMM dims
    M, N, K,
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
        tmp_k  = offs_k // KW
        kh_idx = tmp_k % KH
        ic_idx = tmp_k // KH

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m < M
        valid_k = offs_k < K
        valid_x = (
            valid_m[:, None] &
            valid_k[None, :] &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (batch_idx[:, None] < BATCH)
        )

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: shape (OUT_C, IN_C, KH, KW) -> index [offs_n, ic_idx, kh_idx, kw_idx]
        valid_w = valid_k[:, None] & (offs_n[None, :] < N)
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = (Out_ptr
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


def triton_conv2d(x, weight, stride=(1,1), padding=(0,0), groups=1):
    N, C_in, H, W = x.shape
    C_out, C_in_per_group, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OUT_H = (H + 2*PH - KH) // SH + 1
    OUT_W = (W + 2*PW - KW) // SW + 1

    out = torch.empty((N, C_out, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    if groups == 1:
        M = N * OUT_H * OUT_W
        K_dim = C_in * KH * KW
        N_dim = C_out

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N_dim, meta['BLOCK_N']),
        )
        conv2d_implicit_gemm_kernel[grid](
            x, weight, out,
            N, C_in, H, W,
            C_out, OUT_H, OUT_W,
            KH, KW,
            SH, SW,
            PH, PW,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
            out.stride(0), out.stride(1), out.stride(2), out.stride(3),
            M, N_dim, K_dim,
        )
    else:
        # Depthwise: groups == C_in == C_out, C_in_per_group == 1
        # Process each group separately
        group_size_in  = C_in // groups
        group_size_out = C_out // groups
        for g in range(groups):
            x_g = x[:, g*group_size_in:(g+1)*group_size_in, :, :]
            w_g = weight[g*group_size_out:(g+1)*group_size_out, :, :, :]
            M = N * OUT_H * OUT_W
            K_dim = group_size_in * KH * KW
            N_dim = group_size_out
            grid = lambda meta: (
                triton.cdiv(M, meta['BLOCK_M']),
                triton.cdiv(N_dim, meta['BLOCK_N']),
            )
            out_g = out[:, g*group_size_out:(g+1)*group_size_out, :, :]
            conv2d_implicit_gemm_kernel[grid](
                x_g.contiguous(), w_g.contiguous(), out_g,
                N, group_size_in, H, W,
                group_size_out, OUT_H, OUT_W,
                KH, KW,
                SH, SW,
                PH, PW,
                x_g.stride(0), x_g.stride(1), x_g.stride(2), x_g.stride(3),
                w_g.stride(0), w_g.stride(1), w_g.stride(2), w_g.stride(3),
                out_g.stride(0), out_g.stride(1), out_g.stride(2), out_g.stride(3),
                M, N_dim, K_dim,
            )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Depthwise Conv2d specialized kernel (groups == in_channels)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    sXn, sXc, sXh, sXw,
    sWc, sWh, sWw,
    sOn, sOc, sOh, sOw,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # Decode flat index -> (n, c, oh, ow)
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c  = tmp2 % C
    n  = tmp2 // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_val = tl.load(X_ptr + n*sXn + c*sXc + ih*sXh + iw*sXw, mask=valid, other=0.0)
            w_val = tl.load(W_ptr + c*sWc + kh*sWh + kw*sWw, mask=mask, other=0.0)
            acc += x_val * w_val

    tl.store(Out_ptr + n*sOn + c*sOc + oh*sOh + ow*sOw, acc, mask=mask)


def triton_depthwise_conv2d(x, weight, stride=(1,1), padding=(0,0)):
    N, C, H, W = x.shape
    # weight shape: (C, 1, KH, KW)
    _, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OUT_H = (H + 2*PH - KH) // SH + 1
    OUT_W = (W + 2*PW - KW) // SW + 1

    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = N * C * OUT_H * OUT_W

    # weight: (C, 1, KH, KW) -> strides
    w_cont = weight.squeeze(1).contiguous()  # (C, KH, KW)

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    depthwise_conv2d_kernel[grid](
        x, w_cont, out,
        N, C, H, W,
        OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w_cont.stride(0), w_cont.stride(1), w_cont.stride(2),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        numel,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Avg Pool 2d kernel
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def avg_pool2d_kernel(
    X_ptr, Out_ptr,
    BATCH, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    sXn, sXc, sXh, sXw,
    sOn, sOc, sOh, sOw,
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
    c  = tmp2 % C
    n  = tmp2 // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = KH * KW

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh
            iw = ow * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_val = tl.load(X_ptr + n*sXn + c*sXc + ih*sXh + iw*sXw, mask=valid, other=0.0)
            acc += x_val

    tl.store(Out_ptr + n*sOn + c*sOc + oh*sOh + ow*sOw, acc / count, mask=mask)


def triton_avg_pool2d(x, kernel_size):
    N, C, H, W = x.shape
    KH = KW = kernel_size
    OUT_H = H // KH
    OUT_W = W // KW

    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    numel = N * C * OUT_H * OUT_W

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    avg_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        OUT_H, OUT_W,
        KH, KW,
        KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        numel,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Linear kernel (GEMM): out = x @ W^T + b
# x: (M, K), W: (N, K), b: (N,), out: (M, N)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    sXm, sXk,
    sWn, sWk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        valid_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        valid_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        x_tile = tl.load(X_ptr + offs_m[:, None]*sXm + offs_k[None, :]*sXk,
                         mask=valid_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_n[:, None]*sWn + offs_k[None, :]*sWk,
                         mask=valid_w, other=0.0)
        acc = tl.dot(x_tile, w_tile.T, acc, allow_tf32=False)

    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Out_ptr + offs_m[:, None]*N + offs_n[None, :], acc, mask=valid_out)


def triton_linear(x, weight, bias):
    # x: (M, K), weight: (N, K), bias: (N,)
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


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel_function: MobileNetV1 forward pass
# ─────────────────────────────────────────────────────────────────────────────
def conv_bn_relu(x, conv_w, bn_w, bn_b, bn_mean, bn_var,
                 stride=(1,1), padding=(1,1), groups=1):
    if groups > 1:
        x = triton_depthwise_conv2d(x, conv_w, stride=stride, padding=padding)
    else:
        x = triton_conv2d(x, conv_w, stride=stride, padding=padding, groups=1)
    x = fused_bn_relu(x, bn_mean, bn_var, bn_w, bn_b)
    return x


def kernel_function(
    p_model_0_0_weight, p_model_0_1_weight, p_model_0_1_bias,
    p_model_1_0_weight, p_model_1_1_weight, p_model_1_1_bias,
    p_model_1_3_weight, p_model_1_4_weight, p_model_1_4_bias,
    p_model_2_0_weight, p_model_2_1_weight, p_model_2_1_bias,
    p_model_2_3_weight, p_model_2_4_weight, p_model_2_4_bias,
    p_model_3_0_weight, p_model_3_1_weight, p_model_3_1_bias,
    p_model_3_3_weight, p_model_3_4_weight, p_model_3_4_bias,
    p_model_4_0_weight, p_model_4_1_weight, p_model_4_1_bias,
    p_model_4_3_weight, p_model_4_4_weight, p_model_4_4_bias,
    p_model_5_0_weight, p_model_5_1_weight, p_model_5_1_bias,
    p_model_5_3_weight, p_model_5_4_weight, p_model_5_4_bias,
    p_model_6_0_weight, p_model_6_1_weight, p_model_6_1_bias,
    p_model_6_3_weight, p_model_6_4_weight, p_model_6_4_bias,
    p_model_7_0_weight, p_model_7_1_weight, p_model_7_1_bias,
    p_model_7_3_weight, p_model_7_4_weight, p_model_7_4_bias,
    p_model_8_0_weight, p_model_8_1_weight, p_model_8_1_bias,
    p_model_8_3_weight, p_model_8_4_weight, p_model_8_4_bias,
    p_model_9_0_weight, p_model_9_1_weight, p_model_9_1_bias,
    p_model_9_3_weight, p_model_9_4_weight, p_model_9_4_bias,
    p_model_10_0_weight, p_model_10_1_weight, p_model_10_1_bias,
    p_model_10_3_weight, p_model_10_4_weight, p_model_10_4_bias,
    p_model_11_0_weight, p_model_11_1_weight, p_model_11_1_bias,
    p_model_11_3_weight, p_model_11_4_weight, p_model_11_4_bias,
    p_model_12_0_weight, p_model_12_1_weight, p_model_12_1_bias,
    p_model_12_3_weight, p_model_12_4_weight, p_model_12_4_bias,
    p_model_13_0_weight, p_model_13_1_weight, p_model_13_1_bias,
    p_model_13_3_weight, p_model_13_4_weight, p_model_13_4_bias,
    p_fc_weight, p_fc_bias,
    b_model_0_1_running_mean, b_model_0_1_running_var,
    b_model_1_1_running_mean, b_model_1_1_running_var,
    b_model_1_4_running_mean, b_model_1_4_running_var,
    b_model_2_1_running_mean, b_model_2_1_running_var,
    b_model_2_4_running_mean, b_model_2_4_running_var,
    b_model_3_1_running_mean, b_model_3_1_running_var,
    b_model_3_4_running_mean, b_model_3_4_running_var,
    b_model_4_1_running_mean, b_model_4_1_running_var,
    b_model_4_4_running_mean, b_model_4_4_running_var,
    b_model_5_1_running_mean, b_model_5_1_running_var,