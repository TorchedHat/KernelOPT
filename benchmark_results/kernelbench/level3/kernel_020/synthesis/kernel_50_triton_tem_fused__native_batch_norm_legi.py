import torch
import triton
import triton.language as tl
import math



import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_bn_relu6_kernel(
    X_ptr, W_ptr, BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    Y_ptr,
    # Input dims
    BATCH, IN_C, IN_H, IN_W,
    # Output dims
    OUT_C, OUT_H, OUT_W,
    # Kernel params
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    GROUPS,
    # BN params
    BN_EPS,
    # Apply relu6?
    APPLY_RELU6: tl.constexpr,
    # Strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # Strides for W (out_c, in_c, kh, kw) -> stored as (out_c, kh, kw, in_c) for NHWC
    s_woc, s_wkh, s_wkw, s_wic,
    # Strides for Y (NHWC layout)
    s_yn, s_yh, s_yw, s_yc,
    # Total M, N, K for autotuning key
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W
    # N = OUT_C (per group)
    # K = GROUP_IN_C * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS

    # Which group does this pid_n belong to?
    group = (pid_n * BLOCK_N) // GROUP_OUT_C
    group = tl.minimum(group, GROUPS - 1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    # Local channel index within group
    local_n = offs_n - group * GROUP_OUT_C

    TOTAL_K = GROUP_IN_C * KH * KW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c_local = offs_k % GROUP_IN_C
        kw_idx = (offs_k // GROUP_IN_C) % KW
        kh_idx = offs_k // (GROUP_IN_C * KW)

        in_c_global = in_c_local + group * GROUP_IN_C

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = (batch_idx < BATCH) & (out_h_idx < OUT_H) & (out_w_idx < OUT_W)
        valid_k = offs_k < TOTAL_K
        valid_x = (
            valid_m[:, None]
            & (ih >= 0) & (ih < IN_H)
            & (iw >= 0) & (iw < IN_W)
        )

        x_ptrs = (
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: stored as (OUT_C, KH, KW, GROUP_IN_C) for NHWC
        valid_w = (offs_k[:, None] < TOTAL_K) & (local_n[None, :] >= 0) & (local_n[None, :] < GROUP_OUT_C)
        w_ptrs = (
        )
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Apply BN + optional ReLU6
    valid_n = (offs_n >= 0) & (offs_n < OUT_C)
    bn_mean = tl.load(BN_mean_ptr + offs_n, mask=valid_n, other=0.0)
    bn_var = tl.load(BN_var_ptr + offs_n, mask=valid_n, other=1.0)
    bn_w = tl.load(BN_weight_ptr + offs_n, mask=valid_n, other=1.0)
    bn_b = tl.load(BN_bias_ptr + offs_n, mask=valid_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + BN_EPS)
    acc = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]

    if APPLY_RELU6:
        acc = tl.maximum(acc, 0.0)
        acc = tl.minimum(acc, 6.0)

    # Store output
    valid_m2 = (batch_idx < BATCH) & (out_h_idx < OUT_H) & (out_w_idx < OUT_W)
    mask_out = valid_m2[:, None] & valid_n[None, :]

    y_ptrs = (
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_k[:, None] * N + offs_n[None, :], mask=mask_w, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def avg_pool_kernel(
    X_ptr, Y_ptr,
    BATCH, C, H, W,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    batch = pid // C
    c = pid % C

    HW = H * W
    acc = tl.zeros((1,), dtype=tl.float32)

    for start in range(0, HW, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < HW
        h_idx = offs // W
        w_idx = offs % W
        # X is NHWC
        x_ptrs = X_ptr + batch * (H * W * C) + h_idx * (W * C) + w_idx * C + c
        vals = tl.load(x_ptrs, mask=mask & (batch < BATCH), other=0.0)
        acc += tl.sum(vals.to(tl.float32), axis=0)

    avg = acc / (H * W)
    tl.store(Y_ptr + batch * C + c, avg, mask=(batch < BATCH) & (c < C))


def conv2d_bn_relu6(
    x_nhwc, w, bn_mean, bn_var, bn_weight, bn_bias,
    stride_h, stride_w, pad_h, pad_w, groups, apply_relu6,
):
    """
    """
    N, H, W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w.shape

    # Convert weight to NHWC-friendly layout: (OUT_C, KH, KW, IN_C//groups)
    w_nhwc = w.permute(0, 2, 3, 1).contiguous()

    M = N * out_h * out_w
    Nk = OUT_C
    K = (IN_C // groups) * KH * KW

    y = torch.empty((N, out_h, out_w, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    # Strides for x_nhwc
    s_xn, s_xh, s_xw, s_xc = x_nhwc.stride()
    # Strides for w_nhwc (OUT_C, KH, KW, IN_C//groups)
    s_woc, s_wkh, s_wkw, s_wic = w_nhwc.stride()
    # Strides for y
    s_yn, s_yh, s_yw, s_yc = y.stride()

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Nk, meta['BLOCK_N']),
    )

    conv2d_bn_relu6_kernel[grid](
        x_nhwc, w_nhwc, bn_mean, bn_var, bn_weight, bn_bias,
        y,
        N, IN_C, H, W,
        OUT_C, out_h, out_w,
        KH, KW, stride_h, stride_w, pad_h, pad_w,
        groups,
        1e-5,
        apply_relu6,
        s_xn, s_xh, s_xw, s_xc,
        s_woc, s_wkh, s_wkw, s_wic,
        s_yn, s_yh, s_yw, s_yc,
        M, Nk, K,
    )
    return y


def compute_out_size(in_size, kernel, stride, pad):
    return (in_size + 2 * pad - kernel) // stride + 1


def kernel_function(
    p_features_0_weight, p_features_1_weight, p_features_1_bias,
    p_features_3_0_weight, p_features_3_1_weight, p_features_3_1_bias,
    p_features_3_3_weight, p_features_3_4_weight, p_features_3_4_bias,
    p_features_4_0_weight, p_features_4_1_weight, p_features_4_1_bias,
    p_features_4_3_weight, p_features_4_4_weight, p_features_4_4_bias,
    p_features_4_6_weight, p_features_4_7_weight, p_features_4_7_bias,
    p_features_5_0_weight, p_features_5_1_weight, p_features_5_1_bias,
    p_features_5_3_weight, p_features_5_4_weight, p_features_5_4_bias,
    p_features_5_6_weight, p_features_5_7_weight, p_features_5_7_bias,
    p_features_6_0_weight, p_features_6_1_weight, p_features_6_1_bias,
    p_features_6_3_weight, p_features_6_4_weight, p_features_6_4_bias,
    p_features_6_6_weight, p_features_6_7_weight, p_features_6_7_bias,
    p_features_7_0_weight, p_features_7_1_weight, p_features_7_1_bias,
    p_features_7_3_weight, p_features_7_4_weight, p_features_7_4_bias,
    p_features_7_6_weight, p_features_7_7_weight, p_features_7_7_bias,
    p_features_8_0_weight, p_features_8_1_weight, p_features_8_1_bias,
    p_features_8_3_weight, p_features_8_4_weight, p_features_8_4_bias,
    p_features_8_6_weight, p_features_8_7_weight, p_features_8_7_bias,
    p_features_9_0_weight, p_features_9_1_weight, p_features_9_1_bias,
    p_features_9_3_weight, p_features_9_4_weight, p_features_9_4_bias,
    p_features_9_6_weight, p_features_9_7_weight, p_features_9_7_bias,
    p_features_10_0_weight, p_features_10_1_weight, p_features_10_1_bias,
    p_features_10_3_weight, p_features_10_4_weight, p_features_10_4_bias,
    p_features_10_6_weight, p_features_10_7_weight, p_features_10_7_bias,
    p_features_11_0_weight, p_features_11_1_weight, p_features_11_1_bias,
    p_features_11_3_weight, p_features_11_4_weight, p_features_11_4_bias,
    p_features_11_6_weight, p_features_11_7_weight, p_features_11_7_bias,
    p_features_12_0_weight, p_features_12_1_weight, p_features_12_1_bias,
    p_features_12_3_weight, p_features_12_4_weight, p_features_12_4_bias,
    p_features_12_6_weight, p_features_12_7_weight, p_features_12_7_bias,
    p_features_13_0_weight, p_features_13_1_weight, p_features_13_1_bias,
    p_features_13_3_weight, p_features_13_4_weight, p_features_13_4_bias,
    p_features_13_6_weight, p_features_13_7_weight, p_features_13_7_bias,
    p_features_14_0_weight, p_features_14_1_weight, p_features_14_1_bias,
    p_features_14_3_weight, p_features_14_4_weight, p_features_14_4_bias,
    p_features_14_6_weight, p_features_14_7_weight, p_features_14_7_bias,
    p_features_15_0_weight, p_features_15_1_weight, p_features_15_1_bias,
    p_features_15_3_weight, p_features_15_4_weight, p_features_15_4_bias,
    p_features_15_6_weight, p_features_15_7_weight, p_features_15_7_bias,
    p_features_16_0_weight, p_features_16_1_weight, p_features_16_1_bias,
    p_features_16_3_weight, p_features_16_4_weight, p_features_16_4_bias,
    p_features_16_6_weight, p_features_16_7_weight, p_features_16_7_bias,
    p_features_17_0_weight, p_features_17_1_weight, p_features_17_1_bias,
    p_features_17_3_weight, p_features_17_4_weight, p_features_17_4_bias,
    p_features_17_6_weight, p_features_17_7_weight, p_features_17_7_bias,
    p_features_18_0_weight, p_features_18_1_weight, p_features_18_1_bias,
    p_features_18_3_weight, p_features_18_4_weight, p_features_18_4_bias,
    p_features_18_6_weight, p_features_18_7_weight, p_features_18_7_bias,
    p_features_19_0_weight, p_features_19_1_weight, p_features_19_1_bias,
    p_features_19_3_weight, p_features_19_4_weight, p_features_19_4_bias,
    p_features_19_6_weight, p_features_19_7_weight, p_features_19_7_bias,
    p_features_20_weight, p_features_21_weight, p_features_21_bias,
    p_classifier_1_weight, p_classifier_1_bias,
    b_features_1_running_mean, b_features_1_running_var,
    b_features_3_1_running_mean, b_features_3_1_running_var,
    b_features_3_4_running_mean, b_features_3_4_running_var,
    b_features_4_1_running_mean, b_features_4_1_running_var,
    b_features_4_4_running_mean, b_features_4_4_running_var,
    b_features_4_7_running_mean, b_features_4_7_running_var,
    b_features_5_1_running_mean, b_features_5_1_running_var,
    b_features_5_4_running_mean, b_features_5_4_running_var,
    b_features_5_7_running_mean, b_features_5_7_running_var,
    b_features_6_1_running_mean, b_features_6_1_running_var,
    b_features_6_4_running_mean, b_features_6_4_running_var,
    b_features_6_7_running_mean, b_features_6_7_running_var,
    b_features_7_1_running_mean, b_features_7_1_running_var,
    b_features_7_4_running_mean, b_features_7_4_running_var,
    b_features_7_7_running_mean, b_features_7_7_running_var,
    b_features_8_1_running_mean, b_features_8_1_running_var,
    b_features_8_4_running_mean, b_features_8_4_running_var,
    b_features_8_7_running_mean, b_features_8_7_running_var,
    b_features_9_1_running_mean, b_features_9_1_running_var,
    b_features_9_4_running_mean, b_features_9_4_running_var,
    b_features_9_7_running_mean, b_features_9_7_running_var,
    b_features_10_1_running_mean, b_features_10_1_running_var,
    b_features_10_4_running_mean, b_features_10_4_running_var,
    b_features_10_7_running_mean, b_features_10_7_running_var,
    b_features_11_1_running_mean, b_features_11_1_running_var,
    b_features_11_4_running_mean, b_features_11_4_running_var,
    b_features_11_7_running_mean, b_features_11_7_running_var,
    b_features_12_1_running_mean, b_features_12_1_running_var,
    b_features_12_4_running_mean, b_features_12_4_running_var,
    b_features_12_7_running_mean, b_features_12_7_running_var,
    b_features_13_1_running_mean, b_features_13_1_running_var,
    b_features_13_4_running_mean, b_features_13_4_running_var,
    b_features_13_7_running_mean, b_features_13_7_running_var,
    b_features_14_1_running_mean, b_features_14_1_running_var,
    b_features_14_4_running_mean, b_features_14_4_running_var,
    b_features_14_7_running_mean, b_features_14_7_running_var,
    b_features_15_1_running_mean, b_features_15_1_running_var,
    b_features_15_4_running_mean, b_features_15_4_running_var,
    b_features_15_7_running_mean, b_features_15_7_running_var,
    b_features_16_1_running_mean, b_features_16_1_running_var,
    b_features_16_4_running_mean, b_features_16_4_running_var,
    b_features_16_7_running_mean, b_features_16_7_running_var,
    b_features_17_1_running_mean, b_features_17_1_running_var,
    b_features_17_4_running_mean, b_features_17_4_running_var,
    b_features_17_7_running_mean, b_features_17_7_running_var,
    b_features_18_1_running_mean, b_features_18_1_running_var,
    b_features_18_4_running_mean, b_features_18_4_running_var,
    b_features_18_7_running_mean, b_features_18_7_running_var,
    b_features_19_1_running_mean, b_features_19_1_running_var,
    b_features_19_4_running_mean, b_features_19_4_running_var,
    b_features_19_7_running_mean, b_features_19_7_running_var,
    b_features_21_running_mean, b_features_21_running_var,
    x,
):
    device = x.device

    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    N = x.shape[0]

    # features[0]: Conv2d(3, 32, 3, stride=2, padding=1) + BN + ReLU6
    # Input: (10, 224, 224, 3), Output: (10, 112, 112, 32)
    oh = compute_out_size(224, 3, 2, 1)
    ow = compute_out_size(224, 3, 2, 1)
    x = conv2d_bn_relu6(
        x_nhwc, p_features_0_weight,
        b_features_1_running_mean, b_features_1_running_var,
        p_features_1_weight, p_features_1_bias,
    )
    # x is now NHWC: (10, 112, 112, 32)

    # features[3][0]: DW Conv2d(32, 32, 3, stride=1, padding=1, groups=32) + BN + ReLU6
    oh = compute_out_size(112, 3, 1, 1)
    ow = compute_out_size(112, 3, 1, 1)
    x = conv2d_bn_relu6(
        x, p_features_3_0_weight,
        b_features_3_1_running_mean, b_features_3_1_running_var,
        p_features_3_1_weight, p_features_3_1_bias,
    )
    # (10, 112, 112, 32)

    # features[3][3]: PW Conv2d(32, 16, 1) + BN (no relu)
    oh = compute_out_size(112, 1, 1, 0)
    ow = compute_out_size(112, 1, 1, 0)
    x = conv2d_bn_relu6(
        x, p_features_3_3_weight,