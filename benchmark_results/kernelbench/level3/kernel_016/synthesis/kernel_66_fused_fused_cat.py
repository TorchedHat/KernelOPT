import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute channel index: layout is (N, C, H, W) contiguous
    # offset = n * C * HW + c * HW + hw
    c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def bn_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=1.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    tl.store(out_ptr + offsets, out, mask=mask)


def fused_bn_relu(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * HW
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        numel, C, HW,
        eps,
    )
    return out


def fused_bn(x, weight, bias, running_mean, running_var, eps=1e-5):
    N, C, H, W = x.shape
    HW = H * W
    numel = N * C * HW
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_kernel[grid](
        x, running_mean, running_var, weight, bias,
        out,
        numel, C, HW,
        eps,
    )
    return out


# Implicit GEMM conv2d kernel for small spatial sizes
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
def conv2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N_out, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = (offs_m[:, None] < M)
        valid_k = (offs_k[None, :] < K)
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid = valid_m & valid_k & valid_ih & valid_iw

        x_idx = batch[:, None] * (IN_C * IN_H * IN_W) + in_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=valid, other=0.0)

        # Weight: (OUT_C, IN_C, KH, KW) -> index: oc * IN_C*KH*KW + in_c * KH*KW + kh * KW + kw
        w_idx = offs_n[None, :] * K + offs_k[:, None]
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(W_ptr + w_idx, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Store output: Y shape (N, OUT_C, OUT_H, OUT_W)
    valid_out_m = offs_m[:, None] < M
    valid_out_n = offs_n[None, :] < OUT_C
    valid_out = valid_out_m & valid_out_n

    y_idx = (batch[:, None] * OUT_C * OUT_H * OUT_W +
             offs_n[None, :] * OUT_H * OUT_W +
             out_h[:, None] * OUT_W +
             out_w[:, None])
    tl.store(Y_ptr + y_idx, acc.to(tl.float32), mask=valid_out)


def triton_conv2d(x, w, stride=(1,1), padding=(0,0)):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    STRIDE_H, STRIDE_W = stride
    PAD_H, PAD_W = padding
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1

    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW

    # Reshape weight to (OUT_C, K) for easier indexing
    w_flat = w.reshape(OUT_C, K)

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    conv2d_kernel[grid](
        x, w_flat, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M, OUT_C, K,
    )
    return y


def dense_layer(x, bn_w, bn_b, bn_mean, bn_var, conv_w, dropout_p=0.0):
    """BN + ReLU + Conv2d + Dropout (dropout_p=0 in inference)"""
    x = fused_bn_relu(x, bn_w, bn_b, bn_mean, bn_var)
    x = triton_conv2d(x, conv_w, stride=(1,1), padding=(1,1))
    # dropout with p=0 is identity
    return x


def transition_layer(x, bn_w, bn_b, bn_mean, bn_var, conv_w):
    """BN + ReLU + Conv2d(1x1) + AvgPool2d(2,2)"""
    x = fused_bn_relu(x, bn_w, bn_b, bn_mean, bn_var)
    x = triton_conv2d(x, conv_w, stride=(1,1), padding=(0,0))
    x = F.avg_pool2d(x, kernel_size=2, stride=2)
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
    p_dense_blocks_2_layers_11_2_weight,
    p_dense_blocks_2_layers_12_0_weight,
    p_dense_blocks_2_layers_12_0_bias,
    p_dense_blocks_2_layers_12_2_weight,
    p_dense_blocks_2_layers_13_0_weight,
    p_dense_blocks_2_layers_13_0_bias,
    p_dense_blocks_2_layers_13_2_weight,
    p_dense_blocks_2_layers_14_0_weight,
    p_dense_blocks_2_layers_14_0_bias,
    p_dense_blocks_2_layers_14_2_weight,
    p_dense_blocks_2_layers_15_0_weight,
    p_dense_blocks_2_layers_15_0_bias,
    p_dense_blocks_2_layers_15_2_weight,
    p_dense_blocks_2_layers_16_0_weight,
    p_dense_blocks_2_layers_16_0_bias,
    p_dense_blocks_2_layers_16_2_weight,
    p_dense_blocks_2_layers_17_0_weight,
    p_dense_blocks_2_layers_17_0_bias,
    p_dense_blocks_2_layers_17_2_weight,
    p_dense_blocks_2_layers_18_0_weight,
    p_dense_blocks_2_layers_18_0_bias,
    p_dense_blocks_2_layers_18_2_weight,
    p_dense_blocks_2_layers_19_0_weight,
    p_dense_blocks_2_layers_19_0_bias,
    p_dense_blocks_2_layers_19_2_weight,
    p_dense_blocks_2_layers_20_0_weight,
    p_dense_blocks_2_layers_20_0_bias,
    p_dense_blocks_2_layers_20_2_weight,
    p_dense_blocks_2_layers_21_0_weight,
    p_dense_blocks_2_layers_21_0_bias,
    p_dense_blocks_2_layers_21_2_weight,
    p_dense_blocks_2_layers_22_0_weight,
    p_dense_blocks_2_layers_22_0_bias,
    p_dense_blocks_2_layers_22_2_weight,
    p_dense_blocks_2_layers_23_0_weight,
    p_dense_blocks_2_layers_23_0_bias,
    p_dense_blocks_2_layers_23_2_weight,
    p_dense_blocks_2_layers_24_0_weight,
    p_dense_blocks_2_layers_24_0_bias,
    p_dense_blocks_2_layers_24_2_weight,
    p_dense_blocks_2_layers_25_0_weight,
    p_dense_blocks_2_layers_25_0_bias,
    p_dense_blocks_2_layers_25_2_weight,
    p_dense_blocks_2_layers_26_0_weight,
    p_dense_blocks_2_layers_26_0_bias,
    p_dense_blocks_2_layers_26_2_weight,
    p_dense_blocks_2_layers_27_0_weight,
    p_dense_blocks_2_layers_27_0_bias,
    p_dense_blocks_2_layers_27_2_weight,
    p_dense_blocks_2_layers_28_0_weight,
    p_dense_blocks_2_layers_28_0_bias,
    p_dense_blocks_2_layers_28_2_weight,
    p_dense_blocks_2_layers_29_0_weight,
    p_dense_blocks_2_layers_29_0_bias,
    p_dense_blocks_2_layers_29_2_weight,
    p_dense_blocks_2_layers_30_0_weight,
    p_dense_blocks_2_layers_30_0_bias,
    p_dense_blocks_2_layers_30_2_weight,
    p_dense_blocks_2_layers_31_0_weight,
    p_dense_blocks_2_layers_31_0_bias,
    p_dense_blocks_2_layers_31_2_weight,
    p_dense_blocks_2_layers_32_0_weight,
    p_dense_blocks_2_layers_32_0_bias,
    p_dense_blocks_2_layers_32_2_weight,
    p_dense_blocks_2_layers_33_0_weight,
    p_dense_blocks_2_layers_33_0_bias,
    p_dense_blocks_2_layers_33_2_weight,
    p_dense_blocks_2_layers_34_0_weight,
    p_dense_blocks_2_layers_34_0_bias,
    p_dense_blocks_2_layers_34_2_weight,
    p_dense_blocks_2_layers_35_0_weight,
    p_dense_blocks_2_layers_35_0_bias,
    p_dense_blocks_2_layers_35_2_weight,
    p_dense_blocks_2_layers_36_0_weight,
    p_dense_blocks_2_layers_36_0_bias,
    p_dense_blocks_2_layers_36_2_weight,
    p_dense_blocks_2_layers_37_0_weight,
    p_dense_blocks_2_layers_37_0_bias,
    p_dense_blocks_2_layers_37_2_weight,
    p_dense_blocks_2_layers_38_0_weight,
    p_dense_blocks_2_layers_38_0_bias,
    p_dense_blocks_2_layers_38_2_weight,
    p_dense_blocks_2_layers_39_0_weight,
    p_dense_blocks_2_layers_39_0_bias,
    p_dense_blocks_2_layers_39_2_weight,
    p_dense_blocks_2_layers_40_0_weight,
    p_dense_blocks_2_layers_40_0_bias,
    p_dense_blocks_2_layers_40_2_weight,
    p_dense_blocks_2_layers_41_0_weight,
    p_dense_blocks_2_layers_41_0_bias,
    p_dense_blocks_2_layers_41_2_weight,
    p_dense_blocks_2_layers_42_0_weight,
    p_dense_blocks_2_layers_42_0_bias,
    p_dense_blocks_2_layers_42_2_weight,
    p_dense_blocks_2_layers_43_0_weight,
    p_dense_blocks_2_layers_43_0_bias,
    p_dense_blocks_2_layers_43_2_weight,
    p_dense_blocks_2_layers_44_0_weight,
    p_dense_blocks_2_layers_44_0_bias,
    p_dense_blocks_2_layers_44_2_weight,
    p_dense_blocks_2_layers_45_0_weight,
    p_dense_blocks_2_layers_45_0_bias,
    p_dense_blocks_2_layers_45_2_weight,
    p_dense_blocks_2_layers_46_0_weight,
    p_dense_blocks_2_layers_46_0_bias,
    p_dense_blocks_2_layers_46_2_weight,
    p_dense_blocks_2_layers_47_0_weight,
    p_dense_blocks_2_layers_47_0_bias,
    p_dense_blocks_2_layers_47_2_weight,
    p_transition_layers_2_transition_0_weight,
    p_transition_layers_2_transition_0_bias,
    p_transition_layers_2_transition_2_weight,
    p_dense_blocks_3_layers_0_0_weight,
    p_dense_blocks_3_layers_0_0_bias,
    p_dense_blocks_3_layers_0_2_weight,
    p_dense_blocks_3_layers_1_0_weight,
    p_dense_blocks_3_layers_1_0_bias,
    p_dense_blocks_3_layers_1_2_weight,
    p_dense_blocks_3_layers_2_0_weight,
    p_dense_blocks_3_layers_2_0_bias,
    p_dense_blocks_3_layers_2_2_weight,
    p_dense_blocks_3_layers_3_0_weight,
    p_dense_blocks_3_layers_3_0_bias,
    p_dense_blocks_3_layers_3_2_weight,
    p_dense_blocks_3_layers_4_0_weight,
    p_dense_blocks_3_layers_4_0_bias,
    p_dense_blocks_3_layers_4_2_weight,
    p_dense_blocks_3_layers_5_0_weight,
    p_dense_blocks_3_layers_5_0_bias,
    p_dense_blocks_3_layers_5_2_weight,
    p_dense_blocks_3_layers_6_0_weight,
    p_dense_blocks_3_layers_6_0_bias,
    p_dense_blocks_3_layers_6_2_weight,
    p_dense_blocks_3_layers_7_0_weight,
    p_dense_blocks_3_layers_7_0_bias,
    p_dense_blocks_3_layers_7_2_weight,
    p_dense_blocks_3_layers_8_0_weight,
    p_dense_blocks_3_layers_8_0_bias,
    p_dense_blocks_3_layers_8_2_weight,
    p_dense_blocks_3_layers_9_0_weight,
    p_dense_blocks_3_layers_9_0_bias,
    p_dense_blocks_3_layers_9_2_weight,
    p_dense_blocks_3_layers_10_0_weight,
    p_dense_blocks_3_layers_10_0_bias,
    p_dense_blocks_3_layers_10_2_weight,
    p_dense_blocks_3_layers_11_0_weight,
    p_dense_blocks_3_layers_11_0_bias,
    p_dense_blocks_3_layers_11_2_weight,
    p_dense_blocks_3_layers_12_0_weight,
    p_dense_blocks_3_layers_12_0_bias,
    p_dense_blocks_3_layers_12_2_weight,
    p_dense_blocks_3_layers_13_0_weight,
    p_dense_blocks_3_layers_13_0_bias,
    p_dense_blocks_3_layers_13_2_weight,
    p_dense_blocks_3_layers_14_0_weight,
    p_dense_blocks_3_layers_14_0_bias,
    p_dense_blocks_3_layers_14_2_weight,
    p_dense_blocks_3_layers_15_0_weight,
    p_dense_blocks_3_layers_15_0_bias,