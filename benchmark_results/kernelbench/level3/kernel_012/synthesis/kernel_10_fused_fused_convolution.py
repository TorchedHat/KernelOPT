import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Fused Conv2d (implicit GEMM) + optional bias + optional ReLU
# Input layout: NCHW, Weight layout: (Co, Ci, KH, KW)
# We convert internally to NHWC for coalesced access
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KH', 'KW'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    HAS_BIAS: tl.constexpr, HAS_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Decode M index -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW

    # X is in NHWC layout: stride_n = IN_H*IN_W*IN_C, stride_h = IN_W*IN_C, stride_w = IN_C, stride_c = 1
    # W is in (OUT_C, IN_C, KH, KW) -> we access as (KH, KW, IN_C, OUT_C) effectively
    # W original: w[oc, ic, kh, kw] = W_ptr + oc*(IN_C*KH*KW) + ic*(KH*KW) + kh*KW + kw

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (ic, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        ic_idx = tmp_k // KH

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m[:, None] & valid_k[None, :] & valid_ih & valid_iw

        # X in NHWC: index = batch*IN_H*IN_W*IN_C + ih*IN_W*IN_C + iw*IN_C + ic
        x_idx = batch[:, None] * (IN_H * IN_W * IN_C) + ih * (IN_W * IN_C) + iw * IN_C + ic_idx[None, :]
        x_tile = tl.load(X_ptr + x_idx, mask=mask_x, other=0.0)

        # W in (OUT_C, IN_C, KH, KW): w[oc, ic, kh, kw]
        # For our tile: w_tile[k, n] = W[offs_n[n], ic_idx[k], kh_idx[k], kw_idx[k]]
        valid_n = offs_n < OUT_C
        mask_w = valid_k[:, None] & valid_n[None, :]
        w_idx = offs_n[None, :] * (IN_C * KH * KW) + ic_idx[:, None] * (KH * KW) + kh_idx[:, None] * KW + kw_idx[:, None]
        w_tile = tl.load(W_ptr + w_idx, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias and apply ReLU
    valid_m = offs_m < M_TOTAL
    valid_n = offs_n < OUT_C
    mask_out = valid_m[:, None] & valid_n[None, :]

    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=valid_n, other=0.0)
        acc = acc + bias[None, :]

    if HAS_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store in NCHW: Y[batch, oc, out_h, out_w] = Y_ptr + batch*(OUT_C*OUT_H*OUT_W) + oc*(OUT_H*OUT_W) + out_h*OUT_W + out_w
    y_idx = batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(Y_ptr + y_idx, acc.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Max Pool 2D kernel (kernel=2, stride=2)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH: tl.constexpr, C: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    N_TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: offs -> (batch, c, out_h, out_w)
    out_w = offs % OUT_W
    tmp = offs // OUT_W
    out_h = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    batch = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h * STRIDE_H + kh
            iw = out_w * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_idx = batch * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    y_idx = batch * (C * OUT_H * OUT_W) + c * (OUT_H * OUT_W) + out_h * OUT_W + out_w
    tl.store(Y_ptr + y_idx, max_val, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Linear + bias + optional ReLU kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    HAS_RELU: tl.constexpr,
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

    if HAS_RELU:
        acc = tl.maximum(acc, 0.0)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def conv2d_triton(x_nhwc, w, b, out_h, out_w, stride_h, stride_w, pad_h, pad_w, has_relu):
    """x_nhwc: (N, H, W, C), w: (OUT_C, IN_C, KH, KW)"""
    N, H, W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w.shape
    M_TOTAL = N * out_h * out_w

    y = torch.empty((N, OUT_C, out_h, out_w), device=x_nhwc.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        y,
        N, IN_C, H, W,
        OUT_C, out_h, out_w,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        b is not None, has_relu,
        BATCH=N, IN_C=IN_C, IN_H=H, IN_W=W,
        OUT_C=OUT_C, OUT_H=out_h, OUT_W=out_w,
        KH=KH, KW=KW,
        STRIDE_H=stride_h, STRIDE_W=stride_w,
        PAD_H=pad_h, PAD_W=pad_w,
        HAS_BIAS=b is not None,
        HAS_RELU=has_relu,
    )
    return y


def maxpool2d_triton(x, kh, kw, stride_h, stride_w):
    """x: (N, C, H, W) NCHW"""
    N, C, H, W = x.shape
    out_h = (H - kh) // stride_h + 1
    out_w = (W - kw) // stride_w + 1
    N_TOTAL = N * C * out_h * out_w

    y = torch.empty((N, C, out_h, out_w), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)

    maxpool2d_kernel[grid](
        x, y,
        N, C, H, W, out_h, out_w,
        kh, kw, stride_h, stride_w,
        N_TOTAL=N_TOTAL,
        BATCH=N, C=C, IN_H=H, IN_W=W, OUT_H=out_h, OUT_W=out_w,
        KH=kh, KW=kw, STRIDE_H=stride_h, STRIDE_W=stride_w,
    )
    return y


def linear_triton(x, w, b, has_relu):
    """x: (M, K), w: (N, K) -> (M, N)"""
    M, K = x.shape
    N = w.shape[0]

    # W needs to be transposed: we want W[k, n]
    w_t = w.t().contiguous()  # (K, N)

    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_bias_relu_kernel[grid](
        x, w_t, b, y,
        M=M, N=N, K=K,
        HAS_RELU=has_relu,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel_function implementing VGG19 forward pass
# ─────────────────────────────────────────────────────────────────────────────

def kernel_function(
    p_features_0_weight, p_features_0_bias,
    p_features_2_weight, p_features_2_bias,
    p_features_5_weight, p_features_5_bias,
    p_features_7_weight, p_features_7_bias,
    p_features_10_weight, p_features_10_bias,
    p_features_12_weight, p_features_12_bias,
    p_features_14_weight, p_features_14_bias,
    p_features_16_weight, p_features_16_bias,
    p_features_19_weight, p_features_19_bias,
    p_features_21_weight, p_features_21_bias,
    p_features_23_weight, p_features_23_bias,
    p_features_25_weight, p_features_25_bias,
    p_features_28_weight, p_features_28_bias,
    p_features_30_weight, p_features_30_bias,
    p_features_32_weight, p_features_32_bias,
    p_features_34_weight, p_features_34_bias,
    p_classifier_0_weight, p_classifier_0_bias,
    p_classifier_3_weight, p_classifier_3_bias,
    p_classifier_6_weight, p_classifier_6_bias,
    x,
):
    # Convert input to NHWC for conv kernels
    # x: (N, 3, 224, 224) NCHW
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)

    def conv_relu(x_nhwc, w, b, pad=1, stride=1):
        N, H, W, _ = x_nhwc.shape
        OUT_C = w.shape[0]
        KH, KW = w.shape[2], w.shape[3]
        out_h = (H + 2*pad - KH) // stride + 1
        out_w = (W + 2*pad - KW) // stride + 1
        # Returns NCHW
        y_nchw = conv2d_triton(x_nhwc, w, b, out_h, out_w, stride, stride, pad, pad, True)
        # Convert back to NHWC for next conv
        return y_nchw.permute(0, 2, 3, 1).contiguous(), y_nchw

    def do_conv_relu(x_nhwc, w, b):
        y_nhwc, y_nchw = conv_relu(x_nhwc, w, b)
        return y_nhwc

    # Block 1
    x_nhwc = do_conv_relu(x_nhwc, p_features_0_weight, p_features_0_bias)
    x_nhwc = do_conv_relu(x_nhwc, p_features_2_weight, p_features_2_bias)
    # MaxPool: need NCHW
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_nchw = maxpool2d_triton(x_nchw, 2, 2, 2, 2)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    # Block 2
    x_nhwc = do_conv_relu(x_nhwc, p_features_5_weight, p_features_5_bias)
    x_nhwc = do_conv_relu(x_nhwc, p_features_7_weight, p_features_7_bias)
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_nchw = maxpool2d_triton(x_nchw, 2, 2, 2, 2)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    # Block 3
    x_nhwc = do_conv_relu(x_nhwc, p_features_10_weight, p_features_10_bias)
    x_nhwc = do_conv_relu(x_nhwc, p_features_12_weight, p_features_12_bias)
    x_nhwc = do_conv_relu(x_nhwc, p_features_14_weight, p_features_14_bias)
    x_nhwc = do_conv_relu(x_nhwc, p_features_16_weight, p_features_16_bias)
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_nchw = maxpool2d_triton(x_nchw, 2, 2, 2, 2)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    # Block 4
    x_nhwc = do_conv_relu(x_nhwc, p_features_19_weight, p_features_19_bias)
    x_nhwc = do_conv_relu(x_nhwc, p_features_21_weight, p_features_21_bias)
    x_nhwc = do_conv_relu(x_nhwc, p_features_23_weight, p_features_23_bias)
    x_nhwc = do_conv_relu(x_nhwc, p_features_25_weight, p_features_25_bias)
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_nchw = maxpool2d_triton(x_nchw, 2, 2, 2, 2)
    x_nhwc = x_nchw.permute(0, 2, 3, 1).contiguous()

    # Block 5
    x_nhwc = do_conv_relu(x_nhwc, p_features_28_weight, p_features_28_bias)
    x_nhwc = do_conv_relu(x_nhwc, p_features_30_weight, p_features_30_bias)
    x_nhwc = do_conv_relu(x_nhwc, p_features_32_weight, p_features_32_bias)
    x_nhwc = do_conv_relu(x_nhwc, p_features_34_weight, p_features_34_bias)
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_nchw = maxpool2d_triton(x_nchw, 2, 2, 2, 2)

    # Flatten
    x_flat = x_nchw.reshape(x_nchw.shape[0], -1).contiguous()  # (10, 25088)

    # Classifier
    x_flat = linear_triton(x_flat, p_classifier_0_weight, p_classifier_0_bias, has_relu=True)
    # dropout p=0.0 is identity
    x_flat = linear_triton(x_flat, p_classifier_3_weight, p_classifier_3_bias, has_relu=True)
    # dropout p=0.0 is identity
    x_flat = linear_triton(x_flat, p_classifier_6_weight, p_classifier_6_bias, has_relu=False)

    return x_flat


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'
    return [
        torch.randn(64, 3, 3, 3, device=device),      # p_features_0_weight
        torch.randn(64, device=device),                 # p_features_0_bias
        torch.randn(64, 64, 3, 3, device=device),      # p_features_2_weight
        torch.randn(64, device=device),                 # p_features_2_bias
        torch.randn(128, 64, 3, 3, device=device),     # p_features_5_weight
        torch.randn(128, device=device),                # p_features_5_bias
        torch.randn(128, 128, 3, 3, device=device),    # p_features_7_weight
        torch.randn(128, device=device),                # p_features_7_bias
        torch.randn(256, 128, 3, 3, device=device),    # p_features_10_weight
        torch.randn(256, device=device),                # p_features_10_bias
        torch.randn(256, 256, 3, 3, device=device),    # p_features_12_weight
        torch.randn(256, device=device),                # p_features_12_bias
        torch.randn(256, 256, 3, 3, device=device),    # p_features_14_weight
        torch.randn(256, device=device),                # p_features_14_bias
        torch.randn(256, 256, 3, 3, device=device),    # p_features_16_weight
        torch.randn(256, device=device),                # p_features_16_bias
        torch.randn(512, 256, 3, 3, device=device),    # p_features_19_weight
        torch.randn(512, device=device),                # p_features_19_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_21_weight
        torch.randn(512, device=device),                # p_features_21_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_23_weight
        torch.randn(512, device=device),                # p_features_23_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_25_weight
        torch.randn(512, device=device),                # p_features_25_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_28_weight
        torch.randn(512, device=device),                # p_features_28_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_30_weight
        torch.randn(512, device=device),                # p_features_30_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_32_weight
        torch.randn(512, device=device),                # p_features_32_bias
        torch.randn(512, 512, 3, 3, device=device),    # p_features_34_weight
        torch.randn(512, device=device),                # p_features_34_bias
        torch.randn(4096, 25088, device=device),        # p_classifier_0_weight
        torch.randn(4096, device=device),               # p_classifier_0_bias
        torch.randn(4096, 4096, device=device),         # p_classifier_3_weight
        torch.randn(4096, device=device),               # p_classifier_3_bias
        torch.randn(1000, 4096, device=device),         # p_classifier_6_weight
        torch.randn(1000, device=device),               # p_classifier_6_bias
        torch.randn(10, 3, 224, 224, device=device),   # x
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_