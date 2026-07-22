import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# Due to the massive complexity of implementing all VGG16 conv layers in pure Triton,
# I'll implement the key computation kernels: conv2d+relu fusion and linear+relu fusion

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)


import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    APPLY_RELU: tl.constexpr,
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
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc, allow_tf32=False)

    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    y_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(y_ptrs, acc.to(tl.float32), mask=y_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'OUT_H', 'OUT_W', 'IN_C', 'OUT_C', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # Input strides (NCHW)
    stride_xn, stride_xc, stride_xh, stride_xw,
    # Weight strides (NCHW: out_c, in_c, kh, kw)
    stride_woc, stride_wic, stride_wkh, stride_wkw,
    # Output strides (NCHW)
    stride_yn, stride_yc, stride_yh, stride_yw,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KERNEL_H * KERNEL_W
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

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        kw_idx = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh_idx = tmp_k % KERNEL_H
        ic_idx = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m < (BATCH * OUT_H * OUT_W)
        valid_k = offs_k < TOTAL_K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        x_mask = valid_m[:, None] & valid_k[None, :] & valid_spatial

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * stride_xn +
                  ic_idx[None, :] * stride_xc +
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Weight: [BLOCK_K, BLOCK_N] -> w[out_c, in_c, kh, kw]
        w_ptrs = (W_ptr +
                  offs_n[None, :] * stride_woc +
                  ic_idx[:, None] * stride_wic +
                  kh_idx[:, None] * stride_wkh +
                  kw_idx[:, None] * stride_wkw)
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + b[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store output: Y[batch, out_c, out_h, out_w]
    valid_m = offs_m < (BATCH * OUT_H * OUT_W)
    valid_n = offs_n < OUT_C
    out_mask = valid_m[:, None] & valid_n[None, :]

    y_ptrs = (Y_ptr +
              batch_idx[:, None] * stride_yn +
              offs_n[None, :] * stride_yc +
              out_h_idx[:, None] * stride_yh +
              out_w_idx[:, None] * stride_yw)
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['N'],
)
@triton.jit
def max_pool2d_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N

    # Decode: offs -> (batch, c, out_h, out_w)
    out_w_idx = offs % OUT_W
    tmp = offs // OUT_W
    out_h_idx = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c_idx = tmp2 % C
    batch_idx = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            ih = out_h_idx * STRIDE_H + kh
            iw = out_w_idx * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_ptrs = (X_ptr +
            x_val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, x_val)

    y_ptrs = (Y_ptr +
    tl.store(y_ptrs, max_val, mask=mask)


def triton_conv2d_relu(x, weight, bias, stride=1, padding=1, apply_relu=True):
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    stride_h = stride if isinstance(stride, int) else stride[0]
    stride_w = stride if isinstance(stride, int) else stride[1]
    pad_h = padding if isinstance(padding, int) else padding[0]
    pad_w = padding if isinstance(padding, int) else padding[1]

    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1

    y = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OH * OW
    BLOCK_M_val = 256
    BLOCK_N_val = 64

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    conv2d_relu_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W,
        C_out, OH, OW,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        apply_relu,
    )
    return y


def triton_max_pool2d(x, kernel_size=2, stride=2):
    N, C, H, W = x.shape
    KH = kernel_size if isinstance(kernel_size, int) else kernel_size[0]
    KW = kernel_size if isinstance(kernel_size, int) else kernel_size[1]
    SH = stride if isinstance(stride, int) else stride[0]
    SW = stride if isinstance(stride, int) else stride[1]

    OH = (H - KH) // SH + 1
    OW = (W - KW) // SW + 1

    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    max_pool2d_kernel[grid](
        x, y,
        N, C, H, W, OH, OW,
        KH, KW, SH, SW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        total,
    )
    return y


def triton_linear_relu(x, weight, bias, apply_relu=True):
    M, K = x.shape
    N = weight.shape[0]

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_relu_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        apply_relu,
    )
    return y


def kernel_function(
    p_features_0_weight, p_features_0_bias,
    p_features_2_weight, p_features_2_bias,
    p_features_5_weight, p_features_5_bias,
    p_features_7_weight, p_features_7_bias,
    p_features_10_weight, p_features_10_bias,
    p_features_12_weight, p_features_12_bias,
    p_features_14_weight, p_features_14_bias,
    p_features_17_weight, p_features_17_bias,
    p_features_19_weight, p_features_19_bias,
    p_features_21_weight, p_features_21_bias,
    p_features_24_weight, p_features_24_bias,
    p_features_26_weight, p_features_26_bias,
    p_features_28_weight, p_features_28_bias,
    p_classifier_0_weight, p_classifier_0_bias,
    p_classifier_3_weight, p_classifier_3_bias,
    p_classifier_6_weight, p_classifier_6_bias,
    x,
):
    # Block 1
    x = triton_conv2d_relu(x, p_features_0_weight, p_features_0_bias, stride=1, padding=1, apply_relu=True)
    x = triton_conv2d_relu(x, p_features_2_weight, p_features_2_bias, stride=1, padding=1, apply_relu=True)
    x = triton_max_pool2d(x, kernel_size=2, stride=2)

    # Block 2
    x = triton_conv2d_relu(x, p_features_5_weight, p_features_5_bias, stride=1, padding=1, apply_relu=True)
    x = triton_conv2d_relu(x, p_features_7_weight, p_features_7_bias, stride=1, padding=1, apply_relu=True)
    x = triton_max_pool2d(x, kernel_size=2, stride=2)

    # Block 3
    x = triton_conv2d_relu(x, p_features_10_weight, p_features_10_bias, stride=1, padding=1, apply_relu=True)
    x = triton_conv2d_relu(x, p_features_12_weight, p_features_12_bias, stride=1, padding=1, apply_relu=True)
    x = triton_conv2d_relu(x, p_features_14_weight, p_features_14_bias, stride=1, padding=1, apply_relu=True)
    x = triton_max_pool2d(x, kernel_size=2, stride=2)

    # Block 4
    x = triton_conv2d_relu(x, p_features_17_weight, p_features_17_bias, stride=1, padding=1, apply_relu=True)
    x = triton_conv2d_relu(x, p_features_19_weight, p_features_19_bias, stride=1, padding=1, apply_relu=True)
    x = triton_conv2d_relu(x, p_features_21_weight, p_features_21_bias, stride=1, padding=1, apply_relu=True)
    x = triton_max_pool2d(x, kernel_size=2, stride=2)

    # Block 5
    x = triton_conv2d_relu(x, p_features_24_weight, p_features_24_bias, stride=1, padding=1, apply_relu=True)
    x = triton_conv2d_relu(x, p_features_26_weight, p_features_26_bias, stride=1, padding=1, apply_relu=True)
    x = triton_conv2d_relu(x, p_features_28_weight, p_features_28_bias, stride=1, padding=1, apply_relu=True)
    x = triton_max_pool2d(x, kernel_size=2, stride=2)

    # Flatten
    x = x.reshape(x.shape[0], -1)

    # Classifier
    x = triton_linear_relu(x, p_classifier_0_weight, p_classifier_0_bias, apply_relu=True)
    x = triton_linear_relu(x, p_classifier_3_weight, p_classifier_3_bias, apply_relu=True)
    x = triton_linear_relu(x, p_classifier_6_weight, p_classifier_6_bias, apply_relu=False)

    return x


def get_inputs():
    p_features_0_weight = torch.randn(64, 3, 3, 3, dtype=torch.float32, device='cuda')
    p_features_0_bias = torch.randn(64, dtype=torch.float32, device='cuda')
    p_features_2_weight = torch.randn(64, 64, 3, 3, dtype=torch.float32, device='cuda')
    p_features_2_bias = torch.randn(64, dtype=torch.float32, device='cuda')
    p_features_5_weight = torch.randn(128, 64, 3, 3, dtype=torch.float32, device='cuda')
    p_features_5_bias = torch.randn(128, dtype=torch.float32, device='cuda')
    p_features_7_weight = torch.randn(128, 128, 3, 3, dtype=torch.float32, device='cuda')
    p_features_7_bias = torch.randn(128, dtype=torch.float32, device='cuda')
    p_features_10_weight = torch.randn(256, 128, 3, 3, dtype=torch.float32, device='cuda')
    p_features_10_bias = torch.randn(256, dtype=torch.float32, device='cuda')
    p_features_12_weight = torch.randn(256, 256, 3, 3, dtype=torch.float32, device='cuda')
    p_features_12_bias = torch.randn(256, dtype=torch.float32, device='cuda')
    p_features_14_weight = torch.randn(256, 256, 3, 3, dtype=torch.float32, device='cuda')
    p_features_14_bias = torch.randn(256, dtype=torch.float32, device='cuda')
    p_features_17_weight = torch.randn(512, 256, 3, 3, dtype=torch.float32, device='cuda')
    p_features_17_bias = torch.randn(512, dtype=torch.float32, device='cuda')
    p_features_19_weight = torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda')
    p_features_19_bias = torch.randn(512, dtype=torch.float32, device='cuda')
    p_features_21_weight = torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda')
    p_features_21_bias = torch.randn(512, dtype=torch.float32, device='cuda')
    p_features_24_weight = torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda')
    p_features_24_bias = torch.randn(512, dtype=torch.float32, device='cuda')
    p_features_26_weight = torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda')
    p_features_26_bias = torch.randn(512, dtype=torch.float32, device='cuda')
    p_features_28_weight = torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda')
    p_features_28_bias = torch.randn(512, dtype=torch.float32, device='cuda')
    p_classifier_0_weight = torch.randn(4096, 25088, dtype=torch.float32, device='cuda')
    p_classifier_0_bias = torch.randn(4096, dtype=torch.float32, device='cuda')
    p_classifier_3_weight = torch.randn(4096, 4096, dtype=torch.float32, device='cuda')
    p_classifier_3_bias = torch.randn(4096, dtype=torch.float32, device='cuda')
    p_classifier_6_weight = torch.randn(1000, 4096, dtype=torch.float32, device='cuda')
    p_classifier_6_bias = torch.randn(1000, dtype=torch.float32, device='cuda')
    x = torch.randn(10, 3, 224, 224, dtype=torch.float32, device='cuda')
    return [
        p_features_0_weight, p_features_0_bias,
        p_features_2_weight, p_features_2_bias,
        p_features_5_weight, p_features_5_bias,
        p_features_7_weight, p_features_7_bias,
        p_features_10_weight, p_features_10_bias,
        p_features_12_weight, p_features_12_bias,
        p_features_14_weight, p_features_14_bias,
        p_features_17_weight, p_features_17_bias,
        p_features_19_weight, p_features_19_bias,
        p_features_21_weight, p_features_21_bias,
        p_features_24_weight, p_features_24_bias,
        p_features_26_weight, p_features_26_bias,
        p_features_28_weight, p_features_28_bias,
        p_classifier_0_weight, p_classifier_0_bias,
        p_classifier_3_weight, p_classifier_3_bias,
        p_classifier_6_weight, p_classifier_6_bias,
        x,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        # Build reference VGG16
        class VGG16Ref(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(inplace=True),