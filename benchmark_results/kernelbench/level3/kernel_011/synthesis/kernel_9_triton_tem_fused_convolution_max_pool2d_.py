import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math

# ============================================================
# Conv2d kernel (implicit GEMM, NHWC layout internally)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N_out, K,
    # strides for X (NHWC)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (OUT_C, IN_C, KH, KW) -> stored as (OUT_C, KH, KW, IN_C) for NHWC
    s_wo, s_wkh, s_wkw, s_wc,
    # strides for Y (NHWC)
    s_yn, s_yh, s_yw, s_yc,
    APPLY_RELU: tl.constexpr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = IN_C * KH * KW
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m < M
        valid_k = offs_k < K
        valid = (
            valid_m[:, None] &
            valid_k[None, :] &
            (batch_idx[:, None] < N) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W)
        )

        x_ptrs = X_ptr + (
            batch_idx[:, None] * s_xn +
            in_c[None, :] * s_xc
        )
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # W shape: (OUT_C, KH, KW, IN_C) in memory
        # offs_k encodes (kh, kw, in_c)
        w_ptrs = W_ptr + (
            offs_n[None, :] * s_wo +
            kh[:, None] * s_wkh +
            kw[:, None] * s_wkw +
            in_c[:, None] * s_wc
        )
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store Y (NHWC)
    valid_m = offs_m < M
    valid_n = offs_n < OUT_C
    mask_out = valid_m[:, None] & valid_n[None, :]

    y_ptrs = Y_ptr + (
        batch_idx[:, None] * s_yn +
        out_h_idx[:, None] * s_yh +
        out_w_idx[:, None] * s_yw +
        offs_n[None, :] * s_yc
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ============================================================
# Max Pool2d kernel (2x2, stride 2)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def maxpool2x2_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    # X strides (NHWC)
    s_xn, s_xh, s_xw, s_xc,
    # Y strides (NHWC)
    s_yn, s_yh, s_yw, s_yc,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # Decode: total = N * OUT_H * OUT_W * C
    c_idx = offs % C
    tmp = offs // C
    ow = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh = tmp2 % OUT_H
    n_idx = tmp2 // OUT_H

    ih0 = oh * 2
    iw0 = ow * 2

    # Load 4 values
    def load_val(dh, dw):
        ih = ih0 + dh
        iw = iw0 + dw
        valid = mask & (ih < IN_H) & (iw < IN_W) & (n_idx < N)
        ptr = X_ptr + n_idx * s_xn + ih * s_xh + iw * s_xw + c_idx * s_xc
        return tl.load(ptr, mask=valid, other=float('-inf'))

    v00 = load_val(0, 0)
    v01 = load_val(0, 1)
    v10 = load_val(1, 0)
    v11 = load_val(1, 1)

    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    y_ptr = Y_ptr + n_idx * s_yn + oh * s_yh + ow * s_yw + c_idx * s_yc
    tl.store(y_ptr, result, mask=mask)


# ============================================================
# Linear + bias + optional relu kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    APPLY_RELU: tl.constexpr,
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
        # X: (M, K), row-major
        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # W: (N, K), row-major -> W^T is (K, N)
        w_ptrs = W_ptr + offs_n[None, :] * K + offs_k[:, None]
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def run_conv2d(x_nhwc, w_oihw, bias, stride=1, padding=1, apply_relu=True):
    """Run conv2d on NHWC input, weight in OIHW format."""
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w_oihw.shape
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1

    # Convert weight to NHWC-friendly: (OUT_C, KH, KW, IN_C)
    w_conv = w_oihw.permute(0, 2, 3, 1).contiguous()  # (OUT_C, KH, KW, IN_C)

    y_nhwc = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_relu_kernel[grid](
        x_nhwc, w_conv, bias, y_nhwc,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride, stride,
        padding, padding,
        M, OUT_C, K,
        # X strides (NHWC): n, h, w, c
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        # W strides (OUT_C, KH, KW, IN_C)
        w_conv.stride(0), w_conv.stride(1), w_conv.stride(2), w_conv.stride(3),
        # Y strides (NHWC)
        y_nhwc.stride(0), y_nhwc.stride(1), y_nhwc.stride(2), y_nhwc.stride(3),
        APPLY_RELU=apply_relu,
    )
    return y_nhwc


def run_maxpool(x_nhwc):
    N, IN_H, IN_W, C = x_nhwc.shape
    OUT_H = IN_H // 2
    OUT_W = IN_W // 2
    y_nhwc = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)
    total = N * OUT_H * OUT_W * C

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2x2_kernel[grid](
        x_nhwc, y_nhwc,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        y_nhwc.stride(0), y_nhwc.stride(1), y_nhwc.stride(2), y_nhwc.stride(3),
        total,
    )
    return y_nhwc


def run_linear(x, w, b, apply_relu=False):
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_relu_kernel[grid](
        x, w, b, y,
        M, N, K,
        APPLY_RELU=apply_relu,
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
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # Block 1
    x_nhwc = run_conv2d(x_nhwc, p_features_0_weight, p_features_0_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_2_weight, p_features_2_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_maxpool(x_nhwc)

    # Block 2
    x_nhwc = run_conv2d(x_nhwc, p_features_5_weight, p_features_5_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_7_weight, p_features_7_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_maxpool(x_nhwc)

    # Block 3
    x_nhwc = run_conv2d(x_nhwc, p_features_10_weight, p_features_10_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_12_weight, p_features_12_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_14_weight, p_features_14_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_maxpool(x_nhwc)

    # Block 4
    x_nhwc = run_conv2d(x_nhwc, p_features_17_weight, p_features_17_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_19_weight, p_features_19_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_21_weight, p_features_21_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_maxpool(x_nhwc)

    # Block 5
    x_nhwc = run_conv2d(x_nhwc, p_features_24_weight, p_features_24_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_26_weight, p_features_26_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_28_weight, p_features_28_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_maxpool(x_nhwc)

    # Flatten: (N, 7, 7, 512) -> (N, 25088)
    N = x_nhwc.shape[0]
    # Need to convert back to NCHW order for flatten to match PyTorch
    # PyTorch flatten is on NCHW: (N, C, H, W) -> (N, C*H*W)
    # NHWC: (N, H, W, C) -> flatten in NCHW order means we need (N, C, H, W) first
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_flat = x_nchw.reshape(N, -1)

    # Classifier
    x_flat = run_linear(x_flat, p_classifier_0_weight, p_classifier_0_bias, apply_relu=True)
    # dropout p=0.0 is identity
    x_flat = run_linear(x_flat, p_classifier_3_weight, p_classifier_3_bias, apply_relu=True)
    # dropout p=0.0 is identity
    x_flat = run_linear(x_flat, p_classifier_6_weight, p_classifier_6_bias, apply_relu=False)

    return x_flat


def get_inputs():
    torch.manual_seed(42)
    p_features_0_weight = torch.randn(64, 3, 3, 3)
    p_features_0_bias = torch.randn(64)
    p_features_2_weight = torch.randn(64, 64, 3, 3)
    p_features_2_bias = torch.randn(64)
    p_features_5_weight = torch.randn(128, 64, 3, 3)
    p_features_5_bias = torch.randn(128)
    p_features_7_weight = torch.randn(128, 128, 3, 3)
    p_features_7_bias = torch.randn(128)
    p_features_10_weight = torch.randn(256, 128, 3, 3)
    p_features_10_bias = torch.randn(256)
    p_features_12_weight = torch.randn(256, 256, 3, 3)
    p_features_12_bias = torch.randn(256)
    p_features_14_weight = torch.randn(256, 256, 3, 3)
    p_features_14_bias = torch.randn(256)
    p_features_17_weight = torch.randn(512, 256, 3, 3)
    p_features_17_bias = torch.randn(512)
    p_features_19_weight = torch.randn(512, 512, 3, 3)
    p_features_19_bias = torch.randn(512)
    p_features_21_weight = torch.randn(512, 512, 3, 3)
    p_features_21_bias = torch.randn(512)
    p_features_24_weight = torch.randn(512, 512, 3, 3)
    p_features_24_bias = torch.randn(512)
    p_features_26_weight = torch.randn(512, 512, 3, 3)
    p_features_26_bias = torch.randn(512)
    p_features_28_weight = torch.randn(512, 512, 3, 3)
    p_features_28_bias = torch.randn(512)
    p_classifier_0_weight = torch.randn(4096, 25088)
    p_classifier_0_bias = torch.randn(4096)
    p_classifier_3_weight = torch.randn(4096, 4096)
    p_classifier_3_bias = torch.randn(4096)
    p_classifier_6_weight = torch.randn(1000, 4096)
    p_classifier_6_bias = torch.randn(1000)
    x = torch.randn(10, 3, 224, 224)

    device = 'cuda'
    return [
        p_features_0_weight.to(device), p_features_0_bias.to(device),
        p_features_2_weight.to(device), p_features_2_bias.to(device),
        p_features_5_weight.to(device), p_features_5_bias.to(device),
        p_features_7_weight.to(device), p_features_7_bias.to(device),
        p_features_10_weight.to(device), p_features_10_bias.to(device),
        p_features_12_weight.to(device), p_features_12_bias.to(device),
        p_features_14_weight.to(device), p_features_14_bias.to(device),
        p_features_17_weight.to(device), p_features_17_bias.to(device),
        p_features_19_weight.to(device), p_features_19_bias.to(device),
        p_features_21_weight.to(device), p_features_21_bias.to(device),
        p_features_24_weight.to(device), p_features_24_bias.to(device),
        p_features_26_weight.to(device), p_features_26_bias.to(device),
        p_features_28_weight.to(device), p_features_28_bias.to(device),
        p_classifier_0_weight.to(device), p_classifier_0_bias.to(device),
        p_classifier_3_weight.to(device), p_classifier_3_bias.to(device),
        p_classifier_6_weight.to(device), p_classifier_6_bias.to(device),
        x.to(device),
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

        # Build reference model
        class VGG16(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(512*7*7, 4096), nn.ReLU(inplace=True),
                    nn.Dropout(p=0.0),
                    nn.Linear(4096, 4096), nn.ReLU(inplace=True),
                    nn.Dropout(p=0.0),
                    nn.Linear(4096, 1000),
                )

        model = VGG16().cuda().eval()

        # Load weights
        (p_features_0_weight, p_features_0_bias,
         p_features_2_weight, p_features_2_bias,
         p_features_5_weight, p_features_5_bias,
         p_features_7_weight, p_features_7_bias,
         p_features_10_weight, p_features_10_bias,
         p_features_12_weight, p_features_12_bias,
         p_features_14_weight, p_features_14_bias,
         p_features_17_weight, p_features_17_bias,