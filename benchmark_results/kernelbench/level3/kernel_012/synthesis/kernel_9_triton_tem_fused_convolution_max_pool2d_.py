import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Fused Conv2d + Bias + ReLU (Implicit GEMM, NHWC layout internally)
# ─────────────────────────────────────────────────────────────────────────────

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
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # X strides (NHWC)
    s_xn, s_xh, s_xw, s_xc,
    # W strides (Co, Ci, Kh, Kw) -> stored as (Co, Kh, Kw, Ci) for NHWC
    s_wo, s_wkh, s_wkw, s_wc,
    # Y strides (NHWC)
    s_yn, s_yh, s_yw, s_yc,
    DO_RELU: tl.constexpr,
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = IN_C * KERNEL_H * KERNEL_W
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Decode K -> (kh, kw, ic)
        ic = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KERNEL_W
        kh = tmp_k // KERNEL_W

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_x = (mask_m[:, None] & mask_k[None, :] &
                   (ih >= 0) & (ih < IN_H) &
                   (iw >= 0) & (iw < IN_W) &
                   (batch_idx[:, None] < BATCH))

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * s_xn +
                  ic[None, :] * s_xc)
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: stored as (Co, Kh, Kw, Ci) -> index [n, kh, kw, ic]
        valid_w = mask_k[:, None] & mask_n[None, :]
        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_wo +
                  kh[:, None] * s_wkh +
                  kw[:, None] * s_wkw +
                  ic[:, None] * s_wc)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    if DO_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store (NHWC)
    mask_out = mask_m[:, None] & mask_n[None, :]
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * s_yn +
              out_h_idx[:, None] * s_yh +
              out_w_idx[:, None] * s_yw +
              offs_n[None, :] * s_yc)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_bias_relu(x_nhwc, w_ochw, bias, stride=1, padding=1, do_relu=True):
    """
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KERNEL_H, KERNEL_W = w_ochw.shape

    if isinstance(stride, int):
        STRIDE_H = STRIDE_W = stride
    else:
        STRIDE_H, STRIDE_W = stride

    if isinstance(padding, int):
        PAD_H = PAD_W = padding
    else:
        PAD_H, PAD_W = padding

    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    # Rearrange weight to (Co, Kh, Kw, Ci) for NHWC-friendly access
    w_okhwc = w_ochw.permute(0, 2, 3, 1).contiguous()  # (Co, Kh, Kw, Ci)

    y_nhwc = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    M = N * OUT_H * OUT_W
    K = IN_C * KERNEL_H * KERNEL_W
    No = OUT_C

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(No, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        x_nhwc, w_okhwc, bias, y_nhwc,
        M, No, K,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        # X strides (N, H, W, C)
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        # W strides (Co, Kh, Kw, Ci)
        w_okhwc.stride(0), w_okhwc.stride(1), w_okhwc.stride(2), w_okhwc.stride(3),
        # Y strides (N, H, W, C)
        y_nhwc.stride(0), y_nhwc.stride(1), y_nhwc.stride(2), y_nhwc.stride(3),
        DO_RELU=do_relu,
    )
    return y_nhwc


# ─────────────────────────────────────────────────────────────────────────────
# Max Pool 2x2 stride 2 (NHWC)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['total'],
)
@triton.jit
def maxpool2x2_nhwc_kernel(
    X_ptr, Y_ptr,
    N, IN_H, IN_W, C,
    OUT_H, OUT_W,
    s_xn, s_xh, s_xw, s_xc,
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
        valid = mask & (n_idx < N) & (ih < IN_H) & (iw < IN_W)
        ptr = X_ptr + n_idx * s_xn + ih * s_xh + iw * s_xw + c_idx * s_xc
        return tl.load(ptr, mask=valid, other=float('-inf'))

    v00 = load_val(0, 0)
    v01 = load_val(0, 1)
    v10 = load_val(1, 0)
    v11 = load_val(1, 1)

    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    y_ptr = Y_ptr + n_idx * s_yn + oh * s_yh + ow * s_yw + c_idx * s_yc
    tl.store(y_ptr, result, mask=mask)


def maxpool2x2_nhwc(x_nhwc):
    N, IN_H, IN_W, C = x_nhwc.shape
    OUT_H = IN_H // 2
    OUT_W = IN_W // 2
    y_nhwc = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)
    total = N * OUT_H * OUT_W * C

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2x2_nhwc_kernel[grid](
        x_nhwc, y_nhwc,
        N, IN_H, IN_W, C,
        OUT_H, OUT_W,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        y_nhwc.stride(0), y_nhwc.stride(1), y_nhwc.stride(2), y_nhwc.stride(3),
        total,
    )
    return y_nhwc


# ─────────────────────────────────────────────────────────────────────────────
# Fused Linear + Bias + ReLU (GEMM)
# ─────────────────────────────────────────────────────────────────────────────

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
def linear_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # W is (N, K), so W[n, k]
        w_ptrs = W_ptr + offs_n[:, None] * K + offs_k[None, :]
        w_tile = tl.load(w_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    if DO_RELU:
        acc = tl.maximum(acc, 0.0)

    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def linear_bias_relu(x, weight, bias, do_relu=True):
    """x: (M, K), weight: (N, K), bias: (N,) -> (M, N)"""
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_bias_relu_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        DO_RELU=do_relu,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel_function: VGG19 forward pass
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
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # Block 1
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_0_weight, p_features_0_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_2_weight, p_features_2_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = maxpool2x2_nhwc(x_nhwc)

    # Block 2
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_5_weight, p_features_5_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_7_weight, p_features_7_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = maxpool2x2_nhwc(x_nhwc)

    # Block 3
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_10_weight, p_features_10_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_12_weight, p_features_12_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_14_weight, p_features_14_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_16_weight, p_features_16_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = maxpool2x2_nhwc(x_nhwc)

    # Block 4
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_19_weight, p_features_19_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_21_weight, p_features_21_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_23_weight, p_features_23_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_25_weight, p_features_25_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = maxpool2x2_nhwc(x_nhwc)

    # Block 5
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_28_weight, p_features_28_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_30_weight, p_features_30_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_32_weight, p_features_32_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_34_weight, p_features_34_bias, stride=1, padding=1, do_relu=True)
    x_nhwc = maxpool2x2_nhwc(x_nhwc)

    # Flatten: (N, 7, 7, 512) -> (N, 25088)
    N = x_nhwc.shape[0]
    # Convert back to NCHW order for flatten: (N, H, W, C) -> flatten to (N, H*W*C)
    # But VGG flattens in NCHW order: (N, C, H, W) -> (N, C*H*W)
    # x_nhwc is (N, 7, 7, 512), NCHW would be (N, 512, 7, 7)
    # flatten(NCHW) = N, 512*7*7 = 25088
    # flatten(NHWC) = N, 7*7*512 = 25088 (same total, different order!)
    # We need to match PyTorch's flatten which is NCHW
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_flat = x_nchw.reshape(N, -1)  # (N, 25088)

    # Classifier
    x_flat = linear_bias_relu(x_flat, p_classifier_0_weight, p_classifier_0_bias, do_relu=True)
    # dropout p=0.0 is identity
    x_flat = linear_bias_relu(x_flat, p_classifier_3_weight, p_classifier_3_bias, do_relu=True)
    # dropout p=0.0 is identity
    x_flat = linear_bias_relu(x_flat, p_classifier_6_weight, p_classifier_6_bias, do_relu=False)

    return x_flat


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'

    def make_conv_weight(out_c, in_c, kh, kw):
        return torch.randn(out_c, in_c, kh, kw, device=device, dtype=torch.float32) * 0.01

    def make_bias(c):
        return torch.zeros(c, device=device, dtype=torch.float32)

    inputs = [
        make_conv_weight(64, 3, 3, 3), make_bias(64),
        make_conv_weight(64, 64, 3, 3), make_bias(64),
        make_conv_weight(128, 64, 3, 3), make_bias(128),
        make_conv_weight(128, 128, 3, 3), make_bias(128),
        make_conv_weight(256, 128, 3, 3), make_bias(256),
        make_conv_weight(256, 256, 3, 3), make_bias(256),
        make_conv_weight(256, 256, 3, 3), make_bias(256),
        make_conv_weight(256, 256, 3, 3), make_bias(256),
        make_conv_weight(512, 256, 3, 3), make_bias(512),
        make_conv_weight(512, 512, 3, 3), make_bias(512),
        make_conv_weight(512, 512, 3, 3), make_bias(512),
        make_conv_weight(512, 512, 3, 3), make_bias(512),
        make_conv_weight(512, 512, 3, 3), make_bias(512),
        make_conv_weight(512, 512, 3, 3), make_bias(512),
        make_conv_weight(512, 512, 3, 3), make_bias(512),
        make_conv_weight(512, 512, 3, 3), make_bias(512),
        torch.randn(4096, 25088, device=device, dtype=torch.float32) * 0.01,
        make_bias(4096),
        torch.randn(4096, 4096, device=device, dtype=torch.float32) * 0.01,
        make_bias(4096),
        torch.randn(1000, 4096, device=device, dtype=torch.float32) * 0.01,
        make_bias(1000),
        torch.randn(10, 3, 224, 224, device=device, dtype=torch.float32) * 0.1,
    ]
    return inputs


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
        class VGG19(nn.Module):
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
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding