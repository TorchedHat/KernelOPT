import torch
import triton
import triton.language as tl

# ============================================================
# Fused Conv2d + Bias + ReLU kernel (implicit GEMM)
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
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KH', 'KW'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    APPLY_RELU: tl.constexpr,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (out_c, kh, kw, in_c) layout
    s_wo, s_wh, s_ww, s_wi,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
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

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KW
        kh_idx = offs_k // (IN_C * KW)

        ih = out_h[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_h & valid_w

        x_ptrs = X_ptr + batch[:, None] * s_xn + ih * s_xh + iw * s_xw + in_c[None, :] * s_xc
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n
        w_ptrs = W_ptr + offs_n[None, :] * s_wo + kh_idx[:, None] * s_wh + kw_idx[:, None] * s_ww + in_c[:, None] * s_wi
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output in NHWC layout
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    # Y is NHWC: [BATCH, OUT_H, OUT_W, OUT_C]
    y_ptrs = Y_ptr + batch[:, None] * (OUT_H * OUT_W * OUT_C) + out_h[:, None] * (OUT_W * OUT_C) + out_w[:, None] * OUT_C + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ============================================================
# MaxPool2d kernel (2x2, stride 2) - operates on NHWC
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def maxpool2x2_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: NHWC layout
    c_idx = offs % C
    tmp = offs // C
    ow = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh = tmp2 % OUT_H
    n = tmp2 // OUT_H

    ih0 = oh * 2
    iw0 = ow * 2

    base = n * (IN_H * IN_W * C) + c_idx

    v00 = tl.load(X_ptr + base + ih0 * (IN_W * C) + iw0 * C, mask=mask, other=float('-inf'))
    v01 = tl.load(X_ptr + base + ih0 * (IN_W * C) + (iw0 + 1) * C, mask=mask, other=float('-inf'))
    v10 = tl.load(X_ptr + base + (ih0 + 1) * (IN_W * C) + iw0 * C, mask=mask, other=float('-inf'))
    v11 = tl.load(X_ptr + base + (ih0 + 1) * (IN_W * C) + (iw0 + 1) * C, mask=mask, other=float('-inf'))

    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    y_ptrs = Y_ptr + n * (OUT_H * OUT_W * C) + oh * (OUT_W * C) + ow * C + c_idx
    tl.store(y_ptrs, result, mask=mask)


# ============================================================
# Linear + optional ReLU kernel (using tl.dot)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 128}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
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

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_k[:, None] * N + offs_n[None, :], mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=mask_out)


def run_conv2d_relu(x_nhwc, weight, bias, stride=1, padding=1, apply_relu=True):
    """
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = weight.shape

    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1

    # Rearrange weight to [out_c, kh, kw, in_c] for easier indexing
    w_reordered = weight.permute(0, 2, 3, 1).contiguous()  # [out_c, kh, kw, in_c]

    y = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    M_TOTAL = N * OUT_H * OUT_W
    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_relu_kernel[grid](
        x_nhwc, w_reordered, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride, stride,
        padding, padding,
        apply_relu,
        # strides for x_nhwc: [N, H, W, C]
        IN_H * IN_W * IN_C, IN_W * IN_C, IN_C, 1,
        # strides for w_reordered: [out_c, kh, kw, in_c]
        KH * KW * IN_C, KW * IN_C, IN_C, 1,
    )
    return y


def run_maxpool2x2(x_nhwc):
    N, IN_H, IN_W, C = x_nhwc.shape
    OUT_H = IN_H // 2
    OUT_W = IN_W // 2
    y = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)
    N_TOTAL = N * OUT_H * OUT_W * C
    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    maxpool2x2_kernel[grid](
        x_nhwc, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        N_TOTAL,
    )
    return y


def run_linear_relu(x, weight, bias, apply_relu=True):
    """
    """
    M, K = x.shape
    N = weight.shape[0]

    # We need W in [K, N] for the matmul
    w_t = weight.t().contiguous()  # [K, N]

    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_relu_kernel[grid](
        x, w_t, bias, y,
        M, N, K,
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
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_0_weight, p_features_0_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_2_weight, p_features_2_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_maxpool2x2(x_nhwc)

    # Block 2
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_5_weight, p_features_5_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_7_weight, p_features_7_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_maxpool2x2(x_nhwc)

    # Block 3
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_10_weight, p_features_10_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_12_weight, p_features_12_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_14_weight, p_features_14_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_16_weight, p_features_16_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_maxpool2x2(x_nhwc)

    # Block 4
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_19_weight, p_features_19_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_21_weight, p_features_21_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_23_weight, p_features_23_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_25_weight, p_features_25_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_maxpool2x2(x_nhwc)

    # Block 5
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_28_weight, p_features_28_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_30_weight, p_features_30_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_32_weight, p_features_32_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_34_weight, p_features_34_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = run_maxpool2x2(x_nhwc)

    # Flatten: NHWC -> [N, H*W*C]
    N = x_nhwc.shape[0]
    x_flat = x_nhwc.reshape(N, -1).contiguous()

    # Classifier
    x_flat = run_linear_relu(x_flat, p_classifier_0_weight, p_classifier_0_bias, apply_relu=True)
    # dropout p=0.0 is identity
    x_flat = run_linear_relu(x_flat, p_classifier_3_weight, p_classifier_3_bias, apply_relu=True)
    # dropout p=0.0 is identity
    x_flat = run_linear_relu(x_flat, p_classifier_6_weight, p_classifier_6_bias, apply_relu=False)

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
        torch.randn(10, 3, 224, 224, device=device),    # x
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
        import torch.nn.functional as F

        # Reference implementation
        (p_features_0_weight, p_features_0_bias,
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
         x) = inputs

        def ref_conv_relu(x, w, b, padding=1):
            return F.relu(F.conv2d(x, w, b, padding=padding))

        def ref_linear_relu(x, w, b):
            return F.relu(F.linear(x, w, b))

        xr = x
        xr = ref_conv_relu(xr, p_features_0_weight, p_features_0_bias)
        xr = ref_conv_relu(xr, p_features_2_weight, p_features_2_bias)
        xr = F.max_pool2d(xr, 2, 2)
        xr = ref_conv_relu(xr, p_features_5_weight, p_features_5_bias)
        xr = ref_conv_relu(xr, p_features_7_weight, p_features_7_bias)
        xr = F.max_pool2d(xr, 2, 2)
        xr = ref_conv_relu(xr, p_features_10_weight, p_features_10_bias)
        xr = ref_conv_relu(xr, p_features_12_weight, p_features_12_bias)
        xr = ref_conv_relu(xr, p_features_14_weight, p_features_14_bias)
        xr = ref_conv_relu(xr, p_features_16_weight, p_features_16_bias)
        xr = F.max_pool2d(xr, 2, 2)
        xr = ref_conv_relu(xr, p_features_19_weight, p_features_19_bias)
        xr = ref_conv_relu(xr, p_features_21_weight, p_features_21_bias)
        xr = ref_conv_relu(xr, p_features_23_weight, p_features_23_bias)
        xr = ref_conv_relu(xr, p_features_25_weight, p_features_25_bias)
        xr = F.max_pool2d(xr, 2, 2)
        xr = ref_conv_relu(xr, p_features_28_weight, p_features_28_bias)
        xr = ref_conv_relu(xr, p_features_30_weight, p_features_30_bias)
        xr = ref_conv_relu(xr, p_features_32_weight, p_features_32_bias)
        xr = ref_conv_relu(xr, p_features_34_weight, p_features_34_bias)
        xr = F.max_pool2d(xr, 2, 2)
        xr = torch.flatten(xr,