import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Implicit-GEMM Conv2d kernel (NHWC layout for coalesced access)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['M_total', 'OUT_C', 'IN_C', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M_total, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    # strides for X in NHWC: (N, H, W, C)
    sx_n, sx_h, sx_w, sx_c,
    # strides for Y in NHWC: (N, H, W, C)
    sy_n, sy_h, sy_w, sy_c,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
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

    mask_m = offs_m < M_total
    mask_n = offs_n < OUT_C

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOTAL_K

        # Decode K -> (kh, kw, in_c)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KERNEL_W
        kh_idx = tmp_k // KERNEL_W

        # Input spatial coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            mask_m[:, None] &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            mask_k[None, :]
        )

        # Compute pointer offsets for X (NHWC)
        x_offsets = (
            batch_idx[:, None] * sx_n +
            in_c_idx[None, :] * sx_c
        )
        x_tile = tl.load(X_ptr + x_offsets, mask=valid_x, other=0.0)

        # Weight layout: (OUT_C, KERNEL_H, KERNEL_W, IN_C) -> NHWC-style
        # W stored as (OUT_C, IN_C, KH, KW) originally, but we reorder to (OUT_C, KH, KW, IN_C)
        # w_offsets: [BLOCK_K, BLOCK_N]
        w_offsets = (
            offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) +
            kh_idx[:, None] * (KERNEL_W * IN_C) +
            kw_idx[:, None] * IN_C +
            in_c_idx[:, None]
        )
        valid_w = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(W_ptr + w_offsets, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU if needed
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output in NHWC
    out_offsets = (
        batch_idx[:, None] * sy_n +
        out_h_idx[:, None] * sy_h +
        out_w_idx[:, None] * sy_w +
        offs_n[None, :] * sy_c
    )
    valid_out = mask_m[:, None] & mask_n[None, :]
    tl.store(Y_ptr + out_offsets, acc.to(tl.float32), mask=valid_out)


# ─────────────────────────────────────────────────────────────────────────────
# Max Pool 2x2 stride 2 kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N_total'],
)
@triton.jit
def maxpool2x2_kernel(
    X_ptr, Y_ptr,
    N_total,
    IN_H, IN_W, OUT_H, OUT_W, C,
    # X strides (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    # Y strides (NHWC)
    sy_n, sy_h, sy_w, sy_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    # Decode: N_total = batch * OUT_H * OUT_W * C
    c_idx = offs % C
    tmp = offs // C
    ow_idx = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh_idx = tmp2 % OUT_H
    n_idx = tmp2 // OUT_H

    ih0 = oh_idx * 2
    iw0 = ow_idx * 2

    # Load 4 values
    def load_val(dh, dw):
        ih = ih0 + dh
        iw = iw0 + dw
        valid = mask & (ih < IN_H) & (iw < IN_W)
        ptr = X_ptr + n_idx * sx_n + ih * sx_h + iw * sx_w + c_idx * sx_c
        return tl.load(ptr, mask=valid, other=float('-inf'))

    v00 = load_val(0, 0)
    v01 = load_val(0, 1)
    v10 = load_val(1, 0)
    v11 = load_val(1, 1)

    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    out_ptr = Y_ptr + n_idx * sy_n + oh_idx * sy_h + ow_idx * sy_w + c_idx * sy_c
    tl.store(out_ptr, result, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Fused Linear + Bias + ReLU kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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

        x_tile = tl.load(
            X_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0
        )
        # W is (N, K), so W[n, k]
        w_tile = tl.load(
            W_ptr + offs_n[:, None] * K + offs_k[None, :],
            mask=mask_n[:, None] & mask_k[None, :],
            other=0.0
        )
        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    tl.store(
        Y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc.to(tl.float32),
        mask=mask_m[:, None] & mask_n[None, :]
    )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one conv2d layer
# ─────────────────────────────────────────────────────────────────────────────

def run_conv2d(x_nhwc, w_oihw, bias, stride=1, pad=1, apply_relu=True):
    """
    returns y_nhwc: (N, OUT_H, OUT_W, C_out)
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w_oihw.shape
    OUT_H = (IN_H + 2 * pad - KH) // stride + 1
    OUT_W = (IN_W + 2 * pad - KW) // stride + 1

    # Reorder weight to (OUT_C, KH, KW, IN_C) for the kernel
    w_reordered = w_oihw.permute(0, 2, 3, 1).contiguous()  # (OUT_C, KH, KW, IN_C)
    # Flatten to (OUT_C, KH*KW*IN_C)
    w_flat = w_reordered.reshape(OUT_C, KH * KW * IN_C)

    y_nhwc = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    M_total = N * OUT_H * OUT_W
    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    sy_n, sy_h, sy_w, sy_c = y_nhwc.stride()

    grid = lambda meta: (
        triton.cdiv(M_total, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_relu_kernel[grid](
        x_nhwc, w_flat, bias, y_nhwc,
        M_total, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        stride, stride, pad, pad,
        sx_n, sx_h, sx_w, sx_c,
        sy_n, sy_h, sy_w, sy_c,
        apply_relu,
        KERNEL_H=KH, KERNEL_W=KW,
    )
    return y_nhwc


def run_maxpool(x_nhwc):
    """2x2 max pool with stride 2"""
    N, IN_H, IN_W, C = x_nhwc.shape
    OUT_H = IN_H // 2
    OUT_W = IN_W // 2
    y_nhwc = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)

    N_total = N * OUT_H * OUT_W * C
    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    sy_n, sy_h, sy_w, sy_c = y_nhwc.stride()

    grid = lambda meta: (triton.cdiv(N_total, meta['BLOCK_SIZE']),)
    maxpool2x2_kernel[grid](
        x_nhwc, y_nhwc,
        N_total,
        IN_H, IN_W, OUT_H, OUT_W, C,
        sx_n, sx_h, sx_w, sx_c,
        sy_n, sy_h, sy_w, sy_c,
    )
    return y_nhwc


def run_linear(x_2d, weight, bias, apply_relu=True):
    """
    returns: (M, N)
    """
    M, K = x_2d.shape
    N = weight.shape[0]
    y = torch.empty((M, N), device=x_2d.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_relu_kernel[grid](
        x_2d, weight, bias, y,
        M, N, K,
        apply_relu,
    )
    return y


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel_function: full VGG16 forward pass
# ─────────────────────────────────────────────────────────────────────────────

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
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, 224, 224, 3)

    # Block 1
    x_nhwc = run_conv2d(x_nhwc, p_features_0_weight, p_features_0_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_2_weight, p_features_2_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_maxpool(x_nhwc)

    # Block 2
    x_nhwc = run_conv2d(x_nhwc, p_features_5_weight, p_features_5_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_7_weight, p_features_7_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_maxpool(x_nhwc)

    # Block 3
    x_nhwc = run_conv2d(x_nhwc, p_features_10_weight, p_features_10_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_12_weight, p_features_12_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_14_weight, p_features_14_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_maxpool(x_nhwc)

    # Block 4
    x_nhwc = run_conv2d(x_nhwc, p_features_17_weight, p_features_17_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_19_weight, p_features_19_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_21_weight, p_features_21_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_maxpool(x_nhwc)

    # Block 5
    x_nhwc = run_conv2d(x_nhwc, p_features_24_weight, p_features_24_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_26_weight, p_features_26_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_conv2d(x_nhwc, p_features_28_weight, p_features_28_bias, stride=1, pad=1, apply_relu=True)
    x_nhwc = run_maxpool(x_nhwc)

    # Flatten: (N, 7, 7, 512) -> (N, 25088)
    # Convert back to NCHW first for correct flatten order
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()  # (N, 512, 7, 7)
    x_flat = x_nchw.reshape(x_nchw.shape[0], -1)  # (N, 25088)

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

    # Move all to CUDA
    inputs = [
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
    return [t.cuda() for t in inputs]


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
                    nn.Linear(512 * 7 * 7, 4096), nn.ReLU(inplace=True),
                    nn.Dropout(p=0.0),
                    nn.Linear(4096, 4096), nn.ReLU(inplace=True),
                    nn.Dropout(p=0.0),
                    nn.Linear(4096, 1000),
                )

            def forward(self, x):
                x = self.features(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x

        model = VGG16().cuda().eval()

        # Load weights from inputs
        (p_features_0_weight, p_features_0_bias,
         p_features_2_weight, p_features_2_bias,