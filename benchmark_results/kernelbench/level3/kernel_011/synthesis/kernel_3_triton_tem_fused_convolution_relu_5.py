import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton Conv2d kernel (implicit GEMM, NHWC layout internally)
# Fuses bias add + optional ReLU
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
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides for X (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (Co, Ci, Kh, Kw) -> stored as (Co, Kh, Kw, Ci) for NHWC
    sw_o, sw_h, sw_w, sw_i,
    # strides for Y (NHWC)
    sy_n, sy_h, sy_w, sy_c,
    USE_RELU: tl.constexpr,
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

        # Decode K -> (kh, kw, in_c)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KERNEL_W
        kh_idx = tmp_k // KERNEL_W

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (mask_m[:, None] & mask_k[None, :]
                   & (ih >= 0) & (ih < IN_H)
                   & (iw >= 0) & (iw < IN_W)

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: shape (OUT_C, KERNEL_H, KERNEL_W, IN_C) in NHWC-friendly layout
        valid_w = mask_k[:, None] & mask_n[None, :]
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    if USE_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output (NHWC)
    mask_out = mask_m[:, None] & mask_n[None, :]
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_conv2d(x_nhwc, w_nhwc, bias, stride_h, stride_w, pad_h, pad_w, use_relu=True):
    """
    returns: (N, OH, OW, Co) contiguous
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, KERNEL_H, KERNEL_W, _ = w_nhwc.shape
    OUT_H = (IN_H + 2 * pad_h - KERNEL_H) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KERNEL_W) // stride_w + 1

    M = N * OUT_H * OUT_W
    K = IN_C * KERNEL_H * KERNEL_W
    Nout = OUT_C

    y = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    # strides for x (NHWC)
    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    # strides for w (Co, Kh, Kw, Ci)
    sw_o, sw_h, sw_w, sw_i = w_nhwc.stride()
    # strides for y (NHWC)
    sy_n, sy_h, sy_w, sy_c = y.stride()

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Nout, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        x_nhwc, w_nhwc, bias, y,
        M, Nout, K,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        stride_h, stride_w,
        pad_h, pad_w,
        sx_n, sx_h, sx_w, sx_c,
        sw_o, sw_h, sw_w, sw_i,
        sy_n, sy_h, sy_w, sy_c,
        USE_RELU=use_relu,
    )
    return y


# ============================================================
# Max Pool2d kernel (2x2, stride 2)
# ============================================================

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
def maxpool2x2_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    # strides for X (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    # strides for Y (NHWC)
    sy_n, sy_h, sy_w, sy_c,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # Decode: offs -> (n, oh, ow, c)
    c_idx = offs % C
    tmp = offs // C
    ow_idx = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh_idx = tmp2 % OUT_H
    n_idx = tmp2 // OUT_H

    ih0 = oh_idx * 2
    iw0 = ow_idx * 2

    x00 = tl.load(X_ptr + n_idx * sx_n + ih0 * sx_h + iw0 * sx_w + c_idx * sx_c, mask=mask, other=-1e9)
    x01 = tl.load(X_ptr + n_idx * sx_n + ih0 * sx_h + (iw0 + 1) * sx_w + c_idx * sx_c, mask=mask & (iw0 + 1 < IN_W), other=-1e9)
    x10 = tl.load(X_ptr + n_idx * sx_n + (ih0 + 1) * sx_h + iw0 * sx_w + c_idx * sx_c, mask=mask & (ih0 + 1 < IN_H), other=-1e9)
    x11 = tl.load(X_ptr + n_idx * sx_n + (ih0 + 1) * sx_h + (iw0 + 1) * sx_w + c_idx * sx_c, mask=mask & (ih0 + 1 < IN_H) & (iw0 + 1 < IN_W), other=-1e9)

    result = tl.maximum(tl.maximum(x00, x01), tl.maximum(x10, x11))
    tl.store(Y_ptr + n_idx * sy_n + oh_idx * sy_h + ow_idx * sy_w + c_idx * sy_c, result, mask=mask)


def triton_maxpool2x2(x_nhwc):
    N, IN_H, IN_W, C = x_nhwc.shape
    OUT_H = IN_H // 2
    OUT_W = IN_W // 2
    y = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)
    total = N * OUT_H * OUT_W * C
    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    sy_n, sy_h, sy_w, sy_c = y.stride()
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2x2_kernel[grid](
        x_nhwc, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        sx_n, sx_h, sx_w, sx_c,
        sy_n, sy_h, sy_w, sy_c,
        total,
    )
    return y


# ============================================================
# Linear + bias + optional ReLU kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    USE_RELU: tl.constexpr,
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
        w_ptrs = W_ptr + offs_n[None, :] * K + offs_k[:, None]
        # shape: (BLOCK_K, BLOCK_N)
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    if USE_RELU:
        acc = tl.maximum(acc, 0.0)

    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def triton_linear(x, w, b, use_relu=False):
    """
    returns: (M, N)
    """
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_bias_relu_kernel[grid](
        x, w, b, y,
        M, N, K,
        USE_RELU=use_relu,
    )
    return y


# ============================================================
# Main kernel_function: VGG16 forward pass
# ============================================================

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
    def to_nhwc_weight(w):
        # w: (Co, Ci, Kh, Kw) -> (Co, Kh, Kw, Ci)
        return w.permute(0, 2, 3, 1).contiguous()

    def conv_relu(x_nhwc, w_nchw, bias, pad=1, stride=1, relu=True):
        w_nhwc = to_nhwc_weight(w_nchw)
        return triton_conv2d(x_nhwc, w_nhwc, bias, stride, stride, pad, pad, use_relu=relu)

    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # Block 1
    x_nhwc = conv_relu(x_nhwc, p_features_0_weight, p_features_0_bias)
    x_nhwc = conv_relu(x_nhwc, p_features_2_weight, p_features_2_bias)
    x_nhwc = triton_maxpool2x2(x_nhwc)

    # Block 2
    x_nhwc = conv_relu(x_nhwc, p_features_5_weight, p_features_5_bias)
    x_nhwc = conv_relu(x_nhwc, p_features_7_weight, p_features_7_bias)
    x_nhwc = triton_maxpool2x2(x_nhwc)

    # Block 3
    x_nhwc = conv_relu(x_nhwc, p_features_10_weight, p_features_10_bias)
    x_nhwc = conv_relu(x_nhwc, p_features_12_weight, p_features_12_bias)
    x_nhwc = conv_relu(x_nhwc, p_features_14_weight, p_features_14_bias)
    x_nhwc = triton_maxpool2x2(x_nhwc)

    # Block 4
    x_nhwc = conv_relu(x_nhwc, p_features_17_weight, p_features_17_bias)
    x_nhwc = conv_relu(x_nhwc, p_features_19_weight, p_features_19_bias)
    x_nhwc = conv_relu(x_nhwc, p_features_21_weight, p_features_21_bias)
    x_nhwc = triton_maxpool2x2(x_nhwc)

    # Block 5
    x_nhwc = conv_relu(x_nhwc, p_features_24_weight, p_features_24_bias)
    x_nhwc = conv_relu(x_nhwc, p_features_26_weight, p_features_26_bias)
    x_nhwc = conv_relu(x_nhwc, p_features_28_weight, p_features_28_bias)
    x_nhwc = triton_maxpool2x2(x_nhwc)

    # Flatten: (N, 7, 7, 512) -> (N, 25088)
    N = x_nhwc.shape[0]
    # Need to convert back to NCHW order for flatten to match PyTorch
    # PyTorch flatten is on NCHW: (N, C, H, W) -> (N, C*H*W)
    # Our NHWC: (N, H, W, C) -> need to permute to NCHW then flatten
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_flat = x_nchw.reshape(N, -1)

    # Classifier
    x_flat = triton_linear(x_flat, p_classifier_0_weight, p_classifier_0_bias, use_relu=True)
    # dropout p=0.0 is identity
    x_flat = triton_linear(x_flat, p_classifier_3_weight, p_classifier_3_bias, use_relu=True)
    # dropout p=0.0 is identity
    x_flat = triton_linear(x_flat, p_classifier_6_weight, p_classifier_6_bias, use_relu=False)

    return x_flat


def get_inputs():
    torch.manual_seed(42)
    batch_size = 10
    num_classes = 1000

    import torch.nn as nn

    class Model(nn.Module):
        def __init__(self, num_classes=1000):
            super(Model, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(128, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(256, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(512, 512, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.0),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.0),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    model = Model(num_classes).cuda().eval()

    x = torch.rand(batch_size, 3, 224, 224, device='cuda')

    return [
        model.features[0].weight.data,
        model.features[0].bias.data,
        model.features[2].weight.data,
        model.features[2].bias.data,
        model.features[5].weight.data,
        model.features[5].bias.data,
        model.features[7].weight.data,
        model.features[7].bias.data,
        model.features[10].weight.data,
        model.features[10].bias.data,
        model.features[12].weight.data,
        model.features[12].bias.data,
        model.features[14].weight.data,
        model.features[14].bias.data,
        model.features[17].weight.data,
        model.features[17].bias.data,
        model.features[19].weight.data,
        model.features[19].bias.data,
        model.features[21].weight.data,
        model.features[21].bias.data,
        model.features[24].weight.data,
        model.features[24].bias.data,
        model.features[26].weight.data,
        model.features[26].bias.data,
        model.features[28].weight.data,
        model.features[28].bias.data,
        model.classifier[0].weight.data,
        model.classifier[0].bias.data,
        model.classifier[3].weight.data,
        model.classifier[3].bias.data,
        model.classifier[6].weight.data,
        model.classifier[6].bias.data,
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

        class Model(nn.Module):
            def __init__(self, num_classes=1000):
                super(Model, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(64, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(128, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(256, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, kernel_size