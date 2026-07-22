import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M_TOTAL', 'OUT_C', 'IN_C', 'KH', 'KW'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M_TOTAL,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    APPLY_RELU: tl.constexpr,
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

    mask_m = offs_m < M_TOTAL
    mask_n = offs_n < OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOTAL_K

        # Decode K -> (in_c, kh, kw)
        in_c_idx = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KW
        kh_idx = offs_k // (IN_C * KW)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (mask_m[:, None] &
                   (ih >= 0) & (ih < IN_H) &
                   (iw >= 0) & (iw < IN_W) &
                   mask_k[None, :])

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * stride_xn +
                  in_c_idx[None, :] * stride_xc +
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> index [offs_n, in_c_idx, kh_idx, kw_idx]
        valid_w = mask_k[:, None] & mask_n[None, :]
        w_ptrs = (W_ptr +
                  offs_n[None, :] * stride_wn +
                  in_c_idx[:, None] * stride_wc +
                  kh_idx[:, None] * stride_wh +
                  kw_idx[:, None] * stride_ww)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output: Y[batch, out_c, out_h, out_w]
    out_mask = mask_m[:, None] & mask_n[None, :]
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * stride_yn +
              offs_n[None, :] * stride_yc +
              out_h_idx[:, None] * stride_yh +
              out_w_idx[:, None] * stride_yw)
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M_SIZE', 'N_SIZE', 'K_SIZE'],
)
@triton.jit
def linear_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M_SIZE, N_SIZE, K_SIZE,
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

    mask_m = offs_m < M_SIZE
    mask_n = offs_n < N_SIZE

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_SIZE, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K_SIZE

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    out_mask = mask_m[:, None] & mask_n[None, :]
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def max_pool2d_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W,
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

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h_idx * STRIDE_H + kh
            iw = out_w_idx * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_idx = batch_idx * (C * IN_H * IN_W) + c_idx * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    y_idx = batch_idx * (C * OUT_H * OUT_W) + c_idx * (OUT_H * OUT_W) + out_h_idx * OUT_W + out_w_idx
    tl.store(Y_ptr + y_idx, max_val, mask=mask)


def conv2d_relu(x, w, b, stride=1, padding=1, apply_relu=True):
    batch, in_c, in_h, in_w = x.shape
    out_c, _, kh, kw = w.shape
    out_h = (in_h + 2 * padding - kh) // stride + 1
    out_w = (in_w + 2 * padding - kw) // stride + 1
    M_TOTAL = batch * out_h * out_w

    y = torch.empty((batch, out_c, out_h, out_w), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(out_c, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        x, w, b, y,
        batch, in_c, in_h, in_w,
        out_c, out_h, out_w,
        kh, kw,
        stride, stride,
        padding, padding,
        M_TOTAL,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        apply_relu,
    )
    return y


def max_pool2d(x, ksize=2, stride=2):
    batch, c, in_h, in_w = x.shape
    out_h = (in_h - ksize) // stride + 1
    out_w = (in_w - ksize) // stride + 1
    N = batch * c * out_h * out_w
    y = torch.empty((batch, c, out_h, out_w), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, y,
        batch, c, in_h, in_w, out_h, out_w,
        ksize, ksize, stride, stride,
        N,
    )
    return y


def linear_relu(x, w, b, apply_relu=True):
    # x: [M, K], w: [N, K], b: [N]
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    linear_bias_relu_kernel[grid](
        x, w, b, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
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
    x = conv2d_relu(x, p_features_0_weight, p_features_0_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_2_weight, p_features_2_bias, stride=1, padding=1, apply_relu=True)
    x = max_pool2d(x, ksize=2, stride=2)

    # Block 2
    x = conv2d_relu(x, p_features_5_weight, p_features_5_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_7_weight, p_features_7_bias, stride=1, padding=1, apply_relu=True)
    x = max_pool2d(x, ksize=2, stride=2)

    # Block 3
    x = conv2d_relu(x, p_features_10_weight, p_features_10_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_12_weight, p_features_12_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_14_weight, p_features_14_bias, stride=1, padding=1, apply_relu=True)
    x = max_pool2d(x, ksize=2, stride=2)

    # Block 4
    x = conv2d_relu(x, p_features_17_weight, p_features_17_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_19_weight, p_features_19_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_21_weight, p_features_21_bias, stride=1, padding=1, apply_relu=True)
    x = max_pool2d(x, ksize=2, stride=2)

    # Block 5
    x = conv2d_relu(x, p_features_24_weight, p_features_24_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_26_weight, p_features_26_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_relu(x, p_features_28_weight, p_features_28_bias, stride=1, padding=1, apply_relu=True)
    x = max_pool2d(x, ksize=2, stride=2)

    # Flatten
    x = x.reshape(x.shape[0], -1)

    # Classifier
    x = linear_relu(x, p_classifier_0_weight, p_classifier_0_bias, apply_relu=True)
    x = linear_relu(x, p_classifier_3_weight, p_classifier_3_bias, apply_relu=True)
    x = linear_relu(x, p_classifier_6_weight, p_classifier_6_bias, apply_relu=False)

    return x


def get_inputs():
    torch.manual_seed(42)
    batch_size = 10

    def make_conv_params(out_c, in_c, kh=3, kw=3):
        w = torch.randn(out_c, in_c, kh, kw, dtype=torch.float32, device='cuda') * 0.01
        b = torch.zeros(out_c, dtype=torch.float32, device='cuda')
        return w, b

    def make_linear_params(out_f, in_f):
        w = torch.randn(out_f, in_f, dtype=torch.float32, device='cuda') * 0.01
        b = torch.zeros(out_f, dtype=torch.float32, device='cuda')
        return w, b

    w0, b0 = make_conv_params(64, 3)
    w2, b2 = make_conv_params(64, 64)
    w5, b5 = make_conv_params(128, 64)
    w7, b7 = make_conv_params(128, 128)
    w10, b10 = make_conv_params(256, 128)
    w12, b12 = make_conv_params(256, 256)
    w14, b14 = make_conv_params(256, 256)
    w17, b17 = make_conv_params(512, 256)
    w19, b19 = make_conv_params(512, 512)
    w21, b21 = make_conv_params(512, 512)
    w24, b24 = make_conv_params(512, 512)
    w26, b26 = make_conv_params(512, 512)
    w28, b28 = make_conv_params(512, 512)
    wc0, bc0 = make_linear_params(4096, 25088)
    wc3, bc3 = make_linear_params(4096, 4096)
    wc6, bc6 = make_linear_params(1000, 4096)

    x = torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device='cuda')

    return [
        w0, b0, w2, b2, w5, b5, w7, b7,
        w10, b10, w12, b12, w14, b14,
        w17, b17, w19, b19, w21, b21,
        w24, b24, w26, b26, w28, b28,
        wc0, bc0, wc3, bc3, wc6, bc6,
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

        # Build reference model
        class VGG16(nn.Module):
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
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(512, 512, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                self.classifier = nn.Sequential(
                    nn.Linear(512 * 7 * 7, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.0),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
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
        inp = inputs
        idx = 0
        conv_layers = [l for l in model.features if isinstance(l, nn.Conv2d)]
        for layer in conv_layers:
            layer.weight.data.copy_(inp[idx])
            layer.bias.data.copy_(inp[idx + 1])
            idx += 2

        fc_layers = [l for l in model.classifier if isinstance(l, nn.Linear)]
        for layer in fc_layers:
            layer.weight.data.copy_(inp[idx])
            layer.bias.data.copy_(inp[idx + 1])
            idx += 2

        x_ref = inp[-1]

        with torch.no_grad():
            ref = model(x_ref)

        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")