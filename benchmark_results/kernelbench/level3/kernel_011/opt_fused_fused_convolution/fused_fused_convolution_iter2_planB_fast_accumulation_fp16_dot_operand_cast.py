import torch
import triton
import triton.language as tl
import math

# ─── Fused Conv2d (implicit GEMM) + bias + optional ReLU ───────────────────


@triton.autotune(
    configs=[
        # Smallest tiles: lowest register pressure, highest occupancy
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        # Medium tiles: balance parallelism and register use
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=1, num_warps=4),
        # Larger tiles only with num_stages=1 to cap register growth
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=1, num_warps=8),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KH', 'KW'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    STR_H: tl.constexpr, STR_W: tl.constexpr,
    # strides for X (NHWC layout)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (out_c, kh, kw, in_c) layout
    sw_o, sw_h, sw_w, sw_i,
    # strides for Y (NHWC layout)
    sy_n, sy_h, sy_w, sy_c,
    DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode m -> (n, oh, ow)
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    n  = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        # decode k -> (ic, kh, kw)
        ic = k_offs % IN_C
        kw_idx = (k_offs // IN_C) % KW
        kh_idx = k_offs // (IN_C * KW)

        ih = oh[:, None] * STR_H + kh_idx[None, :] - PAD_H
        iw = ow[:, None] * STR_W + kw_idx[None, :] - PAD_W

        valid_x = (
            (n[:, None] < BATCH) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (k_offs[None, :] < TOTAL_K)
        )

        x_ptrs = X_ptr + n[:, None] * sx_n + ih * sx_h + iw * sx_w + ic[None, :] * sx_c
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = (k_offs[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * sw_o + kh_idx[:, None] * sw_h + kw_idx[:, None] * sw_w + ic[:, None] * sw_i
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile.to(tl.float16), w_tile.to(tl.float16), acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    if DO_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store (NHWC)
    mask_out = (
        (n[:, None] < BATCH) &
        (oh[:, None] < OUT_H) &
        (ow[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )
    y_ptrs = Y_ptr + n[:, None] * sy_n + oh[:, None] * sy_h + ow[:, None] * sy_w + offs_n[None, :] * sy_c
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit

def conv2d_bias_relu(x_nhwc, w_oihw, bias, pad=1, stride=1, do_relu=True):
    """x_nhwc: (N,H,W,C), w_oihw: (OC,IC,KH,KW), returns (N,OH,OW,OC)"""
    N, H, W, IC = x_nhwc.shape
    OC, IC2, KH, KW = w_oihw.shape
    assert IC == IC2

    OH = (H + 2 * pad - KH) // stride + 1
    OW = (W + 2 * pad - KW) // stride + 1

    y = torch.empty((N, OH, OW, OC), device=x_nhwc.device, dtype=torch.float32)

    M = N * OH * OW

    # W in (OC, KH, KW, IC) layout for kernel
    w_okhwi = w_oihw.permute(0, 2, 3, 1).contiguous()  # (OC, KH, KW, IC)

    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    sw_o = w_okhwi.stride(0)
    sw_h = w_okhwi.stride(1)
    sw_w = w_okhwi.stride(2)
    sw_i = w_okhwi.stride(3)
    sy_n, sy_h, sy_w, sy_c = y.stride()

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )

    conv2d_bias_relu_kernel[grid](
        x_nhwc, w_okhwi, bias, y,
        N, IC, H, W,
        OC, OH, OW,
        KH, KW,
        pad, pad,
        stride, stride,
        sx_n, sx_h, sx_w, sx_c,
        sw_o, sw_h, sw_w, sw_i,
        sy_n, sy_h, sy_w, sy_c,
        do_relu,
    )
    return y


# ─── MaxPool2d kernel ────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    STR_H: tl.constexpr, STR_W: tl.constexpr,
    # X strides (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    # Y strides (NHWC)
    sy_n, sy_h, sy_w, sy_c,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    total = N * OH * OW * C
    mask = offs < total

    # decode: offs -> (n, oh, ow, c)
    c   = offs % C
    ow  = (offs // C) % OW
    oh  = (offs // (C * OW)) % OH
    n   = offs // (C * OW * OH)

    ih_base = oh * STR_H
    iw_base = ow * STR_W

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for ki in range(KH):
        for kj in range(KW):
            ih = ih_base + ki
            iw = iw_base + kj
            valid = mask & (ih < IH) & (iw < IW) & (n < N)
            x_ptrs = X_ptr + n * sx_n + ih * sx_h + iw * sx_w + c * sx_c
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    y_ptrs = Y_ptr + n * sy_n + oh * sy_h + ow * sy_w + c * sy_c
    tl.store(y_ptrs, max_val, mask=mask)


def maxpool2d(x_nhwc, kernel_size=2, stride=2):
    N, H, W, C = x_nhwc.shape
    OH = (H - kernel_size) // stride + 1
    OW = (W - kernel_size) // stride + 1
    y = torch.empty((N, OH, OW, C), device=x_nhwc.device, dtype=torch.float32)

    total = N * OH * OW * C
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    maxpool2d_kernel[grid](
        x_nhwc, y,
        N, C, H, W, OH, OW,
        kernel_size, kernel_size,
        stride, stride,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
    )
    return y


# ─── Fused Linear + bias + optional ReLU ────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[None, :] < N) & (offs_k[:, None] < K)

        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    if DO_RELU:
        acc = tl.maximum(acc, 0.0)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def linear_bias_relu(x, w, b, do_relu=True):
    """x: (M, K), w: (N, K), b: (N,) -> (M, N)"""
    M, K = x.shape
    N = w.shape[0]
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    linear_bias_relu_kernel[grid](
        x, w, b, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        do_relu,
    )
    return y


# ─── Main kernel_function ────────────────────────────────────────────────────

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
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_0_weight, p_features_0_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_2_weight, p_features_2_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = maxpool2d(x_nhwc, kernel_size=2, stride=2)

    # Block 2
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_5_weight, p_features_5_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_7_weight, p_features_7_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = maxpool2d(x_nhwc, kernel_size=2, stride=2)

    # Block 3
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_10_weight, p_features_10_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_12_weight, p_features_12_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_14_weight, p_features_14_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = maxpool2d(x_nhwc, kernel_size=2, stride=2)

    # Block 4
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_17_weight, p_features_17_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_19_weight, p_features_19_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_21_weight, p_features_21_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = maxpool2d(x_nhwc, kernel_size=2, stride=2)

    # Block 5
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_24_weight, p_features_24_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_26_weight, p_features_26_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = conv2d_bias_relu(x_nhwc, p_features_28_weight, p_features_28_bias, pad=1, stride=1, do_relu=True)
    x_nhwc = maxpool2d(x_nhwc, kernel_size=2, stride=2)

    # Flatten: (N, 7, 7, 512) -> (N, 25088)
    N = x_nhwc.shape[0]
    # Convert back to NCHW then flatten for linear
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_flat = x_nchw.reshape(N, -1)

    # Classifier
    x_flat = linear_bias_relu(x_flat, p_classifier_0_weight, p_classifier_0_bias, do_relu=True)
    # dropout p=0.0 is identity
    x_flat = linear_bias_relu(x_flat, p_classifier_3_weight, p_classifier_3_bias, do_relu=True)
    # dropout p=0.0 is identity
    x_flat = linear_bias_relu(x_flat, p_classifier_6_weight, p_classifier_6_bias, do_relu=False)

    return x_flat


def get_inputs():
    torch.manual_seed(42)
    batch_size = 10
    num_classes = 1000

    import torch.nn as nn

    model = nn.Sequential()  # just to get shapes

    def make_weight(shape):
        return torch.randn(shape, dtype=torch.float32, device='cuda') * 0.01

    def make_bias(n):
        return torch.zeros(n, dtype=torch.float32, device='cuda')

    return [
        make_weight([64, 3, 3, 3]),    # p_features_0_weight
        make_bias(64),                  # p_features_0_bias
        make_weight([64, 64, 3, 3]),
        make_bias(64),
        make_weight([128, 64, 3, 3]),
        make_bias(128),
        make_weight([128, 128, 3, 3]),
        make_bias(128),
        make_weight([256, 128, 3, 3]),
        make_bias(256),
        make_weight([256, 256, 3, 3]),
        make_bias(256),
        make_weight([256, 256, 3, 3]),
        make_bias(256),
        make_weight([512, 256, 3, 3]),
        make_bias(512),
        make_weight([512, 512, 3, 3]),
        make_bias(512),
        make_weight([512, 512, 3, 3]),
        make_bias(512),
        make_weight([512, 512, 3, 3]),
        make_bias(512),
        make_weight([512, 512, 3, 3]),
        make_bias(512),
        make_weight([512, 512, 3, 3]),
        make_bias(512),
        make_weight([4096, 25088]),
        make_bias(4096),
        make_weight([4096, 4096]),
        make_bias(4096),
        make_weight([1000, 4096]),
        make_bias(1000),
        torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device='cuda'),
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

        model = Model().cuda().eval()

        # Load weights from inputs
        (p_features_0_weight, p_features_0_bias,
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
         x) = inputs

        with torch.no_grad():
            model.features[0].weight.copy_(p_features_0_weight)
            model.features[0].bias.copy_(p_features_0_bias)
            model.features[2].weight.copy_(p_features_2_weight)
            model.features[2].bias.copy_(p_features_2_bias)
            model.features[5].weight.copy_(p_features_5_weight)
            model.features[5].bias.copy_(p_features_5_bias)
            model.features[7].weight.copy_(p_features_7_weight)
            model.features[7].bias.copy_(p_features_7_bias)
            model.features[10].weight.copy_(p_features_10_weight)
            model.features[10].bias.copy_(p_features_10_bias)
            model.features[12].weight.copy_(p_features_12_weight)
            model.features[12].bias.copy_(p_features_12_bias)
            model.features[14].weight.copy_(p_features_14_weight)
            model.features[14].bias.copy_(p_features_14_bias)
            model.features[17].weight.copy_(p_features_17_weight)
            model.features[17].bias.copy_(p_features_17_bias)
            model.features[19].weight.copy_(p_features_19_weight)