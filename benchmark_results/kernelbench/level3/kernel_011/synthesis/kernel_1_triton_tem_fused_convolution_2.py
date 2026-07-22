import torch
import triton
import triton.language as tl
import math

# ─── Pointwise: bias + relu (in-place on output) ───────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=4),
        triton.Config({'BLOCK': 128},  num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def bias_relu_kernel(out_ptr, bias_ptr, N, C, do_relu: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(out_ptr + offs, mask=mask, other=0.0)
    c = offs % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    x = x + b
    if do_relu:
        x = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offs, x, mask=mask)

# ─── Max Pool 2x2 stride 2 ─────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=4),
        triton.Config({'BLOCK': 128},  num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def maxpool2x2_kernel(
    inp_ptr, out_ptr,
    N,
    B, C, IH, IW, OH, OW,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # decode: offs = b*C*OH*OW + c*OH*OW + oh*OW + ow
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    b = tmp2 // C
    ih0 = oh * 2
    iw0 = ow * 2
    # 4 positions
    def load_inp(ih, iw):
        valid = (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW) & mask
        idx = b * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
        return tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
    v00 = load_inp(ih0,   iw0)
    v01 = load_inp(ih0,   iw0+1)
    v10 = load_inp(ih0+1, iw0)
    v11 = load_inp(ih0+1, iw0+1)
    vmax = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    tl.store(out_ptr + offs, vmax, mask=mask)

# ─── Implicit GEMM Conv2d (NCHW, 3x3, pad=1, stride=1) ────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_3x3_p1_s1_kernel(
    X_ptr, W_ptr, Out_ptr,
    B, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = B*OUT_H*OUT_W, N = OUT_C, K = IN_C*9
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    # decode m -> (batch, out_h, out_w)
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    bn = tmp // OUT_H
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # decode k -> (in_c, kh, kw)
        kw = offs_k % 3
        tmp_k = offs_k // 3
        kh = tmp_k % 3
        ic = tmp_k // 3
        # input coords
        ih = oh[:, None] + kh[None, :] - 1  # pad=1
        iw = ow[:, None] + kw[None, :] - 1
        valid_x = (bn[:, None] < B) & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & (offs_k[None, :] < K)
        x_idx = bn[:, None] * (IN_C * IN_H * IN_W) + ic[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=valid_x, other=0.0)
        # weight: [OUT_C, IN_C, 3, 3] -> index [n, k] = offs_n[n]*K + offs_k[k]
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_idx = offs_n[None, :] * K + offs_k[:, None]
        w_tile = tl.load(W_ptr + w_idx, mask=valid_w, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    # store
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_idx = bn[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + oh[:, None] * OUT_W + ow[:, None]
    tl.store(Out_ptr + out_idx, acc, mask=valid_out)

# ─── Linear (GEMM): [M, K] x [N, K]^T -> [M, N] ──────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    do_relu: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        valid_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=valid_x, other=0.0)
        valid_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_tile = tl.load(W_ptr + offs_n[:, None] * K + offs_k[None, :], mask=valid_w, other=0.0)
        acc = tl.dot(x_tile, w_tile.T, acc, allow_tf32=False)
    # add bias + optional relu
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    bias = tl.load(B_ptr + offs_n[None, :], mask=(offs_n[None, :] < N), other=0.0)
    acc = acc + bias
    if do_relu:
        acc = tl.where(acc > 0.0, acc, 0.0)
    tl.store(Out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=valid_out)


def run_conv(x, w, bias, do_relu=True):
    B, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    assert KH == 3 and KW == 3
    OUT_H = IN_H  # pad=1, stride=1
    OUT_W = IN_W
    M = B * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * 9
    out = torch.empty((B, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    # Reshape weight to [OUT_C, K]
    w_flat = w.reshape(OUT_C, K).contiguous()
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    conv2d_3x3_p1_s1_kernel[grid](
        x, w_flat, out,
        B, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        M, N, K,
    )
    # add bias + relu
    total = B * OUT_C * OUT_H * OUT_W
    grid2 = lambda meta: (triton.cdiv(total, meta['BLOCK']),)
    bias_relu_kernel[grid2](out, bias, total, OUT_C, do_relu)
    return out


def run_maxpool(x):
    B, C, IH, IW = x.shape
    OH = IH // 2
    OW = IW // 2
    out = torch.empty((B, C, OH, OW), device=x.device, dtype=x.dtype)
    N = B * C * OH * OW
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    maxpool2x2_kernel[grid](x, out, N, B, C, IH, IW, OH, OW)
    return out


def run_linear(x, w, bias, do_relu=False):
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, w, bias, out, M, N, K, do_relu)
    return out


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
):
    # Block 1
    x = run_conv(x, p_features_0_weight, p_features_0_bias, do_relu=True)
    x = run_conv(x, p_features_2_weight, p_features_2_bias, do_relu=True)
    x = run_maxpool(x)
    # Block 2
    x = run_conv(x, p_features_5_weight, p_features_5_bias, do_relu=True)
    x = run_conv(x, p_features_7_weight, p_features_7_bias, do_relu=True)
    x = run_maxpool(x)
    # Block 3
    x = run_conv(x, p_features_10_weight, p_features_10_bias, do_relu=True)
    x = run_conv(x, p_features_12_weight, p_features_12_bias, do_relu=True)
    x = run_conv(x, p_features_14_weight, p_features_14_bias, do_relu=True)
    x = run_maxpool(x)
    # Block 4
    x = run_conv(x, p_features_17_weight, p_features_17_bias, do_relu=True)
    x = run_conv(x, p_features_19_weight, p_features_19_bias, do_relu=True)
    x = run_conv(x, p_features_21_weight, p_features_21_bias, do_relu=True)
    x = run_maxpool(x)
    # Block 5
    x = run_conv(x, p_features_24_weight, p_features_24_bias, do_relu=True)
    x = run_conv(x, p_features_26_weight, p_features_26_bias, do_relu=True)
    x = run_conv(x, p_features_28_weight, p_features_28_bias, do_relu=True)
    x = run_maxpool(x)
    # Flatten
    x = x.reshape(x.shape[0], -1)
    # Classifier
    x = run_linear(x, p_classifier_0_weight, p_classifier_0_bias, do_relu=True)
    x = run_linear(x, p_classifier_3_weight, p_classifier_3_bias, do_relu=True)
    x = run_linear(x, p_classifier_6_weight, p_classifier_6_bias, do_relu=False)
    return x


def get_inputs():
    torch.manual_seed(0)
    device = 'cuda'
    return [
        torch.randn(64, 3, 3, 3, device=device),
        torch.randn(64, device=device),
        torch.randn(64, 64, 3, 3, device=device),
        torch.randn(64, device=device),
        torch.randn(128, 64, 3, 3, device=device),
        torch.randn(128, device=device),
        torch.randn(128, 128, 3, 3, device=device),
        torch.randn(128, device=device),
        torch.randn(256, 128, 3, 3, device=device),
        torch.randn(256, device=device),
        torch.randn(256, 256, 3, 3, device=device),
        torch.randn(256, device=device),
        torch.randn(256, 256, 3, 3, device=device),
        torch.randn(256, device=device),
        torch.randn(512, 256, 3, 3, device=device),
        torch.randn(512, device=device),
        torch.randn(512, 512, 3, 3, device=device),
        torch.randn(512, device=device),
        torch.randn(512, 512, 3, 3, device=device),
        torch.randn(512, device=device),
        torch.randn(512, 512, 3, 3, device=device),
        torch.randn(512, device=device),
        torch.randn(512, 512, 3, 3, device=device),
        torch.randn(512, device=device),
        torch.randn(512, 512, 3, 3, device=device),
        torch.randn(512, device=device),
        torch.randn(4096, 25088, device=device),
        torch.randn(4096, device=device),
        torch.randn(4096, 4096, device=device),
        torch.randn(4096, device=device),
        torch.randn(1000, 4096, device=device),
        torch.randn(1000, device=device),
        torch.randn(10, 3, 224, 224, device=device),
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
                    nn.Linear(25088, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.0),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.0),
                    nn.Linear(4096, 1000),
                )

        model = VGG16().cuda().eval()
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
            model.features[19].bias.copy_(p_features_19_bias)
            model.features[21].weight.copy_(p_features_21_weight)
            model.features[21].bias.copy_(p_features_21_bias)
            model.features[24].weight.copy_(p_features_24_weight)
            model.features[24].bias.copy_(p_features_24_bias)
            model.features[26].weight.copy_(p_features_26_weight)
            model.features[26].bias.copy_(p_features_26_bias)
            model.features[28].weight.copy_(p_features_28_weight)
            model.features[28].bias.copy_(p_features_28_bias)
            model.classifier[0].weight.copy_(p_classifier_0_weight)
            model.classifier[0].bias.copy_(p_classifier_0_bias)
            model.classifier[3].weight.copy_(p_classifier_3_weight)
            model.classifier[3].bias.copy_(p_classifier_3_bias)
            model.classifier[6].weight.copy_(p_classifier_6_weight)
            model.classifier[6].bias.copy_(p_classifier_6_bias)
            ref = model(x)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")