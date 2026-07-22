import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused bias + ReLU pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def fused_bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = offsets % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    out = x + bias
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Max Pool 2D kernel (3x3, stride=2, ceil_mode=True)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def maxpool2d_kernel(
    x_ptr,
    out_ptr,
    N, C,
    IH, IW,
    OH, OW,
    KH, KW,
    SH, SW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * OH * OW
    mask = offsets < total

    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh
            iw = ow * SW + kw
            valid = mask & (ih < IH) & (iw < IW)
            idx = n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(x_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid & (val > max_val), val, max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# ============================================================
# Implicit GEMM Conv2d kernel (NHWC layout internally)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    SH, SW,
    PH, PW,
    M_TOTAL, TOTAL_K,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)

        ih = out_h[:, None] * SH + kh[None, :] - PH
        iw = out_w[:, None] * SW + kw[None, :] - PW

        valid_m = (offs_m < M_TOTAL)
        valid_k = (offs_k < TOTAL_K)
        valid_x = (
            valid_m[:, None]
            & (ih >= 0)
            & (iw >= 0)
        )

        # X in NCHW: batch * IN_C*IN_H*IN_W + in_c * IN_H*IN_W + ih * IN_W + iw
        x_idx = (
            batch[:, None] * (IN_C * IN_H * IN_W)
        )
        x_tile = tl.load(X_ptr + x_idx, mask=valid_x, other=0.0)

        # W in NCHW: out_c * IN_C*KH*KW + in_c * KH*KW + kh * KW + kw
        valid_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_idx = (
            offs_n[None, :] * (IN_C * KH * KW)
        )
        w_tile = tl.load(W_ptr + w_idx, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias and ReLU
    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=(offs_n < OUT_C), other=0.0)
        acc = acc + bias[None, :]

    # Store in NCHW: batch * OUT_C*OUT_H*OUT_W + out_c * OUT_H*OUT_W + out_h * OUT_W + out_w
    valid_out = (
        (offs_m[:, None] < M_TOTAL)
    )
    y_idx = (
        batch[:, None] * (OUT_C * OUT_H * OUT_W)
    )
    tl.store(Y_ptr + y_idx, acc.to(tl.float32), mask=valid_out)


def triton_conv2d_relu(x, weight, bias, stride=(1,1), padding=(0,0)):
    """Run conv2d + bias + relu using Triton implicit GEMM kernel."""
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1

    M_TOTAL = N * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x, weight, bias if bias is not None else x,
        y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        M_TOTAL, TOTAL_K,
        HAS_BIAS=(bias is not None),
    )
    return y


def triton_maxpool2d(x, kernel_size=3, stride=2, ceil_mode=True):
    """Max pool 2D with Triton kernel."""
    N, C, IH, IW = x.shape
    KH = KW = kernel_size
    SH = SW = stride

    if ceil_mode:
        OH = math.ceil((IH - KH) / SH) + 1
        OW = math.ceil((IW - KW) / SW) + 1
    else:
        OH = (IH - KH) // SH + 1
        OW = (IW - KW) // SW + 1

    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, out,
        N, C,
        IH, IW,
        OH, OW,
        KH, KW,
        SH, SW,
    )
    return out


def fire_module(x,
                sq_w, sq_b,
                ex1_w, ex1_b,
    """Fire module: squeeze -> (expand1x1, expand3x3) -> cat."""
    # Squeeze
    sq = triton_conv2d_relu(x, sq_w, sq_b, stride=(1,1), padding=(0,0))
    # Expand 1x1
    ex1 = triton_conv2d_relu(sq, ex1_w, ex1_b, stride=(1,1), padding=(0,0))
    # Expand 3x3
    ex3 = triton_conv2d_relu(sq, ex3_w, ex3_b, stride=(1,1), padding=(1,1))
    return torch.cat([ex1, ex3], dim=1)


def kernel_function(
    p_features_0_weight,
    p_features_0_bias,
    p_features_3_squeeze_weight,
    p_features_3_squeeze_bias,
    p_features_3_expand1x1_weight,
    p_features_3_expand1x1_bias,
    p_features_3_expand3x3_weight,
    p_features_3_expand3x3_bias,
    p_features_4_squeeze_weight,
    p_features_4_squeeze_bias,
    p_features_4_expand1x1_weight,
    p_features_4_expand1x1_bias,
    p_features_4_expand3x3_weight,
    p_features_4_expand3x3_bias,
    p_features_5_squeeze_weight,
    p_features_5_squeeze_bias,
    p_features_5_expand1x1_weight,
    p_features_5_expand1x1_bias,
    p_features_5_expand3x3_weight,
    p_features_5_expand3x3_bias,
    p_features_7_squeeze_weight,
    p_features_7_squeeze_bias,
    p_features_7_expand1x1_weight,
    p_features_7_expand1x1_bias,
    p_features_7_expand3x3_weight,
    p_features_7_expand3x3_bias,
    p_features_8_squeeze_weight,
    p_features_8_squeeze_bias,
    p_features_8_expand1x1_weight,
    p_features_8_expand1x1_bias,
    p_features_8_expand3x3_weight,
    p_features_8_expand3x3_bias,
    p_features_9_squeeze_weight,
    p_features_9_squeeze_bias,
    p_features_9_expand1x1_weight,
    p_features_9_expand1x1_bias,
    p_features_9_expand3x3_weight,
    p_features_9_expand3x3_bias,
    p_features_10_squeeze_weight,
    p_features_10_squeeze_bias,
    p_features_10_expand1x1_weight,
    p_features_10_expand1x1_bias,
    p_features_10_expand3x3_weight,
    p_features_10_expand3x3_bias,
    p_features_12_squeeze_weight,
    p_features_12_squeeze_bias,
    p_features_12_expand1x1_weight,
    p_features_12_expand1x1_bias,
    p_features_12_expand3x3_weight,
    p_features_12_expand3x3_bias,
    p_classifier_1_weight,
    p_classifier_1_bias,
    x,
):
    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    out = triton_conv2d_relu(x, p_features_0_weight, p_features_0_bias,
                             stride=(2, 2), padding=(0, 0))

    # features[2]: MaxPool2d(3, stride=2, ceil_mode=True)
    out = triton_maxpool2d(out, kernel_size=3, stride=2, ceil_mode=True)

    # features[3]: FireModule(96, 16, 64, 64)
    out = fire_module(out,
                      p_features_3_squeeze_weight, p_features_3_squeeze_bias,
                      p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,

    # features[4]: FireModule(128, 16, 64, 64)
    out = fire_module(out,
                      p_features_4_squeeze_weight, p_features_4_squeeze_bias,
                      p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,

    # features[5]: FireModule(128, 32, 128, 128)
    out = fire_module(out,
                      p_features_5_squeeze_weight, p_features_5_squeeze_bias,
                      p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,

    # features[6]: MaxPool2d(3, stride=2, ceil_mode=True)
    out = triton_maxpool2d(out, kernel_size=3, stride=2, ceil_mode=True)

    # features[7]: FireModule(256, 32, 128, 128)
    out = fire_module(out,
                      p_features_7_squeeze_weight, p_features_7_squeeze_bias,
                      p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,

    # features[8]: FireModule(256, 48, 192, 192)
    out = fire_module(out,
                      p_features_8_squeeze_weight, p_features_8_squeeze_bias,
                      p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,

    # features[9]: FireModule(384, 48, 192, 192)
    out = fire_module(out,
                      p_features_9_squeeze_weight, p_features_9_squeeze_bias,
                      p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,

    # features[10]: FireModule(384, 64, 256, 256)
    out = fire_module(out,
                      p_features_10_squeeze_weight, p_features_10_squeeze_bias,
                      p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,

    # features[11]: MaxPool2d(3, stride=2, ceil_mode=True)
    out = triton_maxpool2d(out, kernel_size=3, stride=2, ceil_mode=True)

    # features[12]: FireModule(512, 64, 256, 256)
    out = fire_module(out,
                      p_features_12_squeeze_weight, p_features_12_squeeze_bias,
                      p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,

    # classifier[0]: Dropout(p=0.0) -> identity
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    out = triton_conv2d_relu(out, p_classifier_1_weight, p_classifier_1_bias,
                             stride=(1, 1), padding=(0, 0))

    # classifier[3]: AdaptiveAvgPool2d((1,1))
    out = out.mean(dim=(2, 3), keepdim=True)

    # flatten
    out = out.reshape(out.shape[0], -1)

    return out


def get_inputs():
    batch_size = 64
    return [
        torch.randn(96, 3, 7, 7, dtype=torch.float32, device='cuda'),       # p_features_0_weight
        torch.randn(96, dtype=torch.float32, device='cuda'),                  # p_features_0_bias
        torch.randn(16, 96, 1, 1, dtype=torch.float32, device='cuda'),       # p_features_3_squeeze_weight
        torch.randn(16, dtype=torch.float32, device='cuda'),                  # p_features_3_squeeze_bias
        torch.randn(64, 16, 1, 1, dtype=torch.float32, device='cuda'),       # p_features_3_expand1x1_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                  # p_features_3_expand1x1_bias
        torch.randn(64, 16, 3, 3, dtype=torch.float32, device='cuda'),       # p_features_3_expand3x3_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                  # p_features_3_expand3x3_bias
        torch.randn(16, 128, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_4_squeeze_weight
        torch.randn(16, dtype=torch.float32, device='cuda'),                  # p_features_4_squeeze_bias
        torch.randn(64, 16, 1, 1, dtype=torch.float32, device='cuda'),       # p_features_4_expand1x1_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                  # p_features_4_expand1x1_bias
        torch.randn(64, 16, 3, 3, dtype=torch.float32, device='cuda'),       # p_features_4_expand3x3_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                  # p_features_4_expand3x3_bias
        torch.randn(32, 128, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_5_squeeze_weight
        torch.randn(32, dtype=torch.float32, device='cuda'),                  # p_features_5_squeeze_bias
        torch.randn(128, 32, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_5_expand1x1_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                 # p_features_5_expand1x1_bias
        torch.randn(128, 32, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_5_expand3x3_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                 # p_features_5_expand3x3_bias
        torch.randn(32, 256, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_7_squeeze_weight
        torch.randn(32, dtype=torch.float32, device='cuda'),                  # p_features_7_squeeze_bias
        torch.randn(128, 32, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_7_expand1x1_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                 # p_features_7_expand1x1_bias
        torch.randn(128, 32, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_7_expand3x3_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                 # p_features_7_expand3x3_bias
        torch.randn(48, 256, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_8_squeeze_weight
        torch.randn(48, dtype=torch.float32, device='cuda'),                  # p_features_8_squeeze_bias
        torch.randn(192, 48, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_8_expand1x1_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                 # p_features_8_expand1x1_bias
        torch.randn(192, 48, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_8_expand3x3_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                 # p_features_8_expand3x3_bias
        torch.randn(48, 384, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_9_squeeze_weight
        torch.randn(48, dtype=torch.float32, device='cuda'),                  # p_features_9_squeeze_bias
        torch.randn(192, 48, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_9_expand1x1_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                 # p_features_9_expand1x1_bias
        torch.randn(192, 48, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_9_expand3x3_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                 # p_features_9_expand3x3_bias
        torch.randn(64, 384, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_10_squeeze_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                  # p_features_10_squeeze_bias
        torch.randn(256, 64, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_10_expand1x1_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),                 # p_features_10_expand1x1_bias
        torch.randn(256, 64, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_10_expand3x3_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),                 # p_features_10_expand3x3_bias
        torch.randn(64, 512, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_12_squeeze_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                  # p_features_12_squeeze_bias
        torch.randn(256, 64, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_12_expand1x1_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),                 # p_features_12_expand1x1_bias
        torch.randn(256, 64, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_12_expand3x3_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),                 # p_features_12_expand3x3_bias
        torch.randn(1000, 512, 1, 1, dtype=torch.float32, device='cuda'),    # p_classifier_1_weight
        torch.randn(1000, dtype=torch.float32, device='cuda'),                # p_classifier_1_bias
        torch.randn(batch_size, 3, 512, 512, dtype=torch.float32, device='cuda'),  # x
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

        class FireModule(nn.Module):
            def __init__(self, in_channels, squeeze_channels, expand1x1_channels, expand3x3_channels):
                super(FireModule, self).__init__()
                self.squeeze = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1)
                self.squeeze_activation = nn.ReLU(inplace=True)
                self.expand1x1 = nn.Conv2d(squeeze_channels, expand1x1_channels, kernel_size=1)
                self.expand1x1_activation = nn.ReLU(inplace=True)
                self.expand3x3 = nn.Conv2d(squeeze_channels, expand3x3_channels, kernel_size=3, padding=1)
                self.expand3x3_activation = nn.ReLU(inplace=True)

            def forward(self, x):
                x = self.squeeze_activation(self.squeeze(x))
                return torch.cat([
                    self.expand1x1_activation(self.expand1x1(x)),
                    self.expand3x3_activation(self.expand3x3(x))
                ], 1)

        class Model(nn.Module):
            def __init__(self, num_classes=1000):
                super(Model, self).__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 96, kernel_size=7, stride=2),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    FireModule(96, 16, 64, 64),
                    FireModule(128, 16, 64, 64),
                    FireModule(128, 32, 128, 128),
                    nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                    FireModule(256, 32, 128, 128),
                    FireModule(256, 48, 192, 192),
                    FireModule(384, 48, 192,