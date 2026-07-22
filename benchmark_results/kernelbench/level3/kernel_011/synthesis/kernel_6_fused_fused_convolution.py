import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused Conv2d + Bias + ReLU (Implicit GEMM, NHWC layout)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K', 'IN_H', 'IN_W', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W', 'STRIDE_H', 'STRIDE_W', 'PAD_H', 'PAD_W'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    M, N, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_RELU: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    
    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)
        
        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid = (
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K) &
            (batch[:, None] < BATCH)
        )
        
        x_ptrs = X_ptr + (
            batch[:, None] * stride_xn +
            in_c[None, :] * stride_xc +
        )
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        w_ptrs = W_ptr + (
            offs_n[None, :] * stride_wn +
            in_c[:, None] * stride_wc +
            kh[:, None] * stride_wh +
            kw[:, None] * stride_ww
        )
        mask_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]
    
    # ReLU
    if USE_RELU:
        acc = tl.maximum(acc, 0.0)
    
    # Store output (NCHW format)
    mask_out = (
        (batch[:, None] < BATCH) &
        (out_h[:, None] < OUT_H) &
        (out_w[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )
    
    y_ptrs = Y_ptr + (
        batch[:, None] * (OUT_C * OUT_H * OUT_W) +
        offs_n[None, :] * (OUT_H * OUT_W) +
        out_h[:, None] * OUT_W +
        out_w[:, None]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_bias_relu(x, weight, bias, stride=1, padding=1, use_relu=True):
    """x: NCHW, weight: (OC, IC, KH, KW), bias: (OC,)"""
    N, IC, IH, IW = x.shape
    OC, _, KH, KW = weight.shape
    
    if isinstance(stride, int):
        stride_h = stride_w = stride
    else:
        stride_h, stride_w = stride
    
    if isinstance(padding, int):
        pad_h = pad_w = padding
    else:
        pad_h, pad_w = padding
    
    OH = (IH + 2 * pad_h - KH) // stride_h + 1
    OW = (IW + 2 * pad_w - KW) // stride_w + 1
    
    M = N * OH * OW
    K_total = IC * KH * KW
    
    y = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)
    
    # x is NCHW: strides are (IC*IH*IW, IH*IW, IW, 1)
    # weight is (OC, IC, KH, KW): strides are (IC*KH*KW, KH*KW, KW, 1)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OC, meta['BLOCK_N']),
    )
    
    conv2d_bias_relu_kernel[grid](
        x, weight, bias, y,
        N, IC, IH, IW,
        OC, OH, OW,
        M, OC, K_total,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        IH, IW,
        OH, OW,
        USE_RELU=use_relu,
    )
    return y


# ============================================================
# Max Pool 2D kernel (2x2, stride=2)
# ============================================================

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
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    total = N * C * OH * OW
    mask = offs < total
    
    # Decode linear index -> (n, c, oh, ow)
    ow = offs % OW
    oh = (offs // OW) % OH
    c = (offs // (OW * OH)) % C
    n = offs // (OW * OH * C)
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            ih = oh * STRIDE_H + kh
            iw = ow * STRIDE_W + kw
            valid = mask & (ih < IH) & (iw < IW)
            x_idx = n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    y_idx = n * (C * OH * OW) + c * (OH * OW) + oh * OW + ow
    tl.store(Y_ptr + y_idx, max_val, mask=mask)


def maxpool2d(x, kernel_size=2, stride=2):
    N, C, IH, IW = x.shape
    OH = (IH - kernel_size) // stride + 1
    OW = (IW - kernel_size) // stride + 1
    
    y = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    
    total = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    
    maxpool2d_kernel[grid](
        x, y,
        N, C, IH, IW, OH, OW,
        kernel_size, kernel_size,
        stride, stride,
    )
    return y


# ============================================================
# Fused Linear + Bias + ReLU using tl.dot
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    USE_RELU: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # W is (N, K) -> W[n, k]
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        mask_w = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    if USE_RELU:
        acc = tl.maximum(acc, 0.0)
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def linear_bias_relu(x, weight, bias, use_relu=True):
    """x: (M, K), weight: (N, K), bias: (N,) -> (M, N)"""
    M, K = x.shape
    N = weight.shape[0]
    
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    linear_bias_relu_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
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
    # Block 1
    x = conv2d_bias_relu(x, p_features_0_weight, p_features_0_bias, stride=1, padding=1, use_relu=True)
    x = conv2d_bias_relu(x, p_features_2_weight, p_features_2_bias, stride=1, padding=1, use_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 2
    x = conv2d_bias_relu(x, p_features_5_weight, p_features_5_bias, stride=1, padding=1, use_relu=True)
    x = conv2d_bias_relu(x, p_features_7_weight, p_features_7_bias, stride=1, padding=1, use_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 3
    x = conv2d_bias_relu(x, p_features_10_weight, p_features_10_bias, stride=1, padding=1, use_relu=True)
    x = conv2d_bias_relu(x, p_features_12_weight, p_features_12_bias, stride=1, padding=1, use_relu=True)
    x = conv2d_bias_relu(x, p_features_14_weight, p_features_14_bias, stride=1, padding=1, use_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 4
    x = conv2d_bias_relu(x, p_features_17_weight, p_features_17_bias, stride=1, padding=1, use_relu=True)
    x = conv2d_bias_relu(x, p_features_19_weight, p_features_19_bias, stride=1, padding=1, use_relu=True)
    x = conv2d_bias_relu(x, p_features_21_weight, p_features_21_bias, stride=1, padding=1, use_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 5
    x = conv2d_bias_relu(x, p_features_24_weight, p_features_24_bias, stride=1, padding=1, use_relu=True)
    x = conv2d_bias_relu(x, p_features_26_weight, p_features_26_bias, stride=1, padding=1, use_relu=True)
    x = conv2d_bias_relu(x, p_features_28_weight, p_features_28_bias, stride=1, padding=1, use_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Flatten
    x = x.reshape(x.shape[0], -1)
    
    # Classifier
    x = linear_bias_relu(x, p_classifier_0_weight, p_classifier_0_bias, use_relu=True)
    # dropout p=0.0 is identity
    x = linear_bias_relu(x, p_classifier_3_weight, p_classifier_3_bias, use_relu=True)
    # dropout p=0.0 is identity
    x = linear_bias_relu(x, p_classifier_6_weight, p_classifier_6_bias, use_relu=False)
    
    return x


def get_inputs():
    torch.manual_seed(42)
    batch_size = 10
    
    def make_conv_weight(oc, ic, kh, kw):
        return torch.randn(oc, ic, kh, kw, dtype=torch.float32, device='cuda') * 0.01
    
    def make_bias(c):
        return torch.zeros(c, dtype=torch.float32, device='cuda')
    
    def make_linear_weight(out, inp):
        return torch.randn(out, inp, dtype=torch.float32, device='cuda') * 0.01
    
    return [
        make_conv_weight(64, 3, 3, 3),    # p_features_0_weight
        make_bias(64),                      # p_features_0_bias
        make_conv_weight(64, 64, 3, 3),   # p_features_2_weight
        make_bias(64),                      # p_features_2_bias
        make_conv_weight(128, 64, 3, 3),  # p_features_5_weight
        make_bias(128),                     # p_features_5_bias
        make_conv_weight(128, 128, 3, 3), # p_features_7_weight
        make_bias(128),                     # p_features_7_bias
        make_conv_weight(256, 128, 3, 3), # p_features_10_weight
        make_bias(256),                     # p_features_10_bias
        make_conv_weight(256, 256, 3, 3), # p_features_12_weight
        make_bias(256),                     # p_features_12_bias
        make_conv_weight(256, 256, 3, 3), # p_features_14_weight
        make_bias(256),                     # p_features_14_bias
        make_conv_weight(512, 256, 3, 3), # p_features_17_weight
        make_bias(512),                     # p_features_17_bias
        make_conv_weight(512, 512, 3, 3), # p_features_19_weight
        make_bias(512),                     # p_features_19_bias
        make_conv_weight(512, 512, 3, 3), # p_features_21_weight
        make_bias(512),                     # p_features_21_bias
        make_conv_weight(512, 512, 3, 3), # p_features_24_weight
        make_bias(512),                     # p_features_24_bias
        make_conv_weight(512, 512, 3, 3), # p_features_26_weight
        make_bias(512),                     # p_features_26_bias
        make_conv_weight(512, 512, 3, 3), # p_features_28_weight
        make_bias(512),                     # p_features_28_bias
        make_linear_weight(4096, 25088),   # p_classifier_0_weight
        make_bias(4096),                    # p_classifier_0_bias
        make_linear_weight(4096, 4096),    # p_classifier_3_weight
        make_bias(4096),                    # p_classifier_3_bias
        make_linear_weight(1000, 4096),    # p_classifier_6_weight
        make_bias(1000),                    # p_classifier_6_bias
        torch.randn(batch_size, 3, 224, 224, dtype=torch.float32, device='cuda') * 0.1,  # x
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
            model.features[19].bias.copy_(p_