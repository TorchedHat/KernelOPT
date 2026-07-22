import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused Conv2d + Bias + ReLU kernel (implicit GEMM)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW, SH, SW, PH, PW,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yh, stride_yw,
    M, K,
    apply_relu: tl.constexpr,
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
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KW
        kh_idx = tmp_k // KW
        
        # Input coordinates
        ih = out_h_idx[:, None] * SH + kh_idx[None, :] - PH
        iw = out_w_idx[:, None] * SW + kw_idx[None, :] - PW
        
        valid_x = (
            (offs_m[:, None] < M) &
            (batch_idx[:, None] < N) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < K)
        )
        
        x_ptrs = (X_ptr + 
                  batch_idx[:, None] * stride_xn +
                  in_c_idx[None, :] * stride_xc +
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        valid_w = (
            (offs_n[None, :] < OUT_C) &
            (offs_k[:, None] < K)
        )
        
        w_ptrs = (W_ptr +
                  offs_n[None, :] * stride_wco +
                  in_c_idx[:, None] * stride_wci +
                  kh_idx[:, None] * stride_wkh +
                  kw_idx[:, None] * stride_wkw)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]
    
    # Apply ReLU
    if apply_relu:
        acc = tl.maximum(acc, 0.0)
    
    # Store output
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * stride_yn +
              offs_n[None, :] * stride_yc +
              out_h_idx[:, None] * stride_yh +
              out_w_idx[:, None] * stride_yw)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_bias_relu(x, weight, bias, stride=1, padding=1, apply_relu=True):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    SH = SW = stride
    PH = PW = padding
    
    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1
    
    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_bias_relu_kernel[grid](
        x, weight, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        M, K,
        apply_relu,
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
    key=['total'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, SH, SW,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total
    
    # Decode linear index -> (n, c, oh, ow)
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C
    
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh
            iw = ow * SW + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_idx = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    y_idx = n * (C * OUT_H * OUT_W) + c * (OUT_H * OUT_W) + oh * OUT_W + ow
    tl.store(Y_ptr + y_idx, max_val, mask=mask)


def maxpool2d(x, kernel_size=2, stride=2):
    N, C, IN_H, IN_W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    OUT_H = (IN_H - KH) // SH + 1
    OUT_W = (IN_W - KW) // SW + 1
    
    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    total = N * C * OUT_H * OUT_W
    
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, SH, SW,
        total,
    )
    return y


# ============================================================
# Fused Linear + Bias + ReLU kernel (GEMM)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=2, num_warps=4),
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
    apply_relu: tl.constexpr,
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
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        
        # W is (N, K) - weight matrix, we need W^T effectively
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    if apply_relu:
        acc = tl.maximum(acc, 0.0)
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def linear_bias_relu(x, weight, bias, apply_relu=True):
    # x: (M, K), weight: (N, K), bias: (N,)
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
        y.stride(0), y.stride(1),
        apply_relu,
    )
    return y


# ============================================================
# Main kernel function
# ============================================================

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
    # Block 1
    x = conv2d_bias_relu(x, p_features_0_weight, p_features_0_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_bias_relu(x, p_features_2_weight, p_features_2_bias, stride=1, padding=1, apply_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 2
    x = conv2d_bias_relu(x, p_features_5_weight, p_features_5_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_bias_relu(x, p_features_7_weight, p_features_7_bias, stride=1, padding=1, apply_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 3
    x = conv2d_bias_relu(x, p_features_10_weight, p_features_10_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_bias_relu(x, p_features_12_weight, p_features_12_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_bias_relu(x, p_features_14_weight, p_features_14_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_bias_relu(x, p_features_16_weight, p_features_16_bias, stride=1, padding=1, apply_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 4
    x = conv2d_bias_relu(x, p_features_19_weight, p_features_19_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_bias_relu(x, p_features_21_weight, p_features_21_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_bias_relu(x, p_features_23_weight, p_features_23_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_bias_relu(x, p_features_25_weight, p_features_25_bias, stride=1, padding=1, apply_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Block 5
    x = conv2d_bias_relu(x, p_features_28_weight, p_features_28_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_bias_relu(x, p_features_30_weight, p_features_30_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_bias_relu(x, p_features_32_weight, p_features_32_bias, stride=1, padding=1, apply_relu=True)
    x = conv2d_bias_relu(x, p_features_34_weight, p_features_34_bias, stride=1, padding=1, apply_relu=True)
    x = maxpool2d(x, kernel_size=2, stride=2)
    
    # Flatten
    x = x.reshape(x.shape[0], -1)
    
    # Classifier
    x = linear_bias_relu(x, p_classifier_0_weight, p_classifier_0_bias, apply_relu=True)
    x = linear_bias_relu(x, p_classifier_3_weight, p_classifier_3_bias, apply_relu=True)
    x = linear_bias_relu(x, p_classifier_6_weight, p_classifier_6_bias, apply_relu=False)
    
    return x


def get_inputs():
    p_features_0_weight = torch.randn(64, 3, 3, 3, device='cuda')
    p_features_0_bias = torch.randn(64, device='cuda')
    p_features_2_weight = torch.randn(64, 64, 3, 3, device='cuda')
    p_features_2_bias = torch.randn(64, device='cuda')
    p_features_5_weight = torch.randn(128, 64, 3, 3, device='cuda')
    p_features_5_bias = torch.randn(128, device='cuda')
    p_features_7_weight = torch.randn(128, 128, 3, 3, device='cuda')
    p_features_7_bias = torch.randn(128, device='cuda')
    p_features_10_weight = torch.randn(256, 128, 3, 3, device='cuda')
    p_features_10_bias = torch.randn(256, device='cuda')
    p_features_12_weight = torch.randn(256, 256, 3, 3, device='cuda')
    p_features_12_bias = torch.randn(256, device='cuda')
    p_features_14_weight = torch.randn(256, 256, 3, 3, device='cuda')
    p_features_14_bias = torch.randn(256, device='cuda')
    p_features_16_weight = torch.randn(256, 256, 3, 3, device='cuda')
    p_features_16_bias = torch.randn(256, device='cuda')
    p_features_19_weight = torch.randn(512, 256, 3, 3, device='cuda')
    p_features_19_bias = torch.randn(512, device='cuda')
    p_features_21_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_21_bias = torch.randn(512, device='cuda')
    p_features_23_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_23_bias = torch.randn(512, device='cuda')
    p_features_25_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_25_bias = torch.randn(512, device='cuda')
    p_features_28_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_28_bias = torch.randn(512, device='cuda')
    p_features_30_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_30_bias = torch.randn(512, device='cuda')
    p_features_32_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_32_bias = torch.randn(512, device='cuda')
    p_features_34_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_34_bias = torch.randn(512, device='cuda')
    p_classifier_0_weight = torch.randn(4096, 25088, device='cuda')
    p_classifier_0_bias = torch.randn(4096, device='cuda')
    p_classifier_3_weight = torch.randn(4096, 4096, device='cuda')
    p_classifier_3_bias = torch.randn(4096, device='cuda')
    p_classifier_6_weight = torch.randn(1000, 4096, device='cuda')
    p_classifier_6_bias = torch.randn(1000, device='cuda')
    x = torch.randn(10, 3, 224, 224, device='cuda')
    
    return [
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
        model = nn.Sequential(
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
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
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
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        (
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
            x_in,
        ) = inputs
        
        # Set weights manually
        layers = list(model.children())
        conv_layers = [l for l in layers if isinstance(l, nn.Conv2d)]
        weights = [
            (p_features_0_weight, p_features_0_bias),
            (p_features_2_weight, p_features_2_bias),
            (p_features_5_weight, p_features_5_bias),
            (p_features_7_weight, p_features_7_bias),
            (p_features_10_weight, p_features_10_bias),
            (p_features_12_weight, p_features_12_bias),