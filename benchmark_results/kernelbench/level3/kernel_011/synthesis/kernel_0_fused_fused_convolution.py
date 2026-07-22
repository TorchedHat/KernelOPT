import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# Fused bias add + ReLU pointwise kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + bias
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


# Max pool 2x2 stride 2 kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['n_out'],
)
@triton.jit
def maxpool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    OH, OW,
    n_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_out
    # decode: offsets -> (n, c, oh, ow)
    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C
    # gather 2x2
    ih0 = oh * 2
    iw0 = ow * 2
    # base pointer
    base = n * C * H * W + c * H * W
    p00 = base + ih0 * W + iw0
    p01 = base + ih0 * W + (iw0 + 1)
    p10 = base + (ih0 + 1) * W + iw0
    p11 = base + (ih0 + 1) * W + (iw0 + 1)
    v00 = tl.load(x_ptr + p00, mask=mask, other=float('-inf'))
    v01 = tl.load(x_ptr + p01, mask=mask & (iw0 + 1 < W), other=float('-inf'))
    v10 = tl.load(x_ptr + p10, mask=mask & (ih0 + 1 < H), other=float('-inf'))
    v11 = tl.load(x_ptr + p11, mask=mask & (ih0 + 1 < H) & (iw0 + 1 < W), other=float('-inf'))
    vmax = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    tl.store(out_ptr + offsets, vmax, mask=mask)


# Implicit GEMM convolution kernel (3x3, stride=1, pad=1)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    N_batch, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    M, N, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # decode m -> (batch, out_h, out_w)
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    nb = tmp // OUT_H
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = IN_C * 3 * 3
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # decode k -> (in_c, kh, kw)
        kw = offs_k % 3
        tmp_k = offs_k // 3
        kh = tmp_k % 3
        ic = tmp_k // 3
        
        # input coordinates
        ih = oh[:, None] + kh[None, :] - 1  # padding=1
        iw = ow[:, None] + kw[None, :] - 1
        
        valid_m = offs_m < M
        valid_k = offs_k < K
        valid = valid_m[:, None] & valid_k[None, :] & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        
        x_ptrs = X_ptr + nb[:, None] * stride_xn + ic[None, :] * stride_xc + ih * stride_xh + iw * stride_xw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        # weight: [K, N] = [IN_C*kh*kw, OUT_C]
        valid_w = valid_k[:, None] & (offs_n[None, :] < N)
        w_ptrs = W_ptr + offs_n[None, :] * stride_wco + ic[:, None] * stride_wci + kh[:, None] * stride_wkh + kw[:, None] * stride_wkw
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # store
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = Out_ptr + (nb[:, None] * OUT_C * OUT_H * OUT_W + 
                          offs_n[None, :] * OUT_H * OUT_W + 
                          oh[:, None] * OUT_W + ow[:, None])
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


# Linear layer kernel using tl.dot
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
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
        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = W_ptr + offs_n[None, :] * K + offs_k[:, None]  # W is [N, K]
        
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # add bias
    b_mask = offs_n < N
    bias = tl.load(B_ptr + offs_n, mask=b_mask, other=0.0)
    acc = acc + bias[None, :]
    
    if do_relu:
        acc = tl.where(acc > 0.0, acc, 0.0)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = Out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def run_conv2d_relu(x, weight, bias, relu=True):
    """Run conv2d (3x3, pad=1, stride=1) + optional relu"""
    N_batch, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    OUT_H = IN_H
    OUT_W = IN_W
    
    M = N_batch * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KH * KW
    
    out = torch.empty((N_batch, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    conv2d_kernel[grid](
        x, weight, out,
        N_batch, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
    )
    
    # fuse bias + relu
    n_elements = out.numel()
    if relu:
        grid2 = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        bias_relu_kernel[grid2](out, bias, out, n_elements, OUT_C)
    else:
        grid2 = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        bias_kernel[grid2](out, bias, out, n_elements, OUT_C)
    
    return out


def run_maxpool2d(x):
    N, C, H, W = x.shape
    OH = H // 2
    OW = W // 2
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    n_out = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(n_out, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](x, out, N, C, H, W, OH, OW, n_out)
    return out


def run_linear(x, weight, bias, relu=False):
    """x: [M, K], weight: [N, K], bias: [N]"""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_relu_kernel[grid](x, weight, bias, out, M, N, K, relu)
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
    x = run_conv2d_relu(x, p_features_0_weight, p_features_0_bias, relu=True)
    x = run_conv2d_relu(x, p_features_2_weight, p_features_2_bias, relu=True)
    x = run_maxpool2d(x)
    
    # Block 2
    x = run_conv2d_relu(x, p_features_5_weight, p_features_5_bias, relu=True)
    x = run_conv2d_relu(x, p_features_7_weight, p_features_7_bias, relu=True)
    x = run_maxpool2d(x)
    
    # Block 3
    x = run_conv2d_relu(x, p_features_10_weight, p_features_10_bias, relu=True)
    x = run_conv2d_relu(x, p_features_12_weight, p_features_12_bias, relu=True)
    x = run_conv2d_relu(x, p_features_14_weight, p_features_14_bias, relu=True)
    x = run_maxpool2d(x)
    
    # Block 4
    x = run_conv2d_relu(x, p_features_17_weight, p_features_17_bias, relu=True)
    x = run_conv2d_relu(x, p_features_19_weight, p_features_19_bias, relu=True)
    x = run_conv2d_relu(x, p_features_21_weight, p_features_21_bias, relu=True)
    x = run_maxpool2d(x)
    
    # Block 5
    x = run_conv2d_relu(x, p_features_24_weight, p_features_24_bias, relu=True)
    x = run_conv2d_relu(x, p_features_26_weight, p_features_26_bias, relu=True)
    x = run_conv2d_relu(x, p_features_28_weight, p_features_28_bias, relu=True)
    x = run_maxpool2d(x)
    
    # Flatten
    x = x.reshape(x.shape[0], -1)
    
    # Classifier
    x = run_linear(x, p_classifier_0_weight, p_classifier_0_bias, relu=True)
    x = run_linear(x, p_classifier_3_weight, p_classifier_3_bias, relu=True)
    x = run_linear(x, p_classifier_6_weight, p_classifier_6_bias, relu=False)
    
    return x


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
    
    return [
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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    
    inputs = get_inputs()
    # Move to CUDA
    inputs = [t.cuda() for t in inputs]
    
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
                    nn.Linear(25088, 4096), nn.ReLU(inplace=True),
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
         x_in) = inputs
        
        with torch.no_grad():
            model.features[0].weight.data = p_features_0_weight
            model.features[0].bias.data = p_features_0_bias
            model.features[2].weight.data = p_features_2_weight
            model.features[2].bias.data = p_features_2_bias
            model.features[5].weight.data = p_features_5_weight
            model.features[5].bias.data = p_features_5_bias
            model.features[7].weight.data = p_features_7_weight
            model.features[7].bias.data = p_features_7_bias
            model.features[10].weight.data = p_features_10_weight
            model.features[10].bias.data = p_features_10_bias
            model.features[12].weight.data = p_features_12_weight
            model.features[12].bias.data = p_features_12_bias
            model.features[14].weight.data = p_features_14_weight