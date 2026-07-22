import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: bias + ReLU (pointwise epilogue)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
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
    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


# ============================================================
# Triton kernel: bias add (no relu)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def bias_add_kernel(
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
    result = x + bias
    tl.store(out_ptr + offsets, result, mask=mask)


# ============================================================
# Triton kernel: ReLU inplace
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(
    x_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    result = tl.where(x > 0.0, x, 0.0)
    tl.store(x_ptr + offsets, result, mask=mask)


# ============================================================
# Triton kernel: Max Pool 2D (kernel=2, stride=2)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def maxpool2d_kernel(
    x_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    BLOCK_SIZE: tl.constexpr,
):
    # Output layout: N, C, OH, OW (contiguous)
    # Input layout: N, C, IH, IW (contiguous)
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total
    
    # Decode output index
    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C
    
    # Input base
    ih0 = oh * 2
    iw0 = ow * 2
    
    # Load 4 values
    base = n * (C * IH * IW) + c * (IH * IW)
    
    v00 = tl.load(x_ptr + base + ih0 * IW + iw0, mask=mask, other=-float('inf'))
    v01 = tl.load(x_ptr + base + ih0 * IW + (iw0 + 1), mask=mask & (iw0 + 1 < IW), other=-float('inf'))
    v10 = tl.load(x_ptr + base + (ih0 + 1) * IW + iw0, mask=mask & (ih0 + 1 < IH), other=-float('inf'))
    v11 = tl.load(x_ptr + base + (ih0 + 1) * IW + (iw0 + 1), mask=mask & (ih0 + 1 < IH) & (iw0 + 1 < IW), other=-float('inf'))
    
    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    tl.store(out_ptr + offsets, result, mask=mask)


# ============================================================
# Triton kernel: Implicit GEMM Conv2d (NCHW input/output)
# For 3x3 conv with padding=1, stride=1
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=2, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_nchw_kernel(
    X_ptr, W_ptr, out_ptr,
    N_batch, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    M, N, K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = N_batch * OUT_H * OUT_W
    # N = OUT_C
    # K = IN_C * KERNEL_H * KERNEL_W
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
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (in_c, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c = tmp_k // KERNEL_H
        
        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = (batch < N_batch) & (offs_m < M)
        valid_k = offs_k < K
        valid = valid_m[:, None] & valid_k[None, :] & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        
        x_ptrs = X_ptr + batch[:, None] * stride_xn + in_c[None, :] * stride_xc + ih * stride_xh + iw * stride_xw
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)
        
        # Weight: [OUT_C, IN_C, KH, KW] -> [K, N] = [IN_C*KH*KW, OUT_C]
        valid_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * stride_wco + in_c[:, None] * stride_wci + kh[:, None] * stride_wkh + kw[:, None] * stride_wkw
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Store output in NCHW format
    # out[batch, out_c, out_h, out_w] = acc
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C) & (batch[:, None] < N_batch)
    out_idx = batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=valid_out)


# ============================================================
# Triton kernel: Linear (GEMM) - M x K @ K x N
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    A_ptr, B_ptr, bias_ptr, out_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    HAS_BIAS: tl.constexpr,
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
        
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        b_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        
        acc = tl.dot(a, b, acc, allow_tf32=False)
    
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]
    
    if DO_RELU:
        acc = tl.where(acc > 0.0, acc, 0.0)
    
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_conv2d(x, weight, bias, stride=1, padding=1):
    """Triton conv2d with bias+relu fused separately"""
    N_batch, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    
    STRIDE_H = stride if isinstance(stride, int) else stride[0]
    STRIDE_W = stride if isinstance(stride, int) else stride[1]
    PAD_H = padding if isinstance(padding, int) else padding[0]
    PAD_W = padding if isinstance(padding, int) else padding[1]
    
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1
    
    M = N_batch * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KH * KW
    
    out = torch.empty((N_batch, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    conv2d_nchw_kernel[grid](
        x, weight, out,
        N_batch, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        M, N, K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    )
    
    # Add bias
    if bias is not None:
        n_elements = out.numel()
        n_channels = OUT_C
        # bias is per output channel, out is NCHW
        # We need to add bias[c] to out[:, c, :, :]
        # Reshape to do this efficiently
        out_flat = out.view(N_batch, OUT_C, -1)  # N, C, H*W
        bias_view = bias.view(1, OUT_C, 1)
        out_flat.add_(bias_view)
    
    return out


def triton_maxpool2d(x):
    N, C, IH, IW = x.shape
    OH = IH // 2
    OW = IW // 2
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](x, out, N, C, IH, IW, OH, OW)
    return out


def triton_relu_inplace(x):
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, n_elements)
    return x


def triton_linear(x, weight, bias, do_relu=False):
    # x: [M, K], weight: [N, K] (standard linear weight shape)
    M, K = x.shape
    N = weight.shape[0]
    
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    linear_kernel[grid](
        out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight is [N, K], so stride_bk=stride(1), stride_bn=stride(0)
        bias is not None,
        do_relu,
    )
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
    x = triton_conv2d(x, p_features_0_weight, p_features_0_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_conv2d(x, p_features_2_weight, p_features_2_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_maxpool2d(x)
    
    # Block 2
    x = triton_conv2d(x, p_features_5_weight, p_features_5_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_conv2d(x, p_features_7_weight, p_features_7_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_maxpool2d(x)
    
    # Block 3
    x = triton_conv2d(x, p_features_10_weight, p_features_10_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_conv2d(x, p_features_12_weight, p_features_12_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_conv2d(x, p_features_14_weight, p_features_14_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_maxpool2d(x)
    
    # Block 4
    x = triton_conv2d(x, p_features_17_weight, p_features_17_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_conv2d(x, p_features_19_weight, p_features_19_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_conv2d(x, p_features_21_weight, p_features_21_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_maxpool2d(x)
    
    # Block 5
    x = triton_conv2d(x, p_features_24_weight, p_features_24_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_conv2d(x, p_features_26_weight, p_features_26_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_conv2d(x, p_features_28_weight, p_features_28_bias, stride=1, padding=1)
    x = triton_relu_inplace(x)
    x = triton_maxpool2d(x)
    
    # Flatten
    x = x.reshape(x.shape[0], -1)
    
    # Classifier
    x = triton_linear(x, p_classifier_0_weight, p_classifier_0_bias, do_relu=True)
    # dropout p=0.0 is identity
    x = triton_linear(x, p_classifier_3_weight, p_classifier_3_bias, do_relu=True)
    # dropout p=0.0 is identity
    x = triton_linear(x, p_classifier_6_weight, p_classifier_6_bias, do_relu=False)
    
    return x


def get_inputs():
    # Weights
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
    p_features_17_weight = torch.randn(512, 256, 3, 3, device='cuda')
    p_features_17_bias = torch.randn(512, device='cuda')
    p_features_19_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_19_bias = torch.randn(512, device='cuda')
    p_features_21_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_21_bias = torch.randn(512, device='cuda')
    p_features_24_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_24_bias = torch.randn(512, device='cuda')
    p_features_26_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_26_bias = torch.randn(512, device='cuda')
    p_features_28_weight = torch.randn(512, 512, 3, 3, device='cuda')
    p_features_28_bias = torch.randn(512, device='cuda')
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
        p_features_17_weight, p_features_17_bias,
        p_features_19_weight, p_features_19_bias,
        p_features_21_weight, p_features_21_bias,
        p_features_24_weight, p_features_24_bias,
        p_features_26_weight, p_features_26_bias,
        p_features_28_weight, p_features_28_bias,
        p_classifier_0_weight, p_classifier_0_bias,
        p_classifier_3_weight, p_classifier_3_bias,
        p_classifier_6_weight, p_classifier_6_bias,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")