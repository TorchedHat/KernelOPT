import torch
import triton
import triton.language as tl
import math



import torch
import triton
import triton.language as tl
import math

# ============================================================
# Generic Conv2d kernel (implicit GEMM, NHWC layout internally)
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
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # X strides (NHWC)
    s_xn, s_xh, s_xw, s_xc,
    # W strides (out_c, in_c, kh, kw) -> stored as (out_c, kh, kw, in_c) for NHWC
    s_wo, s_wkh, s_wkw, s_wi,
    # Y strides (NHWC)
    s_yn, s_yh, s_yw, s_yc,
    APPLY_RELU: tl.constexpr,
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
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # K = IN_C * KERNEL_H * KERNEL_W
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KERNEL_W
        kh = tmp_k // KERNEL_W
        
        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W
        
        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw
        
        x_ptrs = (X_ptr 
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        # W shape: (OUT_C, KERNEL_H, KERNEL_W, IN_C) for NHWC-friendly access
        # offs_k encodes (kh, kw, in_c)
        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n
        w_ptrs = (W_ptr 
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]
    
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)
    
    # Store output (NHWC)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr 
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=3, num_warps=4),
    ],
    key=['N_total'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W,
    # X strides (NHWC)
    s_xn, s_xh, s_xw, s_xc,
    # Y strides (NHWC)
    s_yn, s_yh, s_yw, s_yc,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total
    
    # Decode: N_total = BATCH * OUT_H * OUT_W * C
    c_idx = offs % C
    tmp = offs // C
    ow = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh = tmp2 % OUT_H
    n = tmp2 // OUT_H
    
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh
            iw = ow * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W) & (n < BATCH)
            x_ptrs = X_ptr + n * s_xn + ih * s_xh + iw * s_xw + c_idx * s_xc
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)
    
    y_ptrs = Y_ptr + n * s_yn + oh * s_yh + ow * s_yw + c_idx * s_yc
    tl.store(y_ptrs, max_val, mask=mask)


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
def linear_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_ptrs = W_ptr + offs_n[None, :] * K + offs_k[:, None]
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)
    
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)
    
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_relu(x_nhwc, w_nchw, bias, stride=1, padding=1, apply_relu=True):
    """x_nhwc: (N, H, W, C), w_nchw: (OUT_C, IN_C, KH, KW)"""
    BATCH, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KERNEL_H, KERNEL_W = w_nchw.shape
    STRIDE_H = STRIDE_W = stride
    PAD_H = PAD_W = padding
    
    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1
    
    M = BATCH * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KERNEL_H * KERNEL_W
    
    # Rearrange W to (OUT_C, KH, KW, IN_C) for easier indexing
    # w_nchw: (OUT_C, IN_C, KH, KW) -> (OUT_C, KH, KW, IN_C)
    w_nhwc = w_nchw.permute(0, 2, 3, 1).contiguous()
    
    y_nhwc = torch.empty((BATCH, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)
    
    # X strides (NHWC)
    s_xn, s_xh, s_xw, s_xc = x_nhwc.stride()
    # W strides (OUT_C, KH, KW, IN_C)
    s_wo, s_wkh, s_wkw, s_wi = w_nhwc.stride()
    # Y strides (NHWC)
    s_yn, s_yh, s_yw, s_yc = y_nhwc.stride()
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    conv2d_relu_kernel[grid](
        x_nhwc, w_nhwc, bias, y_nhwc,
        M, N, K,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xh, s_xw, s_xc,
        s_wo, s_wkh, s_wkw, s_wi,
        s_yn, s_yh, s_yw, s_yc,
        APPLY_RELU=apply_relu,
    )
    return y_nhwc


def maxpool2d(x_nhwc, kernel_size=2, stride=2):
    BATCH, IN_H, IN_W, C = x_nhwc.shape
    KH = KW = kernel_size
    STRIDE_H = STRIDE_W = stride
    OUT_H = (IN_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W - KW) // STRIDE_W + 1
    
    y_nhwc = torch.empty((BATCH, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)
    
    N_total = BATCH * OUT_H * OUT_W * C
    
    s_xn, s_xh, s_xw, s_xc = x_nhwc.stride()
    s_yn, s_yh, s_yw, s_yc = y_nhwc.stride()
    
    grid = lambda meta: (triton.cdiv(N_total, meta['BLOCK_SIZE']),)
    
    maxpool2d_kernel[grid](
        x_nhwc, y_nhwc,
        BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW, STRIDE_H, STRIDE_W,
        s_xn, s_xh, s_xw, s_xc,
        s_yn, s_yh, s_yw, s_yc,
        N_total,
    )
    return y_nhwc


def linear_relu(x, w, b, apply_relu=True):
    """x: (M, K), w: (N, K), b: (N,) -> (M, N)"""
    M, K = x.shape
    N = w.shape[0]
    
    x_cont = x.contiguous()
    w_cont = w.contiguous()
    
    y = torch.empty((M, N), device=x.device, dtype=torch.float32)
    
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    
    linear_relu_kernel[grid](
        x_cont, w_cont, b, y,
        M, N, K,
        APPLY_RELU=apply_relu,
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
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    
    # Block 1
    x_nhwc = conv2d_relu(x_nhwc, p_features_0_weight, p_features_0_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = conv2d_relu(x_nhwc, p_features_2_weight, p_features_2_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = maxpool2d(x_nhwc, kernel_size=2, stride=2)
    
    # Block 2
    x_nhwc = conv2d_relu(x_nhwc, p_features_5_weight, p_features_5_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = conv2d_relu(x_nhwc, p_features_7_weight, p_features_7_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = maxpool2d(x_nhwc, kernel_size=2, stride=2)
    
    # Block 3
    x_nhwc = conv2d_relu(x_nhwc, p_features_10_weight, p_features_10_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = conv2d_relu(x_nhwc, p_features_12_weight, p_features_12_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = conv2d_relu(x_nhwc, p_features_14_weight, p_features_14_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = conv2d_relu(x_nhwc, p_features_16_weight, p_features_16_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = maxpool2d(x_nhwc, kernel_size=2, stride=2)
    
    # Block 4
    x_nhwc = conv2d_relu(x_nhwc, p_features_19_weight, p_features_19_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = conv2d_relu(x_nhwc, p_features_21_weight, p_features_21_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = conv2d_relu(x_nhwc, p_features_23_weight, p_features_23_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = conv2d_relu(x_nhwc, p_features_25_weight, p_features_25_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = maxpool2d(x_nhwc, kernel_size=2, stride=2)
    
    # Block 5
    x_nhwc = conv2d_relu(x_nhwc, p_features_28_weight, p_features_28_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = conv2d_relu(x_nhwc, p_features_30_weight, p_features_30_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = conv2d_relu(x_nhwc, p_features_32_weight, p_features_32_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = conv2d_relu(x_nhwc, p_features_34_weight, p_features_34_bias, stride=1, padding=1, apply_relu=True)
    x_nhwc = maxpool2d(x_nhwc, kernel_size=2, stride=2)
    
    # Flatten: (N, H, W, C) -> (N, H*W*C)
    BATCH = x_nhwc.shape[0]
    x_flat = x_nhwc.reshape(BATCH, -1).contiguous()
    
    # Classifier
    x_flat = linear_relu(x_flat, p_classifier_0_weight, p_classifier_0_bias, apply_relu=True)
    # dropout p=0.0 is identity
    x_flat = linear_relu(x_flat, p_classifier_3_weight, p_classifier_3_bias, apply_relu=True)
    # dropout p=0.0 is identity
    x_flat = linear_relu(x_flat, p_classifier_6_weight, p_classifier_6_bias, apply_relu=False)
    
    return x_flat


def get_inputs():
    torch.manual_seed(42)
    return [
        torch.randn(64, 3, 3, 3, dtype=torch.float32, device='cuda'),   # p_features_0_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),              # p_features_0_bias
        torch.randn(64, 64, 3, 3, dtype=torch.float32, device='cuda'),  # p_features_2_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),              # p_features_2_bias
        torch.randn(128, 64, 3, 3, dtype=torch.float32, device='cuda'), # p_features_5_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),             # p_features_5_bias
        torch.randn(128, 128, 3, 3, dtype=torch.float32, device='cuda'),# p_features_7_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),             # p_features_7_bias
        torch.randn(256, 128, 3, 3, dtype=torch.float32, device='cuda'),# p_features_10_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),             # p_features_10_bias
        torch.randn(256, 256, 3, 3, dtype=torch.float32, device='cuda'),# p_features_12_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),             # p_features_12_bias
        torch.randn(256, 256, 3, 3, dtype=torch.float32, device='cuda'),# p_features_14_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),             # p_features_14_bias
        torch.randn(256, 256, 3, 3, dtype=torch.float32, device='cuda'),# p_features_16_weight
        torch.randn(256, dtype=torch.float32, device='cuda'),             # p_features_16_bias
        torch.randn(512, 256, 3, 3, dtype=torch.float32, device='cuda'),# p_features_19_weight
        torch.randn(512, dtype=torch.float32, device='cuda'),             # p_features_19_bias
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),# p_features_21_weight
        torch.randn(512, dtype=torch.float32, device='cuda'),             # p_features_21_bias
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),# p_features_23_weight
        torch.randn(512, dtype=torch.float32, device='cuda'),             # p_features_23_bias
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),# p_features_25_weight
        torch.randn(512, dtype=torch.float32, device='cuda'),             # p_features_25_bias
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),# p_features_28_weight
        torch.randn(512, dtype=torch.float32, device='cuda'),             # p_features_28_bias
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),# p_features_30_weight
        torch.randn(512, dtype=torch.float32, device='cuda'),             # p_features_30_bias
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),# p_features_32_weight
        torch.randn(512, dtype=torch.float32, device='cuda'),             # p_features_32_bias
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),# p_features_34_weight
        torch.randn(512, dtype=torch.float32, device='cuda'),             # p_features_34_bias
        torch.randn(4096, 25088, dtype=torch.float32, device='cuda'),   # p_classifier_0_weight
        torch.randn(4096, dtype=torch.float32, device='cuda'),            # p_classifier_0_bias
        torch.randn(4096, 4096, dtype=torch.float32, device='cuda'),    # p_classifier_3_weight
        torch.randn(4096, dtype=torch.float32, device='cuda'),            # p_classifier_3_bias
        torch.randn(1000, 4096, dtype=torch.float32, device='cuda'),    # p_classifier_6_weight
        torch.randn(1000, dtype=torch.float32, device='cuda'),            # p_classifier_6_bias
        torch.randn(10, 3, 224, 224, dtype=torch.float32, device='cuda'), # x
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
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv