import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused Conv2d + Bias + ReLU kernel (implicit GEMM)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    # Tensor dimensions
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    # Kernel params
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    # Strides for X (NHWC layout)
    sx_n, sx_h, sx_w, sx_c,
    # Strides for W (KCRS layout -> stored as OC, IC, KH, KW)
    sw_oc, sw_ic, sw_kh, sw_kw,
    # Strides for Y (NHWC layout)
    sy_n, sy_h, sy_w, sy_c,
    # Total sizes for masking
    M, N, K,
    # Apply relu?
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

    # K = IN_C * KH * KW
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, ic)
        ic = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KW
        kh_idx = offs_k // (IN_C * KW)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw

        x_ptrs = X_ptr + batch_idx[:, None] * sx_n + ih * sx_h + iw * sx_w + ic[None, :] * sx_c
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> access [offs_n, ic, kh_idx, kw_idx]
        valid_n = offs_n[None, :] < N
        valid_k2 = offs_k[:, None] < K
        mask_w = valid_k2 & valid_n

        w_ptrs = W_ptr + offs_n[None, :] * sw_oc + ic[:, None] * sw_ic + kh_idx[:, None] * sw_kh + kw_idx[:, None] * sw_kw
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output (NHWC)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + batch_idx[:, None] * sy_n + out_h_idx[:, None] * sy_h + out_w_idx[:, None] * sy_w + offs_n[None, :] * sy_c
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_bias_relu(x_nhwc, weight, bias, stride_h, stride_w, pad_h, pad_w, apply_relu=True):
    """
    returns: (N, OH, OW, OC) contiguous
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OC, IC, KH, KW = weight.shape
    assert IC == IN_C

    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1

    M = N * OUT_H * OUT_W
    Nk = OC
    K = IN_C * KH * KW

    y = torch.empty((N, OUT_H, OUT_W, OC), device=x_nhwc.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(Nk, meta['BLOCK_N']))

    conv2d_bias_relu_kernel[grid](
        x_nhwc, weight, bias, y,
        N, IN_C, IN_H, IN_W,
        OC, OUT_H, OUT_W,
        KH, KW, stride_h, stride_w, pad_h, pad_w,
        # X strides (NHWC)
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        # W strides
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        # Y strides (NHWC)
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        M, Nk, K,
        APPLY_RELU=apply_relu,
    )
    return y


# ============================================================
# Max Pool 2D kernel (ceil_mode=True, kernel=3, stride=2)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    # X strides (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    # Y strides (NHWC)
    sy_n, sy_h, sy_w, sy_c,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # Decode: total = N * OUT_H * OUT_W * C
    c_idx = offs % C
    tmp = offs // C
    ow = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh = tmp2 % OUT_H
    n_idx = tmp2 // OUT_H

    result = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + n_idx * sx_n + ih * sx_h + iw * sx_w + c_idx * sx_c
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            result = tl.maximum(result, val)

    y_ptrs = Y_ptr + n_idx * sy_n + oh * sy_h + ow * sy_w + c_idx * sy_c
    tl.store(y_ptrs, result, mask=mask)


def maxpool2d_nhwc(x_nhwc, kernel_size=3, stride=2, padding=0, ceil_mode=True):
    N, IN_H, IN_W, C = x_nhwc.shape
    if ceil_mode:
        OUT_H = math.ceil((IN_H + 2 * padding - kernel_size) / stride) + 1
        OUT_W = math.ceil((IN_W + 2 * padding - kernel_size) / stride) + 1
    else:
        OUT_H = (IN_H + 2 * padding - kernel_size) // stride + 1
        OUT_W = (IN_W + 2 * padding - kernel_size) // stride + 1

    y = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)
    total = N * OUT_H * OUT_W * C

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x_nhwc, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kernel_size, kernel_size, stride, stride, padding, padding,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        total,
    )
    return y


# ============================================================
# Adaptive Average Pool 2D kernel (output 1x1)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 128}, num_warps=4),
        triton.Config({'BLOCK_N': 256}, num_warps=8),
        triton.Config({'BLOCK_N': 64}, num_warps=4),
        triton.Config({'BLOCK_N': 512}, num_warps=8),
    ],
    key=['N_total', 'HW'],
)
@triton.jit
def adaptive_avg_pool_1x1_kernel(
    X_ptr, Y_ptr,
    N_total, HW,
    sx_n, sx_h, sx_w, sx_c,
    C,
    BLOCK_N: tl.constexpr,
):
    # Each program handles one (batch, channel) pair
    pid = tl.program_id(0)
    offs = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = offs < N_total

    # offs indexes (batch * C + channel)
    c_idx = offs % C
    n_idx = offs // C

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    # Sum over all spatial positions
    # We'll do this with a loop over HW
    # But HW can be large, so we need to be careful
    # For simplicity, iterate over spatial positions
    # This is not ideal but correct
    for hw in range(HW):
        h = hw // tl.constexpr(1)  # placeholder, need actual H, W
        # Actually we need H and W separately
        pass

    # Better: pass H and W separately
    tl.store(Y_ptr + offs, acc / HW, mask=mask)


# Better adaptive avg pool
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 64}, num_warps=8),
        triton.Config({'BLOCK_C': 16}, num_warps=4),
        triton.Config({'BLOCK_C': 128}, num_warps=8),
    ],
    key=['NC', 'IN_H', 'IN_W'],
)
@triton.jit
def adaptive_avg_pool_1x1_kernel2(
    X_ptr, Y_ptr,
    N, C, IN_H, IN_W,
    sx_n, sx_h, sx_w, sx_c,
    NC,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = offs < NC

    c_idx = offs % C
    n_idx = offs // C

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    HW = IN_H * IN_W

    for h in range(IN_H):
        for w in range(IN_W):
            x_ptrs = X_ptr + n_idx * sx_n + h * sx_h + w * sx_w + c_idx * sx_c
            val = tl.load(x_ptrs, mask=mask, other=0.0)
            acc = acc + val

    acc = acc / HW
    tl.store(Y_ptr + offs, acc, mask=mask)


def adaptive_avg_pool_1x1_nhwc(x_nhwc):
    N, IN_H, IN_W, C = x_nhwc.shape
    y = torch.empty((N, 1, 1, C), device=x_nhwc.device, dtype=torch.float32)
    NC = N * C

    grid = lambda meta: (triton.cdiv(NC, meta['BLOCK_C']),)
    adaptive_avg_pool_1x1_kernel2[grid](
        x_nhwc, y,
        N, C, IN_H, IN_W,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        NC,
    )
    return y


# ============================================================
# Fire Module helper
# ============================================================

def fire_module_nhwc(x_nhwc, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """
    Returns: (N, H, W, C_out) NHWC where C_out = ex1_channels + ex3_channels
    """
    # Squeeze: 1x1 conv + relu
    sq = conv2d_bias_relu(x_nhwc, sq_w, sq_b, 1, 1, 0, 0, apply_relu=True)

    # Expand 1x1 + relu
    ex1 = conv2d_bias_relu(sq, ex1_w, ex1_b, 1, 1, 0, 0, apply_relu=True)

    # Expand 3x3 + relu (padding=1)
    ex3 = conv2d_bias_relu(sq, ex3_w, ex3_b, 1, 1, 1, 1, apply_relu=True)

    # Concatenate along channel dim
    return torch.cat([ex1, ex3], dim=3)


# ============================================================
# Main kernel function
# ============================================================

def kernel_function(
):
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [64, 512, 512, 3]

    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    # Output: [64, 253, 253, 96]
    out = conv2d_bias_relu(x_nhwc, p_features_0_weight, p_features_0_bias,
                           stride_h=2, stride_w=2, pad_h=0, pad_w=0, apply_relu=True)

    # features[2]: MaxPool2d(3, stride=2, ceil_mode=True)
    # Input: [64, 253, 253, 96] -> Output: [64, 126, 126, 96]
    out = maxpool2d_nhwc(out, kernel_size=3, stride=2, padding=0, ceil_mode=True)

    # features[3]: FireModule(96, 16, 64, 64)
    # Input: [64, 126, 126, 96] -> Output: [64, 126, 126, 128]
    out = fire_module_nhwc(out,
                           p_features_3_squeeze_weight, p_features_3_squeeze_bias,
                           p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,

    # features[4]: FireModule(128, 16, 64, 64)
    # Input: [64, 126, 126, 128] -> Output: [64, 126, 126, 128]
    out = fire_module_nhwc(out,
                           p_features_4_squeeze_weight, p_features_4_squeeze_bias,
                           p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,

    # features[5]: FireModule(128, 32, 128, 128)
    # Input: [64, 126, 126, 128] -> Output: [64, 126, 126, 256]
    out = fire_module_nhwc(out,
                           p_features_5_squeeze_weight, p_features_5_squeeze_bias,
                           p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,

    # features[6]: MaxPool2d(3, stride=2, ceil_mode=True)
    # Input: [64, 126, 126, 256] -> Output: [64, 63, 63, 256]
    out = maxpool2d_nhwc(out, kernel_size=3, stride=2, padding=0, ceil_mode=True)

    # features[7]: FireModule(256, 32, 128, 128)
    # Input: [64, 63, 63, 256] -> Output: [64, 63, 63, 256]
    out = fire_module_nhwc(out,
                           p_features_7_squeeze_weight, p_features_7_squeeze_bias,
                           p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,

    # features[8]: FireModule(256, 48, 192, 192)
    # Input: [64, 63, 63, 256] -> Output: [64, 63, 63, 384]
    out = fire_module_nhwc(out,
                           p_features_8_squeeze_weight, p_features_8_squeeze_bias,
                           p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,

    # features[9]: FireModule(384, 48, 192, 192)
    # Input: [64, 63, 63, 384] -> Output: [64, 63, 63, 384]
    out = fire_module_nhwc(out,
                           p_features_9_squeeze_weight, p_features_9_squeeze_bias,
                           p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,

    # features[10]: FireModule(384, 64, 256, 256)
    # Input: [64, 63, 63, 384] -> Output: [64, 63, 63, 512]
    out = fire_module_nhwc(out,
                           p_features_10_squeeze_weight, p_features_10_squeeze_bias,
                           p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,

    # features[11]: MaxPool2d(3, stride=2, ceil_mode=True)
    # Input: [64, 63, 63, 512] -> Output: [64, 31, 31, 512]
    out = maxpool2d_nhwc(out, kernel_size=3, stride=2, padding=0, ceil_mode=True)

    # features[12]: FireModule(512, 64, 256, 256)
    # Input: [64, 31, 31, 512] -> Output: [64, 31, 31, 512]
    out = fire_module_nhwc(out,
                           p_features_12_squeeze_weight, p_features_12_squeeze_bias,
                           p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,

    # classifier[0]: Dropout(p=0.0) -> identity
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    # Input: [64, 31, 31, 512] -> Output: [64, 31, 31, 1000]
    out = conv2d_bias_relu(out, p_classifier_1_weight, p_classifier_1_bias,
                           stride_h=1, stride_w=1, pad_h=0, pad_w=0, apply_relu=True)

    # classifier[3]: AdaptiveAvgPool2d(1, 1)
    # Input: [64, 31, 31, 1000] -> Output: [64, 1, 1, 1000]
    out = adaptive_avg_pool_1x1_nhwc(out)

    # flatten: [64, 1000]
    out = out.reshape(64, 1000)

    return out


def get_inputs():
    torch.manual_seed(42)
    return [
        torch.randn(96, 3, 7, 7, dtype=torch.float32, device='cuda'),    # p_features_0_weight
        torch.randn(96, dtype=torch.float32, device='cuda'),              # p_features_0_bias
        torch.randn(16, 96, 1, 1, dtype=torch.float32, device='cuda'),   # p_features_3_squeeze_weight
        torch.randn(16, dtype=torch.float32, device='cuda'),              # p_features_3_squeeze_bias
        torch.randn(64, 16, 1, 1, dtype=torch.float32, device='cuda'),   # p_features_3_expand1x1_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),              # p_features_3_expand1x1_bias