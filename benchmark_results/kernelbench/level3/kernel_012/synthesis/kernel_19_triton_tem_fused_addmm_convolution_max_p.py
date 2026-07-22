import torch
import triton
import triton.language as tl
import math

# [Full implementation follows]


import torch
import triton
import triton.language as tl
import math
import argparse

# Triton GEMM kernel for linear layers (M x K) x (K x N) + bias -> ReLU
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    HAS_BIAS: tl.constexpr,
    DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        x = tl.load(x_ptrs, mask=(offs_m[:, None] < M) & mask_k[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=mask_k[:, None] & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(x, w, acc)
        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    if HAS_BIAS:
        bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    if DO_RELU:
        acc = tl.maximum(acc, 0.0)

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# Conv2d as implicit GEMM kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # X strides (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # W strides (out_c, in_c, kh, kw) -> stored as (out_c, kh, kw, in_c) for NHWC
    s_wo, s_wkh, s_wkw, s_wi,
    # Y strides (NHWC layout)
    s_yn, s_yh, s_yw, s_yc,
    HAS_BIAS: tl.constexpr,
    DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m[:, None] & valid_k[None, :] & valid_ih & valid_iw

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: shape (OUT_C, KERNEL_H, KERNEL_W, IN_C) in NHWC-friendly layout
        # w_ptrs: [BLOCK_K, BLOCK_N]
        mask_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + bias[None, :]

    if DO_RELU:
        acc = tl.maximum(acc, 0.0)

    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# Max pool 2x2 stride 2 kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=5, num_warps=2),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    s_xn, s_xh, s_xw, s_xc,
    s_yn, s_yh, s_yw, s_yc,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: N_TOTAL = BATCH * OUT_H * OUT_W * C
    c_idx = offs % C
    tmp = offs // C
    ow = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh = tmp2 % OUT_H
    bn = tmp2 // OUT_H

    ih0 = oh * 2
    iw0 = ow * 2

    # Load 4 values for 2x2 pool
    def load_val(dh, dw):
        ih = ih0 + dh
        iw = iw0 + dw
        valid = mask & (ih < IN_H) & (iw < IN_W)
        ptrs = X_ptr + bn * s_xn + ih * s_xh + iw * s_xw + c_idx * s_xc
        return tl.load(ptrs, mask=valid, other=float('-inf'))

    v00 = load_val(0, 0)
    v01 = load_val(0, 1)
    v10 = load_val(1, 0)
    v11 = load_val(1, 1)

    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    y_ptrs = Y_ptr + bn * s_yn + oh * s_yh + ow * s_yw + c_idx * s_yc
    tl.store(y_ptrs, result, mask=mask)


def run_conv2d_relu(x_nhwc, w, b, out_h, out_w, stride_h, stride_w, pad_h, pad_w, do_relu):
    """Run conv2d with optional ReLU. x_nhwc: (N,H,W,C), w: (OC,IC,KH,KW)"""
    N, H, W, IC = x_nhwc.shape
    OC, _, KH, KW = w.shape

    # Rearrange weight to (OC, KH, KW, IC) for NHWC-friendly access
    w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # (OC, KH, KW, IC)

    y = torch.empty((N, out_h, out_w, OC), device=x_nhwc.device, dtype=x_nhwc.dtype)

    M_TOTAL = N * out_h * out_w
    grid = lambda meta: (triton.cdiv(M_TOTAL, meta['BLOCK_M']), triton.cdiv(OC, meta['BLOCK_N']))

    has_bias = b is not None
    if not has_bias:
        b_dummy = torch.empty(1, device=x_nhwc.device, dtype=x_nhwc.dtype)
    else:
        b_dummy = b

    conv2d_relu_kernel[grid](
        x_nhwc, w_nhwc, b_dummy, y,
        N, IC, H, W,
        OC, out_h, out_w,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w_nhwc.stride(0), w_nhwc.stride(1), w_nhwc.stride(2), w_nhwc.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        HAS_BIAS=has_bias,
        DO_RELU=do_relu,
    )
    return y


def run_maxpool2d(x_nhwc):
    """2x2 max pool with stride 2. x_nhwc: (N,H,W,C)"""
    N, H, W, C = x_nhwc.shape
    out_h = H // 2
    out_w = W // 2
    y = torch.empty((N, out_h, out_w, C), device=x_nhwc.device, dtype=x_nhwc.dtype)
    N_TOTAL = N * out_h * out_w * C
    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x_nhwc, y,
        N, C, H, W, out_h, out_w,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        N_TOTAL,
    )
    return y


def run_linear_relu(x, w, b, do_relu):
    """x: (M, K), w: (N, K), b: (N,) -> (M, N)"""
    M, K = x.shape
    N = w.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    has_bias = b is not None
    if not has_bias:
        b_dummy = torch.empty(1, device=x.device, dtype=x.dtype)
    else:
        b_dummy = b
    linear_relu_kernel[grid](
        x, w, b_dummy, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        HAS_BIAS=has_bias,
        DO_RELU=do_relu,
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
):
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # Block 1
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_0_weight, p_features_0_bias, 224, 224, 1, 1, 1, 1, True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_2_weight, p_features_2_bias, 224, 224, 1, 1, 1, 1, True)
    x_nhwc = run_maxpool2d(x_nhwc)

    # Block 2
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_5_weight, p_features_5_bias, 112, 112, 1, 1, 1, 1, True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_7_weight, p_features_7_bias, 112, 112, 1, 1, 1, 1, True)
    x_nhwc = run_maxpool2d(x_nhwc)

    # Block 3
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_10_weight, p_features_10_bias, 56, 56, 1, 1, 1, 1, True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_12_weight, p_features_12_bias, 56, 56, 1, 1, 1, 1, True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_14_weight, p_features_14_bias, 56, 56, 1, 1, 1, 1, True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_16_weight, p_features_16_bias, 56, 56, 1, 1, 1, 1, True)
    x_nhwc = run_maxpool2d(x_nhwc)

    # Block 4
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_19_weight, p_features_19_bias, 28, 28, 1, 1, 1, 1, True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_21_weight, p_features_21_bias, 28, 28, 1, 1, 1, 1, True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_23_weight, p_features_23_bias, 28, 28, 1, 1, 1, 1, True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_25_weight, p_features_25_bias, 28, 28, 1, 1, 1, 1, True)
    x_nhwc = run_maxpool2d(x_nhwc)

    # Block 5
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_28_weight, p_features_28_bias, 14, 14, 1, 1, 1, 1, True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_30_weight, p_features_30_bias, 14, 14, 1, 1, 1, 1, True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_32_weight, p_features_32_bias, 14, 14, 1, 1, 1, 1, True)
    x_nhwc = run_conv2d_relu(x_nhwc, p_features_34_weight, p_features_34_bias, 14, 14, 1, 1, 1, 1, True)
    x_nhwc = run_maxpool2d(x_nhwc)

    # Flatten: (N, 7, 7, 512) -> (N, 25088)
    N = x_nhwc.shape[0]
    # Convert back to NCHW then flatten
    x_nchw = x_nhwc.permute(0, 3, 1, 2).contiguous()
    x_flat = x_nchw.reshape(N, -1)

    # Classifier
    x_flat = run_linear_relu(x_flat, p_classifier_0_weight, p_classifier_0_bias, True)
    # dropout p=0.0 is identity
    x_flat = run_linear_relu(x_flat, p_classifier_3_weight, p_classifier_3_bias, True)
    # dropout p=0.0 is identity
    x_flat = run_linear_relu(x_flat, p_classifier_6_weight, p_classifier_6_bias, False)

    return x_flat


def get_inputs():
    torch.manual_seed(42)
    return [
        torch.randn(64, 3, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(64, dtype=torch.float32, device='cuda'),
        torch.randn(64, 64, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(64, dtype=torch.float32, device='cuda'),
        torch.randn(128, 64, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(128, dtype=torch.float32, device='cuda'),
        torch.randn(128, 128, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(128, dtype=torch.float32, device='cuda'),
        torch.randn(256, 128, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(256, dtype=torch.float32, device='cuda'),
        torch.randn(256, 256, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(256, dtype=torch.float32, device='cuda'),
        torch.randn(256, 256, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(256, dtype=torch.float32, device='cuda'),
        torch.randn(256, 256, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(256, dtype=torch.float32, device='cuda'),
        torch.randn(512, 256, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(512, dtype=torch.float32, device='cuda'),
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(512, dtype=torch.float32, device='cuda'),
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(512, dtype=torch.float32, device='cuda'),
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(512, dtype=torch.float32, device='cuda'),
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(512, dtype=torch.float32, device='cuda'),
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(512, dtype=torch.float32, device='cuda'),
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(512, dtype=torch.float32, device='cuda'),
        torch.randn(512, 512, 3, 3, dtype=torch.float32, device='cuda'),
        torch.randn(512, dtype=torch.float32, device='cuda'),
        torch.randn(4096, 25088, dtype=torch.float32, device='cuda'),
        torch.randn(4096, dtype=torch.float32, device='cuda'),
        torch.randn(4096, 4096, dtype=torch.float32, device='cuda'),
        torch.randn(4096, dtype=torch.float32, device='cuda'),
        torch.randn(1000, 4096, dtype=torch.float32, device='cuda'),
        torch.randn(1000, dtype=torch.float32, device='cuda'),
        torch.randn(10, 3, 224, 224, dtype=torch.float32, device='cuda'),
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn
        import torch.nn.functional as F

        # Reference implementation
        (p_features_0_weight, p_features_0_bias,
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