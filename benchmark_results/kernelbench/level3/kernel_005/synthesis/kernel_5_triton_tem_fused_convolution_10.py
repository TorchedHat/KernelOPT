import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Triton kernels
# ─────────────────────────────────────────────────────────────────────────────

# ── 1. Implicit-GEMM convolution (NHWC layout internally) ────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N, K,
    # strides for X (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (Co, Ci, KH, KW) -> stored as (Co, KH, KW, Ci) for NHWC
    sw_o, sw_kh, sw_kw, sw_i,
    # strides for Y (NHWC)
    sy_n, sy_h, sy_w, sy_c,
    HAS_BIAS: tl.constexpr,
    RELU: tl.constexpr,
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

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KW
        kh_idx = tmp_k // KW

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m < M
        valid_k = offs_k < K
        valid = (
            valid_m[:, None] &
            valid_k[None, :] &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W)
        )

        x_ptrs = (
            batch_idx[:, None] * sx_n +
            in_c_idx[None, :] * sx_c
        )
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Weight: shape (OUT_C, KH, KW, IN_C) in NHWC-friendly layout
        # offs_n -> out_c, offs_k -> (kh, kw, in_c)
        valid_n = offs_n < N
        w_ptrs = (
            offs_n[None, :] * sw_o +
            kh_idx[:, None] * sw_kh +
            kw_idx[:, None] * sw_kw +
            in_c_idx[:, None] * sw_i
        )
        w_mask = valid_k[:, None] & valid_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    # ReLU
    if RELU:
        acc = tl.maximum(acc, 0.0)

    # Store
    valid_m = offs_m < M
    valid_n = offs_n < N
    mask_out = valid_m[:, None] & valid_n[None, :]

    y_ptrs = (
        batch_idx[:, None] * sy_n +
        out_h_idx[:, None] * sy_h +
        out_w_idx[:, None] * sy_w +
        offs_n[None, :] * sy_c
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ── 2. Max Pool 2D (NHWC) ────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['total'],
)
@triton.jit
def maxpool2d_nhwc_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # Decode: total = BATCH * OUT_H * OUT_W * C
    c_idx = offs % C
    tmp = offs // C
    ow = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh = tmp2 % OUT_H
    n = tmp2 // OUT_H

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * STRIDE_H + kh - PAD_H
            iw = ow * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + n * (IN_H * IN_W * C) + ih * (IN_W * C) + iw * C + c_idx
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    y_ptrs = Y_ptr + n * (OUT_H * OUT_W * C) + oh * (OUT_W * C) + ow * C + c_idx
    tl.store(y_ptrs, max_val, mask=mask)


# ── 3. Fused GEMM + Bias + ReLU (for FC layers) ──────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_relu_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    HAS_BIAS: tl.constexpr,
    RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    a_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        a_mask = (offs_m[:, None] < M) & (offs_k[None, :] < k_remaining)
        b_mask = (offs_k[:, None] < k_remaining) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if HAS_BIAS:
        bias = tl.load(Bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    if RELU:
        acc = tl.maximum(acc, 0.0)

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run one conv2d layer (NCHW input -> NHWC internal -> NCHW output)
# ─────────────────────────────────────────────────────────────────────────────
def _conv2d_relu(x_nhwc, w_nchw, bias, stride_h, stride_w, pad_h, pad_w, relu=True):
    """
    returns: (N, OH, OW, C_out) contiguous
    """
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w_nchw.shape

    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1

    # Rearrange weight to (OUT_C, KH, KW, IN_C) for NHWC-friendly access
    w_ohwi = w_nchw.permute(0, 2, 3, 1).contiguous()  # (OUT_C, KH, KW, IN_C)

    M = N * OUT_H * OUT_W
    Nmat = OUT_C
    K = KH * KW * IN_C

    y_nhwc = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    # Strides for x_nhwc
    sx_n = IN_H * IN_W * IN_C
    sx_h = IN_W * IN_C
    sx_w = IN_C
    sx_c = 1

    # Strides for w_ohwi (OUT_C, KH, KW, IN_C)
    sw_o = KH * KW * IN_C
    sw_kh = KW * IN_C
    sw_kw = IN_C
    sw_i = 1

    # Strides for y_nhwc
    sy_n = OUT_H * OUT_W * OUT_C
    sy_h = OUT_W * OUT_C
    sy_w = OUT_C
    sy_c = 1

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Nmat, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_ohwi, bias, y_nhwc,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        M, Nmat, K,
        sx_n, sx_h, sx_w, sx_c,
        sw_o, sw_kh, sw_kw, sw_i,
        sy_n, sy_h, sy_w, sy_c,
        HAS_BIAS=(bias is not None),
        RELU=relu,
    )
    return y_nhwc


def _maxpool2d_nhwc(x_nhwc, kh, kw, stride_h, stride_w, pad_h=0, pad_w=0):
    N, IN_H, IN_W, C = x_nhwc.shape
    OUT_H = (IN_H + 2 * pad_h - kh) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - kw) // stride_w + 1
    y = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)
    total = N * OUT_H * OUT_W * C
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_nhwc_kernel[grid](
        x_nhwc, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kh, kw, stride_h, stride_w, pad_h, pad_w,
        total,
    )
    return y


def _fc(x, weight, bias, relu=False):
    """
    """
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    gemm_bias_relu_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight is (N,K), we want K-major for B
        out.stride(0), out.stride(1),
        HAS_BIAS=(bias is not None),
        RELU=relu,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel function
# ─────────────────────────────────────────────────────────────────────────────
def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_conv3_weight, p_conv3_bias,
    p_conv4_weight, p_conv4_bias,
    p_conv5_weight, p_conv5_bias,
    p_fc1_weight, p_fc1_bias,
    p_fc2_weight, p_fc2_bias,
    p_fc3_weight, p_fc3_bias,
    x,
):
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, 224, 224, 3)

    # Conv1: (3->96, 11x11, stride=4, pad=2) + ReLU
    x_nhwc = _conv2d_relu(x_nhwc, p_conv1_weight, p_conv1_bias,
                           stride_h=4, stride_w=4, pad_h=2, pad_w=2, relu=True)
    # MaxPool1: 3x3, stride=2
    x_nhwc = _maxpool2d_nhwc(x_nhwc, 3, 3, 2, 2)

    # Conv2: (96->256, 5x5, stride=1, pad=2) + ReLU
    x_nhwc = _conv2d_relu(x_nhwc, p_conv2_weight, p_conv2_bias,
                           stride_h=1, stride_w=1, pad_h=2, pad_w=2, relu=True)
    # MaxPool2: 3x3, stride=2
    x_nhwc = _maxpool2d_nhwc(x_nhwc, 3, 3, 2, 2)

    # Conv3: (256->384, 3x3, stride=1, pad=1) + ReLU
    x_nhwc = _conv2d_relu(x_nhwc, p_conv3_weight, p_conv3_bias,
                           stride_h=1, stride_w=1, pad_h=1, pad_w=1, relu=True)

    # Conv4: (384->384, 3x3, stride=1, pad=1) + ReLU
    x_nhwc = _conv2d_relu(x_nhwc, p_conv4_weight, p_conv4_bias,
                           stride_h=1, stride_w=1, pad_h=1, pad_w=1, relu=True)

    # Conv5: (384->256, 3x3, stride=1, pad=1) + ReLU
    x_nhwc = _conv2d_relu(x_nhwc, p_conv5_weight, p_conv5_bias,
                           stride_h=1, stride_w=1, pad_h=1, pad_w=1, relu=True)
    # MaxPool3: 3x3, stride=2
    x_nhwc = _maxpool2d_nhwc(x_nhwc, 3, 3, 2, 2)

    # Flatten: (N, 6, 6, 256) -> (N, 9216)
    N = x_nhwc.shape[0]
    x_flat = x_nhwc.reshape(N, -1).contiguous()

    # FC1: 9216 -> 4096 + ReLU
    x_flat = _fc(x_flat, p_fc1_weight, p_fc1_bias, relu=True)

    # FC2: 4096 -> 4096 + ReLU
    x_flat = _fc(x_flat, p_fc2_weight, p_fc2_bias, relu=True)

    # FC3: 4096 -> 1000
    x_flat = _fc(x_flat, p_fc3_weight, p_fc3_bias, relu=False)

    return x_flat


# ─────────────────────────────────────────────────────────────────────────────
# get_inputs
# ─────────────────────────────────────────────────────────────────────────────
def get_inputs():
    batch_size = 4  # Use small batch for testing (1024 would OOM on many GPUs)
    device = 'cuda'
    return [
        torch.randn(96, 3, 11, 11, device=device),    # p_conv1_weight
        torch.randn(96, device=device),                 # p_conv1_bias
        torch.randn(256, 96, 5, 5, device=device),     # p_conv2_weight
        torch.randn(256, device=device),                # p_conv2_bias
        torch.randn(384, 256, 3, 3, device=device),    # p_conv3_weight
        torch.randn(384, device=device),                # p_conv3_bias
        torch.randn(384, 384, 3, 3, device=device),    # p_conv4_weight
        torch.randn(384, device=device),                # p_conv4_bias
        torch.randn(256, 384, 3, 3, device=device),    # p_conv5_weight
        torch.randn(256, device=device),                # p_conv5_bias
        torch.randn(4096, 9216, device=device),         # p_fc1_weight
        torch.randn(4096, device=device),               # p_fc1_bias
        torch.randn(4096, 4096, device=device),         # p_fc2_weight
        torch.randn(4096, device=device),               # p_fc2_bias
        torch.randn(1000, 4096, device=device),         # p_fc3_weight
        torch.randn(1000, device=device),               # p_fc3_bias
        torch.randn(batch_size, 3, 224, 224, device=device),  # x
    ]


# ─────────────────────────────────────────────────────────────────────────────
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
        device = 'cuda'
        batch_size = inputs[-1].shape[0]

        (p_conv1_weight, p_conv1_bias,
         p_conv2_weight, p_conv2_bias,
         p_conv3_weight, p_conv3_bias,
         p_conv4_weight, p_conv4_bias,
         p_conv5_weight, p_conv5_bias,
         p_fc1_weight, p_fc1_bias,
         p_fc2_weight, p_fc2_bias,
         p_fc3_weight, p_fc3_bias,
         x) = inputs

        # Reference computation
        ref = x
        ref = F.conv2d(ref, p_conv1_weight, p_conv1_bias, stride=4, padding=2)
        ref = F.relu(ref)
        ref = F.max_pool2d(ref, kernel_size=3, stride=2)

        ref = F.conv2d(ref, p_conv2_weight, p_conv2_bias, stride=1, padding=2)
        ref = F.relu(ref)
        ref = F.max_pool2d(ref, kernel_size=3, stride=2)

        ref = F.conv2d(ref, p_conv3_weight, p_conv3_bias, stride=1, padding=1)
        ref = F.relu(ref)

        ref = F.conv2d(ref, p_conv4_weight, p_conv4_bias, stride=1, padding=1)
        ref = F.relu(ref)

        ref = F.conv2d(ref, p_conv5_weight, p_conv5_bias, stride=1, padding=1)
        ref = F.relu(ref)
        ref = F.max_pool2d(ref, kernel_size=3, stride=2)

        ref = torch.flatten(ref, 1)
        ref = F.linear(ref, p_fc1_weight, p_fc1_bias)
        ref = F.relu(ref)
        ref = F.linear(ref, p_fc2_weight, p_fc2_bias)
        ref = F.relu(ref)
        ref = F.linear(ref, p_fc3_weight, p_fc3_bias)

        torch.cuda.synchronize()

        print(f"Output shape: {output.shape}")
        print(f"Ref shape:    {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={(output - ref).abs().max().item():.4e}"
        print("Kernel executed successfully.")