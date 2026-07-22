import torch
import triton
import triton.language as tl
import math

# ============================================================
# Utility: ceiling division
# ============================================================
def cdiv(a, b):
    return (a + b - 1) // b

# ============================================================
# Fused Conv2d + Bias + ReLU kernel (implicit GEMM)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W', 'STRIDE_H', 'STRIDE_W', 'PAD_H', 'PAD_W', 'USE_RELU'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    USE_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_TOTAL = BATCH * OUT_H * OUT_W
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    # Decode M index -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    valid_m = offs_m < M_TOTAL
    valid_n = offs_n < OUT_C

    # W layout: (OUT_C, IN_C, KERNEL_H, KERNEL_W)
    # For fixed kh_i, kw_i: W[out_c, in_c, kh_i, kw_i]
    #   row stride (out_c dim) = IN_C * KERNEL_H * KERNEL_W
    #   col stride (in_c dim)  = KERNEL_H * KERNEL_W  (NOT 1 — in_c is not innermost)
    # Wait: NCHW weight layout is (OUT_C, IN_C, KH, KW), so:
    #   W[oc, ic, kh, kw] = W_ptr + oc*(IN_C*KH*KW) + ic*(KH*KW) + kh*KW + kw
    # For fixed kh_i, kw_i, varying ic: stride between ic elements = KH*KW (not 1)
    # So we cannot use block_ptr with stride=1 for in_c.
    # Instead use scalar loads with coalesced access pattern.
    # The key optimization: fix kh_i, kw_i in outer loops to avoid modulo/division
    # and compute ih/iw once per (kh, kw) pair.

    for kh_i in tl.static_range(KERNEL_H):
        for kw_i in tl.static_range(KERNEL_W):
            # Compute input spatial coordinates for this (kh_i, kw_i)
            # ih shape: (BLOCK_M,), iw shape: (BLOCK_M,)
            ih = out_h * STRIDE_H + kh_i - PAD_H
            iw = out_w * STRIDE_W + kw_i - PAD_W

            valid_h = (ih >= 0) & (ih < IN_H)
            valid_w = (iw >= 0) & (iw < IN_W)
            valid_spatial = valid_m & valid_h & valid_w

            # Base X pointer for this (batch, kh_i, kw_i): varies over in_c
            # X[b, ic, ih, iw] = X_ptr + b*(IN_C*IN_H*IN_W) + ic*(IN_H*IN_W) + ih*IN_W + iw
            x_base = (batch * (IN_C * IN_H * IN_W) + ih * IN_W + iw)  # shape (BLOCK_M,)

            # W base offset for (kh_i, kw_i): W[oc, ic, kh_i, kw_i]
            # = W_ptr + oc*(IN_C*KH*KW) + ic*(KH*KW) + kh_i*KW + kw_i
            w_kh_kw_offset = kh_i * KERNEL_W + kw_i  # scalar

            for k_off in range(0, IN_C, BLOCK_K):
                offs_k = k_off + tl.arange(0, BLOCK_K)
                valid_k = offs_k < IN_C

                # X tile: shape (BLOCK_M, BLOCK_K)
                # X[b, ic, ih, iw] with ic = offs_k
                mask_x = valid_spatial[:, None] & valid_k[None, :]
                x_ptrs = X_ptr + x_base[:, None] + offs_k[None, :] * (IN_H * IN_W)
                x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

                # W tile: shape (BLOCK_K, BLOCK_N)
                # W[oc, ic, kh_i, kw_i] = W_ptr + oc*(IN_C*KH*KW) + ic*(KH*KW) + w_kh_kw_offset
                # For tl.dot we need (BLOCK_K, BLOCK_N): rows=ic, cols=oc
                # W[ic, oc] = W_ptr + oc*(IN_C*KH*KW) + ic*(KH*KW) + w_kh_kw_offset
                mask_w = valid_k[:, None] & valid_n[None, :]
                w_ptrs = W_ptr + offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) + offs_k[:, None] * (KERNEL_H * KERNEL_W) + w_kh_kw_offset
                w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

                acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    mask_n = offs_n < OUT_C
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    if USE_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store: output is NCHW
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)

def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH: tl.constexpr, C: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    N_TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: N_TOTAL = BATCH * C * OUT_H * OUT_W
    out_w = offs % OUT_W
    tmp = offs // OUT_W
    out_h = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    b = tmp2 // C

    result = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h * STRIDE_H + kh - PAD_H
            iw = out_w * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + b * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            result = tl.maximum(result, val)

    tl.store(Y_ptr + offs, result, mask=mask)


# ============================================================
# ReLU pointwise kernel
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def relu_kernel(X_ptr, Y_ptr, N: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(X_ptr + offs, mask=mask, other=0.0)
    y = tl.maximum(x, 0.0)
    tl.store(Y_ptr + offs, y, mask=mask)


# ============================================================
# FC (GEMM) kernel: Y = X @ W^T + B, optionally with ReLU
# X: (M, K), W: (N, K), B: (N,), Y: (M, N)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def fc_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    USE_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    if USE_RELU:
        acc = tl.maximum(acc, 0.0)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc.to(tl.float32), mask=mask_out)


# ============================================================
# Helper to run conv2d
# ============================================================
def run_conv2d(x, w, b, stride_h, stride_w, pad_h, pad_w, use_relu=True):
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = w.shape
    OUT_H = (IN_H + 2 * pad_h - KERNEL_H) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KERNEL_W) // stride_w + 1

    y = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))

    conv2d_bias_relu_kernel[grid](
        x, w, b, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        stride_h, stride_w,
        pad_h, pad_w,
        use_relu,
    )
    return y


def run_maxpool2d(x, kh, kw, stride_h, stride_w, pad_h=0, pad_w=0):
    BATCH, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + 2 * pad_h - kh) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - kw) // stride_w + 1
    N_TOTAL = BATCH * C * OUT_H * OUT_W

    y = torch.empty((BATCH, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, y,
        BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
        kh, kw, stride_h, stride_w, pad_h, pad_w,
        N_TOTAL,
    )
    return y


def run_fc(x, w, b, use_relu=False):
    M, K = x.shape
    N, K2 = w.shape
    assert K == K2

    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(N, meta['BLOCK_N']))
    fc_relu_kernel[grid](
        x, w, b, y,
        M, N, K,
        use_relu,
    )
    return y


# ============================================================
# Main kernel function
# ============================================================
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
    # Conv1: in=3, out=96, k=11, stride=4, pad=2 -> (1024, 96, 55, 55)
    x = run_conv2d(x, p_conv1_weight, p_conv1_bias, 4, 4, 2, 2, use_relu=True)
    # MaxPool1: k=3, stride=2 -> (1024, 96, 27, 27)
    x = run_maxpool2d(x, 3, 3, 2, 2, 0, 0)

    # Conv2: in=96, out=256, k=5, stride=1, pad=2 -> (1024, 256, 27, 27)
    x = run_conv2d(x, p_conv2_weight, p_conv2_bias, 1, 1, 2, 2, use_relu=True)
    # MaxPool2: k=3, stride=2 -> (1024, 256, 13, 13)
    x = run_maxpool2d(x, 3, 3, 2, 2, 0, 0)

    # Conv3: in=256, out=384, k=3, stride=1, pad=1 -> (1024, 384, 13, 13)
    x = run_conv2d(x, p_conv3_weight, p_conv3_bias, 1, 1, 1, 1, use_relu=True)

    # Conv4: in=384, out=384, k=3, stride=1, pad=1 -> (1024, 384, 13, 13)
    x = run_conv2d(x, p_conv4_weight, p_conv4_bias, 1, 1, 1, 1, use_relu=True)

    # Conv5: in=384, out=256, k=3, stride=1, pad=1 -> (1024, 256, 13, 13)
    x = run_conv2d(x, p_conv5_weight, p_conv5_bias, 1, 1, 1, 1, use_relu=True)
    # MaxPool3: k=3, stride=2 -> (1024, 256, 6, 6)
    x = run_maxpool2d(x, 3, 3, 2, 2, 0, 0)

    # Flatten: (1024, 9216)
    x = x.reshape(x.shape[0], -1)

    # FC1: (1024, 9216) -> (1024, 4096) + ReLU
    x = run_fc(x, p_fc1_weight, p_fc1_bias, use_relu=True)

    # FC2: (1024, 4096) -> (1024, 4096) + ReLU
    x = run_fc(x, p_fc2_weight, p_fc2_bias, use_relu=True)

    # FC3: (1024, 4096) -> (1024, 1000)
    x = run_fc(x, p_fc3_weight, p_fc3_bias, use_relu=False)

    return x


def get_inputs():
    batch_size = 1024
    return [
        torch.randn(96, 3, 11, 11, device='cuda', dtype=torch.float32),
        torch.randn(96, device='cuda', dtype=torch.float32),
        torch.randn(256, 96, 5, 5, device='cuda', dtype=torch.float32),
        torch.randn(256, device='cuda', dtype=torch.float32),
        torch.randn(384, 256, 3, 3, device='cuda', dtype=torch.float32),
        torch.randn(384, device='cuda', dtype=torch.float32),
        torch.randn(384, 384, 3, 3, device='cuda', dtype=torch.float32),
        torch.randn(384, device='cuda', dtype=torch.float32),
        torch.randn(256, 384, 3, 3, device='cuda', dtype=torch.float32),
        torch.randn(256, device='cuda', dtype=torch.float32),
        torch.randn(4096, 9216, device='cuda', dtype=torch.float32),
        torch.randn(4096, device='cuda', dtype=torch.float32),
        torch.randn(4096, 4096, device='cuda', dtype=torch.float32),
        torch.randn(4096, device='cuda', dtype=torch.float32),
        torch.randn(1000, 4096, device='cuda', dtype=torch.float32),
        torch.randn(1000, device='cuda', dtype=torch.float32),
        torch.randn(batch_size, 3, 224, 224, device='cuda', dtype=torch.float32),
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
        ref = F.conv2d(x, p_conv1_weight, p_conv1_bias, stride=4, padding=2)
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

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")