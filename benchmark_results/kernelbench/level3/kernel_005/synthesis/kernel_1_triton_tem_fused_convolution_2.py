import torch
import triton
import triton.language as tl
import argparse

# ─── Fused Linear + Bias + ReLU kernel ───────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    APPLY_RELU: tl.constexpr,
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

    mask_m = offs_m < M
    mask_n = offs_n < N

    X_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    W_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_mask = offs_k < K - k * BLOCK_K
        x = tl.load(X_ptrs, mask=mask_m[:, None] & k_mask[None, :], other=0.0)
        w = tl.load(W_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)
        acc = tl.dot(x, w, acc)
        X_ptrs += BLOCK_K * stride_xk
        W_ptrs += BLOCK_K * stride_wk

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU if needed
    if APPLY_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    Y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(Y_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


# ─── Conv2d implicit GEMM kernel ─────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W',
         'KERNEL_H', 'KERNEL_W', 'STRIDE_H', 'STRIDE_W', 'PAD_H', 'PAD_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    mask_m = offs_m < (BATCH * OUT_H * OUT_W)
    mask_n = offs_n < OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    BLOCK_K_COUNT = tl.cdiv(TOTAL_K, BLOCK_K)

    for k_block in range(BLOCK_K_COUNT):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = offs_k < TOTAL_K

        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (
            mask_m[:, None] &
            k_mask[None, :] &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W)
        )

        # X in NHWC layout: [batch, in_h, in_w, in_c]
        x_ptrs = X_ptr + (
            batch_idx[:, None] * (IN_H * IN_W * IN_C) +
            in_c[None, :]
        )
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # W in [OUT_C, IN_C, KH, KW] -> access as [K, OUT_C]
        # w[oc, ic, kh, kw] = W_ptr + oc*(IN_C*KH*KW) + ic + kh*(IN_C*KW) + kw*IN_C  (NHWC weight)
        # Actually weight is stored as [OUT_C, IN_C, KH, KW] in standard layout
        # We need w[k, n] where k = ic + kw*IN_C + kh*(IN_C*KW), n = oc
        # w_ptr[oc, ic, kh, kw] = oc*(IN_C*KH*KW) + ic*(KH*KW) + kh*KW + kw  (standard OIHW)
        # But we decoded k as: in_c = k%IN_C, kw=(k//IN_C)%KW, kh=k//(IN_C*KW)
        # So standard index: oc*(IN_C*KH*KW) + in_c*(KH*KW) + kh*KW + kw
        w_ptrs = W_ptr + (
            offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) +
            in_c[:, None] * (KERNEL_H * KERNEL_W) +
            kh[:, None] * KERNEL_W +
            kw[:, None]
        )
        w_tile = tl.load(w_ptrs, mask=k_mask[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store in NHWC: Y[batch, out_h, out_w, out_c]
    y_ptrs = Y_ptr + (
        batch_idx[:, None] * (OUT_H * OUT_W * OUT_C) +
        out_h_idx[:, None] * (OUT_W * OUT_C) +
        out_w_idx[:, None] * OUT_C +
        offs_n[None, :]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


# ─── MaxPool2d kernel ─────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_stages=4, num_warps=2),
    ],
    key=['BATCH', 'C', 'IN_H', 'IN_W', 'OUT_H', 'OUT_W', 'KH', 'KW', 'SH', 'SW'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = BATCH * C * OUT_H * OUT_W
    mask = offs < total

    # Decode: output is in NHWC layout
    c_idx = offs % C
    tmp = offs // C
    out_w_idx = tmp % OUT_W
    tmp2 = tmp // OUT_W
    out_h_idx = tmp2 % OUT_H
    batch_idx = tmp2 // OUT_H

    # Compute max over kernel window
    result = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            ih = out_h_idx * SH + kh - PAD_H
            iw = out_w_idx * SW + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + (
            )
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            result = tl.maximum(result, val)

    y_ptrs = Y_ptr + (
    )
    tl.store(y_ptrs, result, mask=mask)


def run_conv2d_relu(x_nhwc, weight, bias, stride_h, stride_w, pad_h, pad_w, apply_relu):
    """x_nhwc: [N, H, W, C], weight: [OC, IC, KH, KW], bias: [OC]"""
    N, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = weight.shape
    OUT_H = (IN_H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KW) // stride_w + 1

    y = torch.empty(N, OUT_H, OUT_W, OUT_C, device=x_nhwc.device, dtype=torch.float32)

    M = N * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))

    conv2d_relu_kernel[grid](
        x_nhwc, weight, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW, stride_h, stride_w, pad_h, pad_w,
        apply_relu,
    )
    return y


def run_maxpool2d(x_nhwc, kh, kw, sh, sw, pad_h=0, pad_w=0):
    """x_nhwc: [N, H, W, C]"""
    N, IN_H, IN_W, C = x_nhwc.shape
    OUT_H = (IN_H + 2 * pad_h - kh) // sh + 1
    OUT_W = (IN_W + 2 * pad_w - kw) // sw + 1

    y = torch.empty(N, OUT_H, OUT_W, C, device=x_nhwc.device, dtype=torch.float32)
    total = N * C * OUT_H * OUT_W

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x_nhwc, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        kh, kw, sh, sw, pad_h, pad_w,
    )
    return y


def run_linear_relu(x, weight, bias, apply_relu):
    """x: [M, K], weight: [N, K], bias: [N] -> [M, N]"""
    M, K = x.shape
    N = weight.shape[0]
    y = torch.empty(M, N, device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
    linear_relu_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        apply_relu,
    )
    return y


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
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

    # Conv1: [1024, 3, 224, 224] -> [1024, 96, 55, 55], stride=4, pad=2, kernel=11
    x_nhwc = run_conv2d_relu(x_nhwc, p_conv1_weight, p_conv1_bias, 4, 4, 2, 2, True)
    # MaxPool1: kernel=3, stride=2
    x_nhwc = run_maxpool2d(x_nhwc, 3, 3, 2, 2)

    # Conv2: [1024, 96, 27, 27] -> [1024, 256, 27, 27], stride=1, pad=2, kernel=5
    x_nhwc = run_conv2d_relu(x_nhwc, p_conv2_weight, p_conv2_bias, 1, 1, 2, 2, True)
    # MaxPool2: kernel=3, stride=2
    x_nhwc = run_maxpool2d(x_nhwc, 3, 3, 2, 2)

    # Conv3: [1024, 256, 13, 13] -> [1024, 384, 13, 13], stride=1, pad=1, kernel=3
    x_nhwc = run_conv2d_relu(x_nhwc, p_conv3_weight, p_conv3_bias, 1, 1, 1, 1, True)

    # Conv4: [1024, 384, 13, 13] -> [1024, 384, 13, 13], stride=1, pad=1, kernel=3
    x_nhwc = run_conv2d_relu(x_nhwc, p_conv4_weight, p_conv4_bias, 1, 1, 1, 1, True)

    # Conv5: [1024, 384, 13, 13] -> [1024, 256, 13, 13], stride=1, pad=1, kernel=3
    x_nhwc = run_conv2d_relu(x_nhwc, p_conv5_weight, p_conv5_bias, 1, 1, 1, 1, True)
    # MaxPool3: kernel=3, stride=2
    x_nhwc = run_maxpool2d(x_nhwc, 3, 3, 2, 2)

    # Flatten: [1024, 6, 6, 256] -> [1024, 9216]
    N = x_nhwc.shape[0]
    x_flat = x_nhwc.reshape(N, -1).contiguous()

    # FC1: [1024, 9216] -> [1024, 4096] + ReLU
    x_fc = run_linear_relu(x_flat, p_fc1_weight, p_fc1_bias, True)

    # FC2: [1024, 4096] -> [1024, 4096] + ReLU
    x_fc = run_linear_relu(x_fc, p_fc2_weight, p_fc2_bias, True)

    # FC3: [1024, 4096] -> [1024, 1000]
    x_fc = run_linear_relu(x_fc, p_fc3_weight, p_fc3_bias, False)

    return x_fc


def get_inputs():
    torch.manual_seed(42)
    batch_size = 1024
    p_conv1_weight = torch.randn(96, 3, 11, 11, device='cuda')
    p_conv1_bias = torch.randn(96, device='cuda')
    p_conv2_weight = torch.randn(256, 96, 5, 5, device='cuda')
    p_conv2_bias = torch.randn(256, device='cuda')
    p_conv3_weight = torch.randn(384, 256, 3, 3, device='cuda')
    p_conv3_bias = torch.randn(384, device='cuda')
    p_conv4_weight = torch.randn(384, 384, 3, 3, device='cuda')
    p_conv4_bias = torch.randn(384, device='cuda')
    p_conv5_weight = torch.randn(256, 384, 3, 3, device='cuda')
    p_conv5_bias = torch.randn(256, device='cuda')
    p_fc1_weight = torch.randn(4096, 9216, device='cuda')
    p_fc1_bias = torch.randn(4096, device='cuda')
    p_fc2_weight = torch.randn(4096, 4096, device='cuda')
    p_fc2_bias = torch.randn(4096, device='cuda')
    p_fc3_weight = torch.randn(1000, 4096, device='cuda')
    p_fc3_bias = torch.randn(1000, device='cuda')
    x = torch.randn(batch_size, 3, 224, 224, device='cuda')
    return [
        p_conv1_weight, p_conv1_bias,
        p_conv2_weight, p_conv2_bias,
        p_conv3_weight, p_conv3_bias,
        p_conv4_weight, p_conv4_bias,
        p_conv5_weight, p_conv5_bias,
        p_fc1_weight, p_fc1_bias,
        p_fc2_weight, p_fc2_bias,
        p_fc3_weight, p_fc3_bias,
        x,
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
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={(output - ref).abs().max().item():.4e}"
        print("Kernel executed successfully.")