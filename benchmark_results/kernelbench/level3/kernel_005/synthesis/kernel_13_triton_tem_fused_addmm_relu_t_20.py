import torch
import triton
import triton.language as tl
import math

# Fused GEMM + Bias + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64, 'GROUP_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bias_relu_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
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

    A_ptrs = A_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    B_ptrs = B_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_remaining = K - k * BLOCK_K
        mask_k = offs_k < k_remaining
        a = tl.load(A_ptrs, mask=(offs_m[:, None] < M) & (mask_k[None, :]), other=0.0)
        b = tl.load(B_ptrs, mask=(mask_k[:, None]) & (offs_n[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
        A_ptrs += BLOCK_K * stride_ak
        B_ptrs += BLOCK_K * stride_bk

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU if needed
    if APPLY_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store result
    C_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptrs, acc.to(tl.float32), mask=mask)


# Implicit GEMM Conv2d kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
    ],
    key=['BATCH', 'OUT_C', 'OUT_H', 'OUT_W', 'IN_C', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, bias_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Decode M index to (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K index to (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (
            (offs_m[:, None] < M_TOTAL) &
            (batch[:, None] < BATCH) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        # X in NHWC layout: [N, H, W, C]
        x_ptrs = X_ptr + (
            batch[:, None] * (IN_H * IN_W * IN_C) +
            in_c[None, :]
        )
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # W in [OUT_C, IN_C, KH, KW] -> access as [K, OUT_C]
        # W[oc, ic, kh, kw] = W_ptr + oc * (IN_C*KH*KW) + ic * (KH*KW) + kh * KW + kw
        # For our K encoding: k = ic + IN_C * (kw + KW * kh)
        # So W[k, oc] = W_ptr + oc * (IN_C*KH*KW) + ic * (KH*KW) + kh * KW + kw
        w_ptrs = W_ptr + (
            offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) +
            in_c[:, None] * (KERNEL_H * KERNEL_W) +
            kh[:, None] * KERNEL_W +
            kw[:, None]
        )
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store in NHWC layout
    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + (
        batch[:, None] * (OUT_H * OUT_W * OUT_C) +
        out_h[:, None] * (OUT_W * OUT_C) +
        out_w[:, None] * OUT_C +
        offs_n[None, :]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


# MaxPool2d kernel (NHWC layout)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: NHWC layout -> [batch, out_h, out_w, c]
    c = offs % C
    tmp = offs // C
    out_w = tmp % OUT_W
    tmp2 = tmp // OUT_W
    out_h = tmp2 % OUT_H
    batch = tmp2 // OUT_H

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KERNEL_H):
        for kw in range(KERNEL_W):
            ih = out_h * STRIDE_H + kh - PAD_H
            iw = out_w * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + (
            )
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    y_ptrs = Y_ptr + offs
    tl.store(y_ptrs, max_val, mask=mask)


def conv2d_relu(x_nhwc, weight, bias, stride_h, stride_w, pad_h, pad_w, apply_relu=True):
    N, H, W, C_in = x_nhwc.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w

    OUT_H = (H + 2 * pad_h - KH) // stride_h + 1
    OUT_W = (W + 2 * pad_w - KW) // stride_w + 1
    M_TOTAL = N * OUT_H * OUT_W

    y = torch.empty((N, OUT_H, OUT_W, C_out), device=x_nhwc.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    conv2d_relu_kernel[grid](
        x_nhwc, weight, bias, y,
        N, C_in, H, W,
        C_out, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        apply_relu,
        BATCH=N, OUT_C=C_out, OUT_H=OUT_H, OUT_W=OUT_W, IN_C=C_in, KERNEL_H=KH, KERNEL_W=KW,
    )
    return y


def maxpool2d(x_nhwc, kernel_h, kernel_w, stride_h, stride_w, pad_h=0, pad_w=0):
    N, H, W, C = x_nhwc.shape
    OUT_H = (H + 2 * pad_h - kernel_h) // stride_h + 1
    OUT_W = (W + 2 * pad_w - kernel_w) // stride_w + 1
    N_TOTAL = N * OUT_H * OUT_W * C

    y = torch.empty((N, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)

    maxpool2d_kernel[grid](
        x_nhwc, y,
        N, C, H, W, OUT_H, OUT_W,
        kernel_h, kernel_w,
        stride_h, stride_w,
        pad_h, pad_w,
        N_TOTAL=N_TOTAL,
    )
    return y


def linear_relu(x, weight, bias, apply_relu=True):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)

    gemm_bias_relu_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight is [N, K], we want B[k, n]
        out.stride(0), out.stride(1),
        apply_relu,
        M=M, N=N, K=K,
    )
    return out


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
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # Conv1: in=3, out=96, k=11, stride=4, pad=2 -> [1024, 55, 55, 96]
    x = conv2d_relu(x_nhwc, p_conv1_weight, p_conv1_bias, 4, 4, 2, 2, apply_relu=True)

    # MaxPool1: k=3, stride=2 -> [1024, 27, 27, 96]
    x = maxpool2d(x, 3, 3, 2, 2, 0, 0)

    # Conv2: in=96, out=256, k=5, stride=1, pad=2 -> [1024, 27, 27, 256]
    x = conv2d_relu(x, p_conv2_weight, p_conv2_bias, 1, 1, 2, 2, apply_relu=True)

    # MaxPool2: k=3, stride=2 -> [1024, 13, 13, 256]
    x = maxpool2d(x, 3, 3, 2, 2, 0, 0)

    # Conv3: in=256, out=384, k=3, stride=1, pad=1 -> [1024, 13, 13, 384]
    x = conv2d_relu(x, p_conv3_weight, p_conv3_bias, 1, 1, 1, 1, apply_relu=True)

    # Conv4: in=384, out=384, k=3, stride=1, pad=1 -> [1024, 13, 13, 384]
    x = conv2d_relu(x, p_conv4_weight, p_conv4_bias, 1, 1, 1, 1, apply_relu=True)

    # Conv5: in=384, out=256, k=3, stride=1, pad=1 -> [1024, 13, 13, 256]
    x = conv2d_relu(x, p_conv5_weight, p_conv5_bias, 1, 1, 1, 1, apply_relu=True)

    # MaxPool3: k=3, stride=2 -> [1024, 6, 6, 256]
    x = maxpool2d(x, 3, 3, 2, 2, 0, 0)

    # Flatten: [1024, 9216]
    x = x.reshape(x.shape[0], -1).contiguous()

    # FC1: [1024, 9216] -> [1024, 4096] + ReLU
    x = linear_relu(x, p_fc1_weight, p_fc1_bias, apply_relu=True)

    # FC2: [1024, 4096] -> [1024, 4096] + ReLU
    x = linear_relu(x, p_fc2_weight, p_fc2_bias, apply_relu=True)

    # FC3: [1024, 4096] -> [1024, 1000]
    x = linear_relu(x, p_fc3_weight, p_fc3_bias, apply_relu=False)

    return x


def get_inputs():
    torch.manual_seed(42)
    p_conv1_weight = torch.randn(96, 3, 11, 11, device='cuda', dtype=torch.float32) * 0.01
    p_conv1_bias = torch.zeros(96, device='cuda', dtype=torch.float32)
    p_conv2_weight = torch.randn(256, 96, 5, 5, device='cuda', dtype=torch.float32) * 0.01
    p_conv2_bias = torch.zeros(256, device='cuda', dtype=torch.float32)
    p_conv3_weight = torch.randn(384, 256, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_conv3_bias = torch.zeros(384, device='cuda', dtype=torch.float32)
    p_conv4_weight = torch.randn(384, 384, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_conv4_bias = torch.zeros(384, device='cuda', dtype=torch.float32)
    p_conv5_weight = torch.randn(256, 384, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_conv5_bias = torch.zeros(256, device='cuda', dtype=torch.float32)
    p_fc1_weight = torch.randn(4096, 9216, device='cuda', dtype=torch.float32) * 0.01
    p_fc1_bias = torch.zeros(4096, device='cuda', dtype=torch.float32)
    p_fc2_weight = torch.randn(4096, 4096, device='cuda', dtype=torch.float32) * 0.01
    p_fc2_bias = torch.zeros(4096, device='cuda', dtype=torch.float32)
    p_fc3_weight = torch.randn(1000, 4096, device='cuda', dtype=torch.float32) * 0.01
    p_fc3_bias = torch.zeros(1000, device='cuda', dtype=torch.float32)
    x = torch.randn(1024, 3, 224, 224, device='cuda', dtype=torch.float32)
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