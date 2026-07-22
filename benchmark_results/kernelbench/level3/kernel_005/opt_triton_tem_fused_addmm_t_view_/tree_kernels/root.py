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
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_stages=4, num_warps=4),
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


def linear_relu(x, weight, bias, apply_relu=True):
    """x: (M, K), weight: (N, K), bias: (N,) -> (M, N)"""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
    
    gemm_bias_relu_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # B is transposed: weight[N,K], we want K->N
        out.stride(0), out.stride(1),
        APPLY_RELU=apply_relu,
    )
    return out


# Max pool 2D kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['total_elements'],
)
@triton.jit
def max_pool2d_kernel(
    input_ptr, output_ptr,
    N, C, H, W,
    out_H, out_W,
    kernel_size, stride_pool,
    stride_n, stride_c, stride_h, stride_w,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    # Decode linear index -> (n, c, oh, ow)
    ow = offs % out_W
    tmp = offs // out_W
    oh = tmp % out_H
    tmp2 = tmp // out_H
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(kernel_size):
        for kw in range(kernel_size):
            ih = oh * stride_pool + kh
            iw = ow * stride_pool + kw
            valid = mask & (ih < H) & (iw < W) & (n < N)
            lin_idx = n * stride_n + c * stride_c + ih * stride_h + iw * stride_w
            val = tl.load(input_ptr + lin_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(output_ptr + offs, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2):
    N, C, H, W = x.shape
    out_H = (H - kernel_size) // stride + 1
    out_W = (W - kernel_size) // stride + 1
    out = torch.empty((N, C, out_H, out_W), device=x.device, dtype=x.dtype)
    total = N * C * out_H * out_W

    grid = lambda META: (triton.cdiv(total, META['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        out_H, out_W,
        kernel_size, stride,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        total,
    )
    return out


# Conv2d + ReLU fused kernel using implicit GEMM
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
    ],
    key=['BATCH_SIZE', 'OUT_C', 'IN_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, bias_ptr, Y_ptr,
    BATCH_SIZE, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # Input strides (NCHW)
    s_xn, s_xc, s_xh, s_xw,
    # Weight strides (NCHW: out_c, in_c, kh, kw)
    s_woc, s_wic, s_wkh, s_wkw,
    # Output strides (NCHW)
    s_yn, s_yc, s_yh, s_yw,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_total = BATCH_SIZE * OUT_H * OUT_W

    # Decode M -> (batch, out_h, out_w)
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c = tmp_k // KERNEL_H

        # Input coordinates
        ih = oh[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_x = (
            (offs_m[:, None] < M_total) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_ptrs = X_ptr + batch[:, None] * s_xn + in_c[None, :] * s_xc + ih * s_xh + iw * s_xw
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: [BLOCK_K, BLOCK_N] -> w[out_c, in_c, kh, kw]
        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * s_woc + in_c[:, None] * s_wic + kh[:, None] * s_wkh + kw[:, None] * s_wkw
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store: output is NCHW
    mask_out = (offs_m[:, None] < M_total) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + batch[:, None] * s_yn + offs_n[None, :] * s_yc + oh[:, None] * s_yh + ow[:, None] * s_yw
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_conv2d_relu(x, weight, bias, stride, padding, apply_relu=True):
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    SH, SW = stride if isinstance(stride, (list, tuple)) else (stride, stride)
    PH, PW = padding if isinstance(padding, (list, tuple)) else (padding, padding)

    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    y = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

    M_total = N * OH * OW

    grid = lambda META: (
        triton.cdiv(M_total, META['BLOCK_M']),
        triton.cdiv(C_out, META['BLOCK_N']),
    )

    conv2d_relu_kernel[grid](
        x, weight, bias, y,
        N, C_in, H, W,
        C_out, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        APPLY_RELU=apply_relu,
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
    # Conv1: (1024, 3, 224, 224) -> (1024, 96, 55, 55), stride=4, pad=2, kernel=11
    x = triton_conv2d_relu(x, p_conv1_weight, p_conv1_bias, stride=(4, 4), padding=(2, 2), apply_relu=True)
    # MaxPool1: kernel=3, stride=2 -> (1024, 96, 27, 27)
    x = triton_max_pool2d(x, kernel_size=3, stride=2)

    # Conv2: (1024, 96, 27, 27) -> (1024, 256, 27, 27), stride=1, pad=2, kernel=5
    x = triton_conv2d_relu(x, p_conv2_weight, p_conv2_bias, stride=(1, 1), padding=(2, 2), apply_relu=True)
    # MaxPool2: kernel=3, stride=2 -> (1024, 256, 13, 13)
    x = triton_max_pool2d(x, kernel_size=3, stride=2)

    # Conv3: (1024, 256, 13, 13) -> (1024, 384, 13, 13), stride=1, pad=1, kernel=3
    x = triton_conv2d_relu(x, p_conv3_weight, p_conv3_bias, stride=(1, 1), padding=(1, 1), apply_relu=True)

    # Conv4: (1024, 384, 13, 13) -> (1024, 384, 13, 13), stride=1, pad=1, kernel=3
    x = triton_conv2d_relu(x, p_conv4_weight, p_conv4_bias, stride=(1, 1), padding=(1, 1), apply_relu=True)

    # Conv5: (1024, 384, 13, 13) -> (1024, 256, 13, 13), stride=1, pad=1, kernel=3
    x = triton_conv2d_relu(x, p_conv5_weight, p_conv5_bias, stride=(1, 1), padding=(1, 1), apply_relu=True)
    # MaxPool3: kernel=3, stride=2 -> (1024, 256, 6, 6)
    x = triton_max_pool2d(x, kernel_size=3, stride=2)

    # Flatten: (1024, 9216)
    x = x.reshape(x.shape[0], -1).contiguous()

    # FC1: (1024, 9216) -> (1024, 4096) + ReLU
    x = linear_relu(x, p_fc1_weight, p_fc1_bias, apply_relu=True)

    # FC2: (1024, 4096) -> (1024, 4096) + ReLU
    x = linear_relu(x, p_fc2_weight, p_fc2_bias, apply_relu=True)

    # FC3: (1024, 4096) -> (1024, 1000)
    x = linear_relu(x, p_fc3_weight, p_fc3_bias, apply_relu=False)

    return x


def get_inputs():
    torch.manual_seed(42)
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

        ref = ref.flatten(1)

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