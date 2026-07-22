import torch
import triton
import triton.language as tl
import math

# Fused bias + relu pointwise kernel
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
def fused_bias_relu_kernel(
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
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def fused_bias_kernel(
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


# Max pool 3x3 stride 2 padding 1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'H_in', 'W_in', 'H_out', 'W_out'],
)
@triton.jit
def max_pool2d_3x3_s2_p1_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * H_out * W_out
    mask = offsets < total

    w_out = offsets % W_out
    tmp = offsets // W_out
    h_out = tmp % H_out
    tmp2 = tmp // H_out
    c = tmp2 % C
    n = tmp2 // C

    h_base = h_out * 2 - 1
    w_base = w_out * 2 - 1

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for dh in range(3):
        for dw in range(3):
            h_in = h_base + dh
            w_in = w_base + dw
            valid = mask & (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
            idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in
            val = tl.load(x_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# Max pool 3x3 stride 1 padding 1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'H_in', 'W_in'],
)
@triton.jit
def max_pool2d_3x3_s1_p1_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * H_in * W_in
    mask = offsets < total

    w_out = offsets % W_in
    tmp = offsets // W_in
    h_out = tmp % H_in
    tmp2 = tmp // H_in
    c = tmp2 % C
    n = tmp2 // C

    h_base = h_out - 1
    w_base = w_out - 1

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for dh in range(3):
        for dw in range(3):
            h_in = h_base + dh
            w_in = w_base + dw
            valid = mask & (h_in >= 0) & (h_in < H_in) & (w_in >= 0) & (w_in < W_in)
            idx = n * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in
            val = tl.load(x_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# Adaptive avg pool 2d -> 1x1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C
    mask = offsets < total

    c = offsets % C
    n = offsets // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    HW = H * W
    for h in range(H):
        for w in range(W):
            idx = n * (C * H * W) + c * (H * W) + h * W + w
            val = tl.load(x_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    avg = acc / (H * W)
    tl.store(out_ptr + offsets, avg, mask=mask)


# Linear kernel: out = x @ W^T + b
# x: [M, K], W: [N, K], b: [N], out: [M, N]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_w, other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


# Conv2d GEMM kernel: implicit GEMM for conv2d
# Supports arbitrary kernel sizes, strides, padding
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    N_batch, C_in, H_in, W_in,
    C_out, H_out, W_out,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # M = N_batch * H_out * W_out
    # N = C_out
    # K = C_in * KH * KW

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode k -> (c_in, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        c_in_idx = tmp_k // KH

        # Decode m -> (n, h_out, w_out)
        w_out_idx = offs_m % W_out
        tmp_m = offs_m // W_out
        h_out_idx = tmp_m % H_out
        n_idx = tmp_m // H_out

        # Input coordinates
        h_in_idx = h_out_idx[:, None] * stride_h + kh_idx[None, :] - pad_h
        w_in_idx = w_out_idx[:, None] * stride_w + kw_idx[None, :] - pad_w

        valid_m = offs_m < M
        valid_k = offs_k < K
        valid_h = (h_in_idx >= 0) & (h_in_idx < H_in)
        valid_w = (w_in_idx >= 0) & (w_in_idx < W_in)
        mask_x = valid_m[:, None] & valid_k[None, :] & valid_h & valid_w

        x_idx = n_idx[:, None] * (C_in * H_in * W_in) + c_in_idx[None, :] * (H_in * W_in) + h_in_idx * W_in + w_in_idx
        x = tl.load(x_ptr + x_idx, mask=mask_x, other=0.0)

        # Weight: [C_out, C_in, KH, KW] -> index [c_out, c_in, kh, kw]
        # w_idx for each (k, n): w[offs_n, c_in_idx, kh_idx, kw_idx]
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_idx = offs_n[None, :] * (C_in * KH * KW) + c_in_idx[:, None] * (KH * KW) + kh_idx[:, None] * KW + kw_idx[:, None]
        w = tl.load(w_ptr + w_idx, mask=mask_w, other=0.0)

        acc = tl.dot(x, w, acc)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), relu=False):
    N_batch, C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = weight.shape
    stride_h, stride_w = stride
    pad_h, pad_w = padding

    H_out = (H_in + 2 * pad_h - KH) // stride_h + 1
    W_out = (W_in + 2 * pad_w - KW) // stride_w + 1

    M = N_batch * H_out * W_out
    N = C_out
    K = C_in * KH * KW

    # Output in NCHW layout but stored as [M, N] then reshaped
    out_flat = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))

    conv2d_implicit_gemm_kernel[grid](
        x, weight, out_flat,
        N_batch, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        M, N, K,
    )

    # out_flat is [M, N] = [N_batch * H_out * W_out, C_out]
    # Need to reshape to [N_batch, C_out, H_out, W_out]
    # Currently stored as [N_batch * H_out * W_out, C_out]
    # The M dimension is ordered as n * H_out * W_out + h * W_out + w
    out = out_flat.view(N_batch, H_out, W_out, C_out).permute(0, 3, 1, 2).contiguous()

    if bias is not None:
        n_elements = out.numel()
        n_channels = C_out
        grid2 = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        if relu:
            fused_bias_relu_kernel[grid2](out, bias, out, n_elements, n_channels)
        else:
            fused_bias_kernel[grid2](out, bias, out, n_elements, n_channels)
    elif relu:
        n_elements = out.numel()
        n_channels = C_out
        # relu without bias
        grid2 = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        fused_bias_relu_kernel[grid2](out, torch.zeros(C_out, device=x.device), out, n_elements, n_channels)

    return out


def triton_max_pool2d_s2(x):
    N, C, H, W = x.shape
    H_out = (H + 2 - 3) // 2 + 1
    W_out = (W + 2 - 3) // 2 + 1
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    total = N * C * H_out * W_out
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    max_pool2d_3x3_s2_p1_kernel[grid](x, out, N, C, H, W, H_out, W_out)
    return out


def triton_max_pool2d_s1(x):
    N, C, H, W = x.shape
    out = torch.empty((N, C, H, W), device=x.device, dtype=x.dtype)
    total = N * C * H * W
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    max_pool2d_3x3_s1_p1_kernel[grid](x, out, N, C, H, W)
    return out


def triton_adaptive_avg_pool2d(x):
    N, C, H, W = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    total = N * C
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](x, out, N, C, H, W)
    return out


def triton_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, weight, bias, out, M, N, K)
    return out


def inception_module(x,
                     w_1x1, b_1x1,
                     w_r3, b_r3, w_3x3, b_3x3,
                     w_r5, b_r5, w_5x5, b_5x5,
    branch1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), relu=True)

    r3 = triton_conv2d(x, w_r3, b_r3, stride=(1,1), padding=(0,0), relu=True)
    branch3 = triton_conv2d(r3, w_3x3, b_3x3, stride=(1,1), padding=(1,1), relu=True)

    r5 = triton_conv2d(x, w_r5, b_r5, stride=(1,1), padding=(0,0), relu=True)
    branch5 = triton_conv2d(r5, w_5x5, b_5x5, stride=(1,1), padding=(2,2), relu=True)

    pooled = triton_max_pool2d_s1(x)
    branch_pool = triton_conv2d(pooled, w_pool, b_pool, stride=(1,1), padding=(0,0), relu=True)

    return torch.cat([branch1, branch3, branch5, branch_pool], dim=1)


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_conv3_weight, p_conv3_bias,
    p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
    p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
    p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
    p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
    p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
    p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
    p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
    p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
    p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
    p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
    p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
    p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
    p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
    p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
    p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
    p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
    p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
    p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
    p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
    p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
    p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
    p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
    p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
    p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
    p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
    p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias,
    p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
    p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
    p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
    p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
    p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
    p_inception4d_branch_pool_1_weight, p_inception4d_branch_pool_1_bias,
    p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
    p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
    p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
    p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
    p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
    p_inception4e_branch_pool_1_weight, p_inception4e_branch_pool_1_bias,
    p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
    p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
    p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
    p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
    p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
    p_inception5a_branch_pool_1_weight, p_inception5a_branch_pool_1_bias,
    p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
    p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
    p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
    p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
    p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
    p_inception5b_branch_pool_1_weight, p_inception5b_branch_pool_1_bias,
    p_fc_weight, p_fc_bias,
    x,
):
    # conv1: 7x7, stride=2, padding=3
    x = triton_conv2d(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), relu=True)
    # maxpool1: 3x3, stride=2, padding=1
    x = triton_max_pool2d_s2(x)
    # conv2: 1x1
    x = triton_conv2d(x, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), relu=True)
    # conv3: 3x3, padding=1
    x = triton_conv2d(x,