import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: fused bias + relu (pointwise epilogue)
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
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
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
def bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Triton kernel: max_pool2d (3x3, stride=2, padding=1)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW', 'IH', 'IW'],
)
@triton.jit
def max_pool2d_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    out_idx = n * (C * OH * OW) + c * (OH * OW) + oh * OW + ow
    tl.store(out_ptr + out_idx, max_val, mask=mask)


# ============================================================
# Triton kernel: max_pool2d (3x3, stride=1, padding=1) for inception branch_pool
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW', 'IH', 'IW'],
)
@triton.jit
def max_pool2d_s1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    ow = offsets % OW
    tmp = offsets // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)

    out_idx = n * (C * OH * OW) + c * (OH * OW) + oh * OW + ow
    tl.store(out_ptr + out_idx, max_val, mask=mask)


# ============================================================
# Triton kernel: adaptive_avg_pool2d to 1x1
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'IH', 'IW'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    c = offsets % C
    n = offsets // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = IH * IW

    for h in range(IH):
        for w in range(IW):
            idx = n * (C * IH * IW) + c * (IH * IW) + h * IW + w
            val = tl.load(inp_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    avg = acc / count
    tl.store(out_ptr + offsets, avg, mask=mask)


# ============================================================
# Triton kernel: linear (matmul) + bias
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
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
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x = tl.load(x_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w = tl.load(w_ptr + offs_k[:, None] * N + offs_n[None, :], mask=w_mask, other=0.0)
        acc = tl.dot(x, w, acc)

    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ============================================================
# Implicit GEMM Conv2d kernel (NHWC layout)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    x_ptr, w_ptr, out_ptr,
    N_batch, C_in, H_in, W_in,
    C_out, H_out, W_out,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = N_batch * H_out * W_out
    # N = C_out
    # K = C_in * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (n, h_out, w_out)
    w_out_idx = offs_m % W_out
    tmp = offs_m // W_out
    h_out_idx = tmp % H_out
    n_idx = tmp // H_out

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # Decode K -> (c_in, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        c_in_idx = tmp_k // KH

        # Input coordinates
        h_in_idx = h_out_idx[:, None] * SH + kh_idx[None, :] - PH
        w_in_idx = w_out_idx[:, None] * SW + kw_idx[None, :] - PW

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_h = (h_in_idx >= 0) & (h_in_idx < H_in)
        valid_w = (w_in_idx >= 0) & (w_in_idx < W_in)
        valid_n = n_idx[:, None] < N_batch
        x_mask = valid_m & valid_k & valid_h & valid_w & valid_n

        # NCHW layout: x[n, c, h, w] = x_ptr[n*C_in*H_in*W_in + c*H_in*W_in + h*W_in + w]
        x_idx = (n_idx[:, None] * C_in * H_in * W_in +
                 c_in_idx[None, :] * H_in * W_in +
        x = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        # Weight layout: w[c_out, c_in, kh, kw] = w_ptr[c_out*C_in*KH*KW + c_in*KH*KW + kh*KW + kw]
        # We need w[k, n] = w[c_in, kh, kw, c_out]
        # w_ptr layout: [C_out, C_in, KH, KW]
        # w[offs_k, offs_n] = w_ptr[offs_n * C_in*KH*KW + c_in_idx * KH*KW + kh_idx * KW + kw_idx]
        w_k_idx = c_in_idx[:, None] * KH * KW + kh_idx[:, None] * KW + kw_idx[:, None]
        w_n_idx = offs_n[None, :] * C_in * KH * KW
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w = tl.load(w_ptr + w_n_idx + w_k_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    # Output NCHW: out[n, c_out, h_out, w_out]
    out_idx = (n_idx[:, None] * C_out * H_out * W_out +
               offs_n[None, :] * H_out * W_out +
               h_out_idx[:, None] * W_out +
               w_out_idx[:, None])
    tl.store(out_ptr + out_idx, acc, mask=out_mask)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), relu=False):
    """Perform conv2d using Triton implicit GEMM kernel."""
    N, C_in, H_in, W_in = x.shape
    C_out, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    H_out = (H_in + 2 * PH - KH) // SH + 1
    W_out = (W_in + 2 * PW - KW) // SW + 1

    M = N * H_out * W_out
    Nk = C_out
    K = C_in * KH * KW

    out = torch.empty((N, C_out, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Nk, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x, weight, out,
        N, C_in, H_in, W_in,
        C_out, H_out, W_out,
        KH, KW, SH, SW, PH, PW,
        M, Nk, K,
    )

    if bias is not None or relu:
        n_elements = out.numel()
        if relu:
            bias_relu_kernel[lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)](
            )
        else:
            bias_kernel[lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)](
            )

    return out


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H_in, W_in = x.shape
    H_out = (H_in + 2 * padding - kernel_size) // stride + 1
    W_out = (W_in + 2 * padding - kernel_size) // stride + 1

    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    total = N * C * H_out * W_out

    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)

    if stride == 2:
        max_pool2d_kernel[grid](
            x, out,
            N, C, H_in, W_in, H_out, W_out,
            kernel_size, kernel_size,
            stride, stride,
            padding, padding,
        )
    else:
        max_pool2d_s1_kernel[grid](
            x, out,
            N, C, H_in, W_in, H_out, W_out,
            kernel_size, kernel_size,
            stride, stride,
            padding, padding,
        )

    return out


def triton_adaptive_avg_pool2d_1x1(x):
    N, C, H_in, W_in = x.shape
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    total = N * C
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
    )
    return out


def triton_linear(x, weight, bias):
    # x: [M, K], weight: [N, K], bias: [N]
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    # weight is [N, K], we need [K, N] for matmul
    w_t = weight.t().contiguous()
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_kernel[grid](x, w_t, bias, out, M, N, K)
    return out


def inception_module(x, w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    """Inception module: 4 branches concatenated."""
    # Branch 1: 1x1 conv + relu
    branch1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), relu=True)

    # Branch 2: 1x1 reduce + relu, then 3x3 conv + relu
    branch2 = triton_conv2d(x, w_3x3_r, b_3x3_r, stride=(1,1), padding=(0,0), relu=True)
    branch2 = triton_conv2d(branch2, w_3x3, b_3x3, stride=(1,1), padding=(1,1), relu=True)

    # Branch 3: 1x1 reduce + relu, then 5x5 conv + relu
    branch3 = triton_conv2d(x, w_5x5_r, b_5x5_r, stride=(1,1), padding=(0,0), relu=True)
    branch3 = triton_conv2d(branch3, w_5x5, b_5x5, stride=(1,1), padding=(2,2), relu=True)

    # Branch 4: max_pool 3x3 s1 p1, then 1x1 conv + relu
    branch4 = triton_max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch4 = triton_conv2d(branch4, w_pool, b_pool, stride=(1,1), padding=(0,0), relu=True)

    return torch.cat([branch1, branch2, branch3, branch4], dim=1)


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