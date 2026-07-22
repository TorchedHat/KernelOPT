import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# Fused bias add + ReLU kernel
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
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


# Max pool 3x3 stride 2 padding 1 kernel (NCHW format)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def maxpool2d_s2_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * OH * OW
    mask = offsets < total

    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)

    ih_base = oh * 2 - 1
    iw_base = ow * 2 - 1

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for di in range(3):
        for dj in range(3):
            ih = ih_base + di
            iw = iw_base + dj
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * C * IH * IW + c * IH * IW + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(val > max_val, val, max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# Max pool 3x3 stride 1 padding 1 kernel (for inception branch pool)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def maxpool2d_s1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * OH * OW
    mask = offsets < total

    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)

    ih_base = oh - 1
    iw_base = ow - 1

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for di in range(3):
        for dj in range(3):
            ih = ih_base + di
            iw = iw_base + dj
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * C * IH * IW + c * IH * IW + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(val > max_val, val, max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


# Adaptive avg pool 7x7 -> 1x1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N', 'C'],
)
@triton.jit
def adaptive_avg_pool_7x7_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C
    mask = offsets < total

    c = offsets % C
    n = offsets // C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = IH * IW

    for i in range(IH):
        for j in range(IW):
            idx = n * C * IH * IW + c * IH * IW + i * IW + j
            val = tl.load(inp_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    avg = acc / count
    tl.store(out_ptr + offsets, avg, mask=mask)


# Linear layer: out = x @ W^T + b
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc)

    offs_n2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    b = tl.load(b_ptr + offs_n2, mask=offs_n2 < N, other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n2[None, :] < N)
    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n2[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


def conv_bias_relu(x, weight, bias, stride=1, padding=0):
    out = F.conv2d(x, weight, None, stride, padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_relu_kernel[grid](out, bias, result, n_elements, C)
    return result


def conv_bias(x, weight, bias, stride=1, padding=0):
    out = F.conv2d(x, weight, None, stride, padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_kernel[grid](out, bias, result, n_elements, C)
    return result


def maxpool_s2(x):
    N, C, H, W = x.shape
    OH = (H + 2 * 1 - 3) // 2 + 1
    OW = (W + 2 * 1 - 3) // 2 + 1
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_s2_kernel[grid](x, out, N, C, H, W, OH, OW)
    return out


def maxpool_s1(x):
    N, C, H, W = x.shape
    out = torch.empty(N, C, H, W, device=x.device, dtype=x.dtype)
    total = N * C * H * W
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_s1_kernel[grid](x, out, N, C, H, W, H, W)
    return out


def adaptive_avg_pool(x):
    N, C, H, W = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N * C, meta['BLOCK_SIZE']),)
    adaptive_avg_pool_7x7_kernel[grid](x, out, N, C, H, W)
    return out


def inception_module(x,
                     w1x1, b1x1,
                     w3x3_r, b3x3_r, w3x3, b3x3,
                     w5x5_r, b5x5_r, w5x5, b5x5,
    branch1 = conv_bias_relu(x, w1x1, b1x1, stride=1, padding=0)
    branch2 = conv_bias_relu(x, w3x3_r, b3x3_r, stride=1, padding=0)
    branch2 = conv_bias_relu(branch2, w3x3, b3x3, stride=1, padding=1)
    branch3 = conv_bias_relu(x, w5x5_r, b5x5_r, stride=1, padding=0)
    branch3 = conv_bias_relu(branch3, w5x5, b5x5, stride=1, padding=2)
    pool = maxpool_s1(x)
    branch4 = conv_bias_relu(pool, wp, bp, stride=1, padding=0)
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
    p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
    p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
    p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
    p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
    p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
    p_inception5b_branch_pool_1_weight, p_inception5b_branch_pool_1_bias,
    p_fc_weight, p_fc_bias,
    x,
):
    # conv1 + relu + maxpool
    x = conv_bias_relu(x, p_conv1_weight, p_conv1_bias, stride=2, padding=3)
    x = maxpool_s2(x)

    # conv2 + relu
    x = conv_bias_relu(x, p_conv2_weight, p_conv2_bias, stride=1, padding=0)

    # conv3 + relu + maxpool
    x = conv_bias_relu(x, p_conv3_weight, p_conv3_bias, stride=1, padding=1)
    x = maxpool_s2(x)

    # inception3a
    x = inception_module(x,
                         p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
                         p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
                         p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
                         p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
                         p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,

    # inception3b
    x = inception_module(x,
                         p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
                         p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
                         p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
                         p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
                         p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,

    x = maxpool_s2(x)

    # inception4a
    x = inception_module(x,
                         p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
                         p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
                         p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
                         p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
                         p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,

    # inception4b
    x = inception_module(x,
                         p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
                         p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
                         p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
                         p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
                         p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,

    # inception4c
    x = inception_module(x,
                         p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
                         p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
                         p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
                         p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
                         p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,

    # inception4d
    x = inception_module(x,
                         p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
                         p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
                         p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
                         p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
                         p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,

    # inception4e
    x = inception_module(x,
                         p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
                         p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
                         p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
                         p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
                         p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,

    x = maxpool_s2(x)

    # inception5a
    x = inception_module(x,
                         p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
                         p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
                         p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
                         p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
                         p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,

    # inception5b
    x = inception_module(x,
                         p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
                         p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
                         p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
                         p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
                         p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,

    # adaptive avg pool
    x = adaptive_avg_pool(x)

    # flatten
    x = x.reshape(x.shape[0], -1)

    # dropout (inference: no-op, but we need to pass through)
    # linear
    M, K = x.shape
    N = p_fc_weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, p_fc_weight, p_fc_bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        p_fc_weight.stride(0), p_fc_weight.stride(1),