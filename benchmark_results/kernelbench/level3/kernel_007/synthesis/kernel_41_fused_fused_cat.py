import torch
import triton
import triton.language as tl


# Fused bias add + ReLU kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
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
    # channel index = offset % n_channels
    chan = offsets % n_channels
    bias = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    out = x + bias
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# Fused bias add (no relu) kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
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
    chan = offsets % n_channels
    bias = tl.load(bias_ptr + chan, mask=mask, other=0.0)
    out = x + bias
    tl.store(out_ptr + offsets, out, mask=mask)


# MaxPool2d kernel: kernel=3, stride=2, padding=1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W', 'IN_H', 'IN_W'],
)
@triton.jit
def maxpool2d_k3s2p1_kernel(
    inp_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each thread handles one output element
    total = N * C * OUT_H * OUT_W
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    # Decode indices
    ow = offsets % OUT_W
    tmp = offsets // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    # Input base
    inp_base = n * C * IN_H * IN_W + c * IN_H * IN_W

    # Pool over 3x3 window with stride=2, padding=1
    ih_start = oh * 2 - 1
    iw_start = ow * 2 - 1

    result = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for ki in range(3):
        for kj in range(3):
            ih = ih_start + ki
            iw = iw_start + kj
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            idx = inp_base + ih * IN_W + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            result = tl.where(valid, tl.maximum(result, val), result)

    tl.store(out_ptr + offsets, result, mask=mask)


# MaxPool2d kernel: kernel=3, stride=1, padding=1 (used in inception branch_pool)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W', 'IN_H', 'IN_W'],
)
@triton.jit
def maxpool2d_k3s1p1_kernel(
    inp_ptr, out_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    BLOCK_SIZE: tl.constexpr,
):
    total = N * C * OUT_H * OUT_W
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    ow = offsets % OUT_W
    tmp = offsets // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    inp_base = n * C * IN_H * IN_W + c * IN_H * IN_W

    # stride=1, padding=1
    ih_start = oh - 1
    iw_start = ow - 1

    result = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for ki in range(3):
        for kj in range(3):
            ih = ih_start + ki
            iw = iw_start + kj
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            idx = inp_base + ih * IN_W + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            result = tl.where(valid, tl.maximum(result, val), result)

    tl.store(out_ptr + offsets, result, mask=mask)


# Adaptive avg pool 2d -> 1x1
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'IN_H', 'IN_W'],
)
@triton.jit
def adaptive_avg_pool_1x1_kernel(
    inp_ptr, out_ptr,
    N, C, IN_H, IN_W,
    BLOCK_SIZE: tl.constexpr,
):
    total = N * C
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < total

    c = offsets % C
    n = offsets // C

    inp_base = n * C * IN_H * IN_W + c * IN_H * IN_W
    area = IN_H * IN_W

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in range(IN_H):
        for j in range(IN_W):
            idx = inp_base + i * IN_W + j
            val = tl.load(inp_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    result = acc / area
    tl.store(out_ptr + offsets, result, mask=mask)


# Linear kernel (fused matmul + bias)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, bias_ptr, out_ptr,
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

        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc=acc)

    offs_m_out = offs_m
    offs_n_out = offs_n
    out_mask = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    bias = tl.load(bias_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m_out[:, None] * N + offs_n_out[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def conv2d_bias_relu(x, weight, bias, stride=1, padding=0):
    """Conv2d + bias + relu using cuDNN conv then fused bias+relu in Triton"""
    # Use PyTorch conv (cuDNN) without bias, then fuse bias+relu
    out = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_relu_kernel[grid](out, bias, result, n_elements, C)
    return result


def conv2d_bias(x, weight, bias, stride=1, padding=0):
    """Conv2d + bias (no relu) using cuDNN conv then fused bias in Triton"""
    out = torch.nn.functional.conv2d(x, weight, bias=None, stride=stride, padding=padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_kernel[grid](out, bias, result, n_elements, C)
    return result


def maxpool_k3s2p1(x):
    """MaxPool2d kernel=3, stride=2, padding=1"""
    N, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H + 2 * 1 - 3) // 2 + 1
    OUT_W = (IN_W + 2 * 1 - 3) // 2 + 1
    # Make contiguous in NCHW
    x_cont = x.contiguous()
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    total = N * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_k3s2p1_kernel[grid](x_cont, out, N, C, IN_H, IN_W, OUT_H, OUT_W)
    return out


def maxpool_k3s1p1(x):
    """MaxPool2d kernel=3, stride=1, padding=1"""
    N, C, IN_H, IN_W = x.shape
    OUT_H = IN_H
    OUT_W = IN_W
    x_cont = x.contiguous()
    out = torch.empty(N, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    total = N * C * OUT_H * OUT_W
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_k3s1p1_kernel[grid](x_cont, out, N, C, IN_H, IN_W, OUT_H, OUT_W)
    return out


def adaptive_avg_pool_1x1(x):
    """Adaptive avg pool to 1x1"""
    N, C, IN_H, IN_W = x.shape
    x_cont = x.contiguous()
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    total = N * C
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    adaptive_avg_pool_1x1_kernel[grid](x_cont, out, N, C, IN_H, IN_W)
    return out


def linear_fused(x, weight, bias):
    """Linear layer with fused bias"""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    """Inception module: 4 branches concatenated"""
    # Branch 1: 1x1 conv + relu
    branch1 = conv2d_bias_relu(x, w_1x1, b_1x1, stride=1, padding=0)

    # Branch 2: 1x1 reduce + relu, then 3x3 + relu
    branch2 = conv2d_bias_relu(x, w_3x3_r, b_3x3_r, stride=1, padding=0)
    branch2 = conv2d_bias_relu(branch2, w_3x3, b_3x3, stride=1, padding=1)

    # Branch 3: 1x1 reduce + relu, then 5x5 + relu
    branch3 = conv2d_bias_relu(x, w_5x5_r, b_5x5_r, stride=1, padding=0)
    branch3 = conv2d_bias_relu(branch3, w_5x5, b_5x5, stride=1, padding=2)

    # Branch 4: maxpool(3,1,1) then 1x1 + relu
    branch4 = maxpool_k3s1p1(x)
    branch4 = conv2d_bias_relu(branch4, w_pool, b_pool, stride=1, padding=0)

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
    # Initial layers
    x = conv2d_bias_relu(x, p_conv1_weight, p_conv1_bias, stride=2, padding=3)
    x = maxpool_k3s2p1(x)
    x = conv2d_bias_relu(x, p_conv2_weight, p_conv2_bias, stride=1, padding=0)
    x = conv2d_bias_relu(x, p_conv3_weight, p_conv3_bias, stride=1, padding=1)
    x = maxpool_k3s2p1(x)

    # Inception 3a
    x = inception_module(x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
        p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    )

    # Inception 3b
    x = inception_module(x,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
        p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
        p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
        p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
        p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    )

    x = maxpool_k3s2p1(x)

    # Inception 4a
    x = inception_module(x,
        p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
        p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
        p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
        p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
        p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
        p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    )

    # Inception 4b
    x = inception_module(x,
        p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
        p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
        p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
        p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
        p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
        p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    )

    # Inception 4c
    x = inception_module(x,
        p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
        p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
        p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
        p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
        p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
        p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias,
    )

    # Inception 4d
    x = inception_module(x,
        p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
        p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
        p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
        p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
        p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
        p_inception4d_branch_pool_1_weight, p_inception4d_branch_pool_1_bias,
    )

    # Inception 4e
    x = inception_module(x,
        p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
        p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
        p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,