import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def bias_relu_kernel(x_ptr, bias_ptr, out_ptr, n_elements, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = offsets % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


import torch
import triton
import triton.language as tl
import torch.nn.functional as F


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements', 'C'],
)
@triton.jit
def fused_bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = offsets % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    
    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements', 'C'],
)
@triton.jit
def fused_bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = offsets % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    result = x + bias
    result = tl.where(result > 0.0, result, 0.0)
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements', 'C'],
)
@triton.jit
def fused_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = offsets % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)

    result = x + bias
    tl.store(out_ptr + offsets, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N', 'C', 'OH', 'OW', 'IH', 'IW'],
)
@triton.jit
def max_pool2d_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    KH, KW, SH, SW, PH, PW,
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
            max_val = tl.where(valid & (val > max_val), val, max_val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N', 'C', 'OH', 'OW'],
)
@triton.jit
def adaptive_avg_pool2d_1x1_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
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

    for ih in range(IH):
        for iw in range(IW):
            idx = n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=mask, other=0.0)
            acc = acc + val

    result = acc / count
    out_idx = n * C + c
    tl.store(out_ptr + out_idx, result, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['n_elements'],
)
@triton.jit
def linear_bias_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, out_features,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = offsets % out_features
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    result = x + bias
    tl.store(out_ptr + offsets, result, mask=mask)


def conv2d_bias_relu(x, weight, bias, stride=1, padding=0):
    # Use PyTorch conv (cuDNN), then fuse bias+relu in Triton
    # conv2d without bias
    out = torch.nn.functional.conv2d(x, weight, None, stride, padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bias_relu_kernel[grid](
        out, bias, result,
        n_elements, C,
    )
    return result


def conv2d_bias(x, weight, bias, stride=1, padding=0):
    out = torch.nn.functional.conv2d(x, weight, None, stride, padding)
    N, C, H, W = out.shape
    n_elements = N * C * H * W
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    fused_bias_kernel[grid](
        out, bias, result,
        n_elements, C,
    )
    return result


def triton_max_pool2d(x, kernel_size, stride, padding):
    N, C, IH, IW = x.shape
    OH = (IH + 2 * padding - kernel_size) // stride + 1
    OW = (IW + 2 * padding - kernel_size) // stride + 1
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out,
        N, C, IH, IW, OH, OW,
        kernel_size, kernel_size, stride, stride, padding, padding,
    )
    return out


def triton_adaptive_avg_pool2d_1x1(x):
    N, C, IH, IW = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    total = N * C
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_1x1_kernel[grid](
        x, out,
        N, C, IH, IW, 1, 1,
    )
    return out


def inception_module(
    x,
    w_1x1, b_1x1,
    w_3x3_reduce, b_3x3_reduce, w_3x3, b_3x3,
    w_5x5_reduce, b_5x5_reduce, w_5x5, b_5x5,
    w_pool_proj, b_pool_proj,
):
    # Branch 1x1
    branch1 = conv2d_bias_relu(x, w_1x1, b_1x1, stride=1, padding=0)

    # Branch 3x3
    b3_reduce = conv2d_bias_relu(x, w_3x3_reduce, b_3x3_reduce, stride=1, padding=0)
    branch3 = conv2d_bias_relu(b3_reduce, w_3x3, b_3x3, stride=1, padding=1)

    # Branch 5x5
    b5_reduce = conv2d_bias_relu(x, w_5x5_reduce, b_5x5_reduce, stride=1, padding=0)
    branch5 = conv2d_bias_relu(b5_reduce, w_5x5, b_5x5, stride=1, padding=2)

    # Branch pool
    pool = triton_max_pool2d(x, kernel_size=3, stride=1, padding=1)
    branch_pool = conv2d_bias_relu(pool, w_pool_proj, b_pool_proj, stride=1, padding=0)

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
    # conv1 + relu
    x = conv2d_bias_relu(x, p_conv1_weight, p_conv1_bias, stride=2, padding=3)
    # maxpool1
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)
    # conv2 + relu
    x = conv2d_bias_relu(x, p_conv2_weight, p_conv2_bias, stride=1, padding=0)
    # conv3 + relu
    x = conv2d_bias_relu(x, p_conv3_weight, p_conv3_bias, stride=1, padding=1)
    # maxpool2
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)

    # inception3a
    x = inception_module(
        x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
        p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    )

    # inception3b
    x = inception_module(
        x,
        p_inception3b_branch1x1_weight, p_inception3b_branch1x1_bias,
        p_inception3b_branch3x3_0_weight, p_inception3b_branch3x3_0_bias,
        p_inception3b_branch3x3_1_weight, p_inception3b_branch3x3_1_bias,
        p_inception3b_branch5x5_0_weight, p_inception3b_branch5x5_0_bias,
        p_inception3b_branch5x5_1_weight, p_inception3b_branch5x5_1_bias,
        p_inception3b_branch_pool_1_weight, p_inception3b_branch_pool_1_bias,
    )

    # maxpool3
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)

    # inception4a
    x = inception_module(
        x,
        p_inception4a_branch1x1_weight, p_inception4a_branch1x1_bias,
        p_inception4a_branch3x3_0_weight, p_inception4a_branch3x3_0_bias,
        p_inception4a_branch3x3_1_weight, p_inception4a_branch3x3_1_bias,
        p_inception4a_branch5x5_0_weight, p_inception4a_branch5x5_0_bias,
        p_inception4a_branch5x5_1_weight, p_inception4a_branch5x5_1_bias,
        p_inception4a_branch_pool_1_weight, p_inception4a_branch_pool_1_bias,
    )

    # inception4b
    x = inception_module(
        x,
        p_inception4b_branch1x1_weight, p_inception4b_branch1x1_bias,
        p_inception4b_branch3x3_0_weight, p_inception4b_branch3x3_0_bias,
        p_inception4b_branch3x3_1_weight, p_inception4b_branch3x3_1_bias,
        p_inception4b_branch5x5_0_weight, p_inception4b_branch5x5_0_bias,
        p_inception4b_branch5x5_1_weight, p_inception4b_branch5x5_1_bias,
        p_inception4b_branch_pool_1_weight, p_inception4b_branch_pool_1_bias,
    )

    # inception4c
    x = inception_module(
        x,
        p_inception4c_branch1x1_weight, p_inception4c_branch1x1_bias,
        p_inception4c_branch3x3_0_weight, p_inception4c_branch3x3_0_bias,
        p_inception4c_branch3x3_1_weight, p_inception4c_branch3x3_1_bias,
        p_inception4c_branch5x5_0_weight, p_inception4c_branch5x5_0_bias,
        p_inception4c_branch5x5_1_weight, p_inception4c_branch5x5_1_bias,
        p_inception4c_branch_pool_1_weight, p_inception4c_branch_pool_1_bias,
    )

    # inception4d
    x = inception_module(
        x,
        p_inception4d_branch1x1_weight, p_inception4d_branch1x1_bias,
        p_inception4d_branch3x3_0_weight, p_inception4d_branch3x3_0_bias,
        p_inception4d_branch3x3_1_weight, p_inception4d_branch3x3_1_bias,
        p_inception4d_branch5x5_0_weight, p_inception4d_branch5x5_0_bias,
        p_inception4d_branch5x5_1_weight, p_inception4d_branch5x5_1_bias,
        p_inception4d_branch_pool_1_weight, p_inception4d_branch_pool_1_bias,
    )

    # inception4e
    x = inception_module(
        x,
        p_inception4e_branch1x1_weight, p_inception4e_branch1x1_bias,
        p_inception4e_branch3x3_0_weight, p_inception4e_branch3x3_0_bias,
        p_inception4e_branch3x3_1_weight, p_inception4e_branch3x3_1_bias,
        p_inception4e_branch5x5_0_weight, p_inception4e_branch5x5_0_bias,
        p_inception4e_branch5x5_1_weight, p_inception4e_branch5x5_1_bias,
        p_inception4e_branch_pool_1_weight, p_inception4e_branch_pool_1_bias,
    )

    # maxpool4
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)

    # inception5a
    x = inception_module(
        x,
        p_inception5a_branch1x1_weight, p_inception5a_branch1x1_bias,
        p_inception5a_branch3x3_0_weight, p_inception5a_branch3x3_0_bias,
        p_inception5a_branch3x3_1_weight, p_inception5a_branch3x3_1_bias,
        p_inception5a_branch5x5_0_weight, p_inception5a_branch5x5_0_bias,
        p_inception5a_branch5x5_1_weight, p_inception5a_branch5x5_1_bias,
        p_inception5a_branch_pool_1_weight, p_inception5a_branch_pool_1_bias,
    )

    # inception5b
    x = inception_module(
        x,
        p_inception5b_branch1x1_weight, p_inception5b_branch1x1_bias,
        p_inception5b_branch3x3_0_weight, p_inception5b_branch3x3_0_bias,
        p_inception5b_branch3x3_1_weight, p_inception5b_branch3x3_1_bias,
        p_inception5b_branch5x5_0_weight, p_inception5b_branch5x5_0_bias,
        p_inception5b_branch5x5_1_weight, p_inception5b_branch5x5_1_bias,
        p_inception5b_branch_pool_1_weight, p_inception5b_branch_pool_1_bias,
    )

    # adaptive avg pool
    x = triton_adaptive_avg_pool2d_1x1(x)

    # flatten
    x = x.reshape(x.shape[0], -1)

    # linear (matmul + bias) - use torch.mm then fuse bias
    out = torch.mm(x, p_fc_weight.t())
    n_elements = out.numel()
    result = torch.empty_like(out)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    linear_bias_kernel[grid](
        out, p_fc_bias, result,
        n_elements, p_fc_bias.shape[0],
    )
    return result


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'

    def make_weight(shape):
        return torch.randn(shape, device=device, dtype=torch.float32) * 0.01

    def make_bias(shape):
        return torch.zeros(shape, device=device, dtype=torch.float32)

    inputs = [
        make_weight([64, 3, 7, 7]),    # p_conv1_weight
        make_bias([64]),                # p_conv1_bias
        make_weight([64, 64, 1, 1]),   # p_conv2_weight
        make_bias([64]),                # p_conv2_bias
        make_weight([192, 64, 3,