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
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_elements'],
)
@triton.jit
def bias_relu_kernel(x_ptr, bias_ptr, out_ptr, n_elements, C, with_relu: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c_idx = offsets % C
    b = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    out = x + b
    if with_relu:
        out = tl.maximum(out, tl.zeros_like(out))
    tl.store(out_ptr + offsets, out, mask=mask)


def apply_bias_relu(x, bias, with_relu=True):
    # x: (N, C, H, W) contiguous
    N, C, H, W = x.shape
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    bias_relu_kernel[grid](x, bias, out, n_elements, C, with_relu)
    return out


# ============================================================
# Triton kernel: implicit GEMM conv2d
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    # dimensions
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N, K,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (out_c, kh, kw, in_c) layout
    s_wo, s_wh, s_ww, s_wi,
    # strides for Out (NHWC layout)
    s_on, s_oh, s_ow, s_oc,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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
        kw_idx = tmp_k % KERNEL_W
        kh_idx = tmp_k // KERNEL_W

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: (OUT_C, KH, KW, IN_C) -> index [offs_n, kh_idx, kw_idx, in_c_idx]
        valid_n = offs_n[None, :] < N
        mask_w = valid_k[:, None] & valid_n
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_ptrs = (Out_ptr
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), with_relu=True):
    """
    """
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    H_out = (H + 2*PH - KH) // SH + 1
    W_out = (W + 2*PW - KW) // SW + 1

    # Convert x to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # Convert weight to (C_out, KH, KW, C_in)
    w_conv = weight.permute(0, 2, 3, 1).contiguous()

    M = N * H_out * W_out
    Nw = C_out
    K = KH * KW * C_in

    # Output in NHWC
    out_nhwc = torch.empty((N, H_out, W_out, C_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(Nw, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_conv, out_nhwc,
        N, C_in, H, W,
        C_out, H_out, W_out,
        KH, KW,
        SH, SW,
        PH, PW,
        M, Nw, K,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w_conv.stride(0), w_conv.stride(1), w_conv.stride(2), w_conv.stride(3),
        out_nhwc.stride(0), out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
    )

    # Convert back to NCHW
    out = out_nhwc.permute(0, 3, 1, 2).contiguous()

    if bias is not None:
        out = apply_bias_relu(out, bias, with_relu=with_relu)
    elif with_relu:
        out = torch.clamp(out, min=0)

    return out


# ============================================================
# Triton kernel: max_pool2d
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['n_out'],
)
@triton.jit
def max_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    H_out, W_out,
    KH, KW,
    SH, SW,
    PH, PW,
    n_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_out

    # Decode: offsets -> (n, c, h_out, w_out)
    w_out_idx = offsets % W_out
    tmp = offsets // W_out
    h_out_idx = tmp % H_out
    tmp2 = tmp // H_out
    c_idx = tmp2 % C
    n_idx = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = h_out_idx * SH + kh - PH
            iw = w_out_idx * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x_idx = n_idx * (C * H * W) + c_idx * (H * W) + ih * W + iw
            val = tl.load(x_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offsets, max_val, mask=mask)


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    if isinstance(kernel_size, int):
        KH = KW = kernel_size
    else:
        KH, KW = kernel_size
    if isinstance(stride, int):
        SH = SW = stride
    else:
        SH, SW = stride
    if isinstance(padding, int):
        PH = PW = padding
    else:
        PH, PW = padding

    H_out = (H + 2*PH - KH) // SH + 1
    W_out = (W + 2*PW - KW) // SW + 1

    x_cont = x.contiguous()
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)
    n_out = N * C * H_out * W_out

    grid = lambda meta: (triton.cdiv(n_out, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x_cont, out,
        N, C, H, W,
        H_out, W_out,
        KH, KW,
        SH, SW,
        PH, PW,
        n_out,
    )
    return out


# ============================================================
# Triton kernel: adaptive avg pool2d (global average pooling)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 32}, num_warps=4),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    n_idx = pid // C
    c_idx = pid % C

    if n_idx >= N:
        return

    base = n_idx * C * H * W + c_idx * H * W
    total = H * W
    acc = tl.zeros((1,), dtype=tl.float32)

    for start in range(0, total, BLOCK_SIZE):
        offs = start + tl.arange(0, BLOCK_SIZE)
        mask = offs < total
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=0)

    out_val = acc / (H * W)
    tl.store(out_ptr + pid, out_val)


def triton_adaptive_avg_pool2d(x, output_size=(1, 1)):
    N, C, H, W = x.shape
    assert output_size == (1, 1) or output_size == 1
    x_cont = x.contiguous()
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    adaptive_avg_pool2d_kernel[grid](x_cont, out.view(-1), N, C, H, W)
    return out


# ============================================================
# Triton kernel: linear (GEMM) with bias
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + offs_k
        mask_a = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        mask_b = (k_offs[:, None] < K) & (offs_n[None, :] < N)

        a = tl.load(A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak,
                    mask=mask_a, other=0.0)
        b = tl.load(B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn,
                    mask=mask_b, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc.to(tl.float32), mask=mask_c)


def triton_linear(x, weight, bias):
    """
    """
    M, K = x.shape
    N = weight.shape[0]
    x_cont = x.contiguous()
    w_cont = weight.contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x_cont, w_cont, bias, out,
        M, N, K,
        x_cont.stride(0), x_cont.stride(1),
        w_cont.stride(1), w_cont.stride(0),
        out.stride(0), out.stride(1),
    )
    return out


# ============================================================
# Inception module using Triton ops
# ============================================================
def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    # Branch 1: 1x1 conv + relu
    b1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), with_relu=True)

    # Branch 2: 1x1 reduce + 3x3 conv + relu
    b2 = triton_conv2d(x, w_3x3_r, b_3x3_r, stride=(1,1), padding=(0,0), with_relu=True)
    b2 = triton_conv2d(b2, w_3x3, b_3x3, stride=(1,1), padding=(1,1), with_relu=True)

    # Branch 3: 1x1 reduce + 5x5 conv + relu
    b3 = triton_conv2d(x, w_5x5_r, b_5x5_r, stride=(1,1), padding=(0,0), with_relu=True)
    b3 = triton_conv2d(b3, w_5x5, b_5x5, stride=(1,1), padding=(2,2), with_relu=True)

    # Branch 4: max pool + 1x1 conv + relu
    b4 = triton_max_pool2d(x, kernel_size=3, stride=1, padding=1)
    b4 = triton_conv2d(b4, w_pool, b_pool, stride=(1,1), padding=(0,0), with_relu=True)

    return torch.cat([b1, b2, b3, b4], dim=1)


# ============================================================
# Main kernel_function
# ============================================================
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
    x = triton_conv2d(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), with_relu=True)
    # maxpool1: 3x3, stride=2, padding=1
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)
    # conv2: 1x1
    x = triton_conv2d(x, p_conv2_weight, p_conv2_bias, stride=(1,1