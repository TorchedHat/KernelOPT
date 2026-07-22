import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernels for fused operations
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
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


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
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    tl.store(out_ptr + offsets, out, mask=mask)


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
def matmul_bias_kernel(
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
    
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc)
    
    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]
    
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=c_mask)


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
def max_pool2d_kernel(
    inp_ptr, out_ptr,
    N, C, H, W,
    OH, OW,
    KH, KW,
    SH, SW,
    PH, PW,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Decode output index: layout is N, C, OH, OW
    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)
    
    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)
    
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & (n < N)
            inp_idx = n * C * H * W + c * H * W + ih * W + iw
            val = tl.load(inp_ptr + inp_idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid, tl.maximum(max_val, val), max_val)
    
    tl.store(out_ptr + offsets, max_val, mask=mask)


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
def adaptive_avg_pool2d_kernel(
    inp_ptr, out_ptr,
    N, C, H, W,
    OH, OW,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)
    
    # For adaptive avg pool with output 1x1, we average all spatial positions
    # h_start = oh * H // OH, h_end = (oh+1) * H // OH
    # w_start = ow * W // OW, w_end = (ow+1) * W // OW
    h_start = oh * H // OH
    h_end = (oh + 1) * H // OH
    w_start = ow * W // OW
    w_end = (ow + 1) * W // OW
    
    pool_h = h_end - h_start
    pool_w = w_end - w_start
    pool_size = pool_h * pool_w
    
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    for kh in range(H):  # Will be bounded by h_start/h_end logic
        for kw in range(W):
            valid_h = (kh >= h_start) & (kh < h_end)
            valid_w = (kw >= w_start) & (kw < w_end)
            valid = mask & valid_h & valid_w & (n < N)
            inp_idx = n * C * H * W + c * H * W + kh * W + kw
            val = tl.load(inp_ptr + inp_idx, mask=valid, other=0.0)
            acc = tl.where(valid, acc + val, acc)
    
    result = acc / pool_size.to(tl.float32)
    tl.store(out_ptr + offsets, result, mask=mask)


# ============================================================
# Implicit GEMM convolution kernel
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M_total', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    SH, SW,
    PH, PW,
    M_total, TOTAL_K,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wn, stride_wc, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yh, stride_yw,
    HAS_BIAS: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    
    # Decode M index: m = n_batch * OUT_H * OUT_W + oh * OUT_W + ow
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    n_batch = offs_m // (OUT_W * OUT_H)
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        
        # Decode K index: k = ic * KH * KW + kh * KW + kw
        kw_idx = offs_k % KW
        kh_idx = (offs_k // KW) % KH
        ic_idx = offs_k // (KW * KH)
        
        ih = oh[:, None] * SH + kh_idx[None, :] - PH
        iw = ow[:, None] * SW + kw_idx[None, :] - PW
        
        valid_x = (
            (n_batch[:, None] < N) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (ic_idx[None, :] < IN_C) &
            (offs_k[None, :] < TOTAL_K)
        )
        
        x_ptrs = (X_ptr +
                  n_batch[:, None] * stride_xn +
                  ic_idx[None, :] * stride_xc +
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)
        
        valid_w = (
            (offs_k[:, None] < TOTAL_K) &
            (ic_idx[:, None] < IN_C) &
            (offs_n[None, :] < OUT_C)
        )
        
        w_ptrs = (W_ptr +
                  offs_n[None, :] * stride_wn +
                  ic_idx[:, None] * stride_wc +
                  kh_idx[:, None] * stride_wh +
                  kw_idx[:, None] * stride_ww)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)
        
        acc = tl.dot(x_tile, w_tile, acc)
    
    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + bias[None, :]
    
    if APPLY_RELU:
        acc = tl.where(acc > 0.0, acc, 0.0)
    
    out_mask = (
        (n_batch[:, None] < N) &
        (oh[:, None] < OUT_H) &
        (ow[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )
    
    y_ptrs = (Y_ptr +
              n_batch[:, None] * stride_yn +
              offs_n[None, :] * stride_yc +
              oh[:, None] * stride_yh +
              ow[:, None] * stride_yw)
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_conv2d(x, weight, bias, stride=(1,1), padding=(0,0), apply_relu=False):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding
    
    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1
    
    M_total = N * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW
    
    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    
    grid = lambda meta: (
        triton.cdiv(M_total, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    
    conv2d_implicit_gemm_kernel[grid](
        y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        SH, SW,
        PH, PW,
        M_total, TOTAL_K,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        y.stride(0), y.stride(1), y.stride(2), y.stride(3),
        HAS_BIAS=(bias is not None),
        APPLY_RELU=apply_relu,
    )
    return y


def triton_max_pool2d(x, kernel_size=3, stride=2, padding=1):
    N, C, H, W = x.shape
    KH = KW = kernel_size
    SH = SW = stride
    PH = PW = padding
    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1
    
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    n_elements = N * C * OH * OW
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        n_elements,
    )
    return out


def triton_adaptive_avg_pool2d_1x1(x):
    N, C, H, W = x.shape
    OH, OW = 1, 1
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    n_elements = N * C * OH * OW
    
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        OH, OW,
        n_elements,
    )
    return out


def inception_module(x,
                     w_1x1, b_1x1,
                     w_3x3_r, b_3x3_r, w_3x3, b_3x3,
                     w_5x5_r, b_5x5_r, w_5x5, b_5x5,
    # Branch 1x1
    b1 = triton_conv2d(x, w_1x1, b_1x1, stride=(1,1), padding=(0,0), apply_relu=True)
    
    # Branch 3x3
    b3_r = triton_conv2d(x, w_3x3_r, b_3x3_r, stride=(1,1), padding=(0,0), apply_relu=True)
    b3 = triton_conv2d(b3_r, w_3x3, b_3x3, stride=(1,1), padding=(1,1), apply_relu=True)
    
    # Branch 5x5
    b5_r = triton_conv2d(x, w_5x5_r, b_5x5_r, stride=(1,1), padding=(0,0), apply_relu=True)
    b5 = triton_conv2d(b5_r, w_5x5, b_5x5, stride=(1,1), padding=(2,2), apply_relu=True)
    
    # Branch pool
    bp = triton_max_pool2d(x, kernel_size=3, stride=1, padding=1)
    bp = triton_conv2d(bp, w_pool, b_pool, stride=(1,1), padding=(0,0), apply_relu=True)
    
    return torch.cat([b1, b3, b5, bp], dim=1)


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
    # conv1: 3->64, 7x7, stride=2, pad=3
    x = triton_conv2d(x, p_conv1_weight, p_conv1_bias, stride=(2,2), padding=(3,3), apply_relu=True)
    # maxpool1: 3x3, stride=2, pad=1
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # conv2: 64->64, 1x1
    x = triton_conv2d(x, p_conv2_weight, p_conv2_bias, stride=(1,1), padding=(0,0), apply_relu=True)
    # conv3: 64->192, 3x3, pad=1
    x = triton_conv2d(x, p_conv3_weight, p_conv3_bias, stride=(1,1), padding=(1,1), apply_relu=True)
    # maxpool2: 3x3, stride=2, pad=1
    x = triton_max_pool2d(x, kernel_size=3, stride=2, padding=1)
    
    # inception3a: in=192, out_1x1=64, reduce_3x3=96, out_3x3=128, reduce_5x5=16, out_5x5=32, pool_proj=32
    x = inception_module(x,
        p_inception3a_branch1x1_weight, p_inception3a_branch1x1_bias,
        p_inception3a_branch3x3_0_weight, p_inception3a_branch3x3_0_bias,
        p_inception3a_branch3x3_1_weight, p_inception3a_branch3x3_1_bias,
        p_inception3a_branch5x5_0_weight, p_inception3a_branch5x5_0_bias,
        p_inception3a_branch5x5_1_weight, p_inception3a_branch5x5_1_bias,
        p_inception3a_branch_pool_1_weight, p_inception3a_branch_pool_1_bias,
    )
    
    # inception3b: in=256, out_1x1=128, reduce_3x3=128, out_3