import torch
import triton
import triton.language as tl
import math


import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: fused bias + ReLU (pointwise epilogue)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def bias_relu_kernel(x_ptr, bias_ptr, out_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c = offs % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def bias_kernel(x_ptr, bias_ptr, out_ptr, N, C, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c = offs % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    tl.store(out_ptr + offs, out, mask=mask)


# ============================================================
# Triton kernel: max pool 2x2 stride 2
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def maxpool2d_kernel(
    inp_ptr, out_ptr,
    N, C, H, W, OH, OW,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # decode: offs = n*C*OH*OW + c*OH*OW + oh*OW + ow
    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C
    ih0 = oh * 2
    iw0 = ow * 2
    ih1 = ih0 + 1
    iw1 = iw0 + 1
    base = n * C * H * W + c * H * W
    idx00 = base + ih0 * W + iw0
    idx01 = base + ih0 * W + iw1
    idx10 = base + ih1 * W + iw0
    idx11 = base + ih1 * W + iw1
    v00 = tl.load(inp_ptr + idx00, mask=mask, other=float('-inf'))
    v01 = tl.load(inp_ptr + idx01, mask=mask, other=float('-inf'))
    v10 = tl.load(inp_ptr + idx10, mask=mask, other=float('-inf'))
    v11 = tl.load(inp_ptr + idx11, mask=mask, other=float('-inf'))
    vmax = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    tl.store(out_ptr + offs, vmax, mask=mask)


# ============================================================
# Triton GEMM kernel (for linear layers)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(
    A_ptr, B_ptr, C_ptr,
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
    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)
        acc = tl.dot(a, b, acc, allow_tf32=False)
    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


# ============================================================
# Triton implicit GEMM conv2d kernel (NCHW input, NCHW output)
# Uses channels-last internally for coalesced access
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_bias_relu_kernel(
    X_ptr, W_ptr, bias_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # X strides (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # W strides (OUT_C, IN_C, KH, KW) -> flattened as (OUT_C, IN_C*KH*KW)
    # We pass W in (OUT_C, IN_C, KH, KW) and decode
    s_wo, s_wi, s_wkh, s_wkw,
    do_relu: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    M_TOTAL = BATCH * OUT_H * OUT_W
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KERNEL_W
        kh_idx = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = oh[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_h & valid_w

        x_ptrs = X_ptr + batch[:, None] * s_xn + ih * s_xh + iw * s_xw + in_c[None, :] * s_xc
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: (OUT_C, IN_C, KH, KW) -> access [offs_n, in_c, kh_idx, kw_idx]
        mask_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + offs_n[None, :] * s_wo + in_c[:, None] * s_wi + kh_idx[:, None] * s_wkh + kw_idx[:, None] * s_wkw
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias_mask = offs_n < OUT_C
    bias = tl.load(bias_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    if do_relu:
        acc = tl.where(acc > 0.0, acc, 0.0)

    # Store output in NCHW: Y[n, c, h, w] = Y[n*OUT_C*OUT_H*OUT_W + c*OUT_H*OUT_W + h*OUT_W + w]
    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    # offs_m = n*OUT_H*OUT_W + oh*OUT_W + ow
    # output index: batch*OUT_C*OUT_H*OUT_W + c*OUT_H*OUT_W + oh*OUT_W + ow
    out_idx = batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + oh[:, None] * OUT_W + ow[:, None]
    tl.store(Y_ptr + out_idx, acc, mask=out_mask)


def conv2d_triton(x_nhwc, w, bias, out_h, out_w, stride_h, stride_w, pad_h, pad_w, do_relu=True):
    """x_nhwc: (N, H, W, C), w: (OUT_C, IN_C, KH, KW)"""
    N, H, W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w.shape
    M_TOTAL = N * out_h * out_w
    y = torch.empty(N, OUT_C, out_h, out_w, device=x_nhwc.device, dtype=x_nhwc.dtype)

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    conv2d_bias_relu_kernel[grid](
        x_nhwc, w, bias, y,
        N, IN_C, H, W,
        OUT_C, out_h, out_w,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        do_relu,
    )
    return y


def maxpool2d_triton(x):
    """x: (N, C, H, W) -> (N, C, H//2, W//2)"""
    N, C, H, W = x.shape
    OH, OW = H // 2, W // 2
    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](x, out, total, C, H, W, OH, OW)
    return out


def linear_triton(x, weight, bias, do_relu=False):
    """x: (M, K), weight: (N, K) -> (M, N)"""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    gemm_kernel[grid](
        x, weight, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(1), weight.stride(0),  # weight is (N, K), so stride_bk=weight.stride(1), stride_bn=weight.stride(0)
        out.stride(0), out.stride(1),
    )
    # Add bias + optional relu
    total = M * N
    grid2 = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    if do_relu:
        bias_relu_kernel[grid2](out, bias, out, total, N)
    else:
        bias_kernel[grid2](out, bias, out, total, N)
    return out


def kernel_function(
    p_features_0_weight, p_features_0_bias,
    p_features_2_weight, p_features_2_bias,
    p_features_5_weight, p_features_5_bias,
    p_features_7_weight, p_features_7_bias,
    p_features_10_weight, p_features_10_bias,
    p_features_12_weight, p_features_12_bias,
    p_features_14_weight, p_features_14_bias,
    p_features_16_weight, p_features_16_bias,
    p_features_19_weight, p_features_19_bias,
    p_features_21_weight, p_features_21_bias,
    p_features_23_weight, p_features_23_bias,
    p_features_25_weight, p_features_25_bias,
    p_features_28_weight, p_features_28_bias,
    p_features_30_weight, p_features_30_bias,
    p_features_32_weight, p_features_32_bias,
    p_features_34_weight, p_features_34_bias,
    p_classifier_0_weight, p_classifier_0_bias,
    p_classifier_3_weight, p_classifier_3_bias,
    p_classifier_6_weight, p_classifier_6_bias,
    x,
):
    # Convert input to NHWC for conv kernels
    def to_nhwc(t):
        return t.permute(0, 2, 3, 1).contiguous()

    def from_nchw_to_nhwc_after_pool(t):
        return to_nhwc(t)

    # Block 1
    x_nhwc = to_nhwc(x)  # (10, 224, 224, 3)
    # conv1: (10,3,224,224) -> (10,64,224,224), pad=1, stride=1
    out = conv2d_triton(x_nhwc, p_features_0_weight, p_features_0_bias, 224, 224, 1, 1, 1, 1, do_relu=True)
    # out is (10, 64, 224, 224) NCHW
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_2_weight, p_features_2_bias, 224, 224, 1, 1, 1, 1, do_relu=True)
    # maxpool
    out = maxpool2d_triton(out)  # (10, 64, 112, 112)

    # Block 2
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_5_weight, p_features_5_bias, 112, 112, 1, 1, 1, 1, do_relu=True)
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_7_weight, p_features_7_bias, 112, 112, 1, 1, 1, 1, do_relu=True)
    out = maxpool2d_triton(out)  # (10, 128, 56, 56)

    # Block 3
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_10_weight, p_features_10_bias, 56, 56, 1, 1, 1, 1, do_relu=True)
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_12_weight, p_features_12_bias, 56, 56, 1, 1, 1, 1, do_relu=True)
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_14_weight, p_features_14_bias, 56, 56, 1, 1, 1, 1, do_relu=True)
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_16_weight, p_features_16_bias, 56, 56, 1, 1, 1, 1, do_relu=True)
    out = maxpool2d_triton(out)  # (10, 256, 28, 28)

    # Block 4
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_19_weight, p_features_19_bias, 28, 28, 1, 1, 1, 1, do_relu=True)
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_21_weight, p_features_21_bias, 28, 28, 1, 1, 1, 1, do_relu=True)
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_23_weight, p_features_23_bias, 28, 28, 1, 1, 1, 1, do_relu=True)
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_25_weight, p_features_25_bias, 28, 28, 1, 1, 1, 1, do_relu=True)
    out = maxpool2d_triton(out)  # (10, 512, 14, 14)

    # Block 5
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_28_weight, p_features_28_bias, 14, 14, 1, 1, 1, 1, do_relu=True)
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_30_weight, p_features_30_bias, 14, 14, 1, 1, 1, 1, do_relu=True)
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_32_weight, p_features_32_bias, 14, 14, 1, 1, 1, 1, do_relu=True)
    x_nhwc = to_nhwc(out)
    out = conv2d_triton(x_nhwc, p_features_34_weight, p_features_34_bias, 14, 14, 1, 1, 1, 1, do_relu=True)
    out = maxpool2d_triton(out)  # (10, 512, 7, 7)

    # Flatten
    out = out.reshape(10, 25088)

    # Classifier
    out = linear_triton(out, p_classifier_0_weight, p_classifier_0_bias, do_relu=True)
    # dropout p=0.0 is identity
    out = linear_triton(out, p_classifier_3_weight, p_classifier_3_bias, do_relu=True)
    # dropout p=0.0 is identity
    out = linear_triton(out, p_classifier_6_weight, p_classifier_6_bias, do_relu=False)

    return out


def get_inputs():
    torch.manual_seed(42)
    p_features_0_weight = torch.randn(64, 3, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_0_bias = torch.zeros(64, device='cuda', dtype=torch.float32)
    p_features_2_weight = torch.randn(64, 64, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_2_bias = torch.zeros(64, device='cuda', dtype=torch.float32)
    p_features_5_weight = torch.randn(128, 64, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_5_bias = torch.zeros(128, device='cuda', dtype=torch.float32)
    p_features_7_weight = torch.randn(128, 128, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_7_bias = torch.zeros(128, device='cuda', dtype=torch.float32)
    p_features_10_weight = torch.randn(256, 128, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_10_bias = torch.zeros(256, device='cuda', dtype=torch.float32)
    p_features_12_weight = torch.randn(256, 256, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_12_bias = torch.zeros(256, device='cuda', dtype=torch.float32)
    p_features_14_weight = torch.randn(256, 256, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_14_bias = torch.zeros(256, device='cuda', dtype=torch.float32)
    p_features_16_weight = torch.randn(256, 256, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_16_bias = torch.zeros(256, device='cuda', dtype=torch.float32)
    p_features_19_weight = torch.randn(512, 256, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_19_bias = torch.zeros(512, device='cuda', dtype=torch.float32)
    p_features_21_weight = torch.randn(512, 512, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_21_bias = torch.zeros(512, device='cuda', dtype=torch.float32)
    p_features_23_weight = torch.randn(512, 512, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_23_bias = torch.zeros(512, device='cuda', dtype=torch.float32)
    p_features_25_weight = torch.randn(512, 512, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_25_bias = torch.zeros(512, device='cuda', dtype=torch.float32)
    p_features_28_weight = torch.randn(512, 512, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_28_bias = torch.zeros(512, device='cuda', dtype=torch.float32)
    p_features_30_weight = torch.randn(512, 512, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_30_bias = torch.zeros(512, device='cuda', dtype=torch.float32)
    p_features_32_weight = torch.randn(512, 512, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_32_bias = torch.zeros(512, device='cuda', dtype=torch.float32)
    p_features_34_weight = torch.randn(512, 512, 3, 3, device='cuda', dtype=torch.float32) * 0.01
    p_features_34_bias = torch.zeros(512, device='cuda', dtype=torch.float32)
    p_classifier_0_weight = torch.randn(4096, 25088, device='cuda', dtype=torch.float32) * 0.01
    p_classifier_0_bias = torch.zeros