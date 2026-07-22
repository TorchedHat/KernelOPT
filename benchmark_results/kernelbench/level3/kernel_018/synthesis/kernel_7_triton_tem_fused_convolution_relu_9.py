import torch
import triton
import triton.language as tl
import math


# ─── Fused Conv2d + Bias + ReLU (Implicit GEMM) ───────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    # tensor dims
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides for X (NHWC layout)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (NCHW -> we'll use KCRS layout)
    sw_o, sw_i, sw_h, sw_w,
    # strides for Y (NHWC layout)
    sy_n, sy_h, sy_w, sy_c,
    # GEMM dims
    M, N, K,
    # apply relu?
    APPLY_RELU: tl.constexpr,
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

    # K = IN_C * KH * KW
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KW
        kh_idx = tmp_k // KW

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            (offs_m[:, None] < M) &
            (batch_idx[:, None] < BATCH) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < K)
        )

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * sx_n +
                  in_c_idx[None, :] * sx_c)
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: shape (OUT_C, IN_C, KH, KW) -> index [n, in_c, kh, kw]
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * sw_o +
                  in_c_idx[:, None] * sw_i +
                  kh_idx[:, None] * sw_h +
                  kw_idx[:, None] * sw_w)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output (NHWC)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * sy_n +
              out_h_idx[:, None] * sy_h +
              out_w_idx[:, None] * sy_w +
              offs_n[None, :] * sy_c)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv2d_relu(x_nhwc, weight, bias, stride=1, padding=0, apply_relu=True):
    """
    returns: (N, OUT_H, OUT_W, OUT_C) contiguous
    """
    BATCH, IN_H, IN_W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = weight.shape

    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    OUT_H = (IN_H + 2 * padding[0] - KH) // stride[0] + 1
    OUT_W = (IN_W + 2 * padding[1] - KW) // stride[1] + 1

    y = torch.empty((BATCH, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    M = BATCH * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KH * KW

    # strides for x_nhwc
    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    # strides for weight (OUT_C, IN_C, KH, KW)
    sw_o, sw_i, sw_h, sw_w = weight.stride()
    # strides for y (NHWC)
    sy_n, sy_h, sy_w, sy_c = y.stride()

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )

    conv2d_relu_kernel[grid](
        x_nhwc, weight, bias, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride[0], stride[1],
        padding[0], padding[1],
        sx_n, sx_h, sx_w, sx_c,
        sw_o, sw_i, sw_h, sw_w,
        sy_n, sy_h, sy_w, sy_c,
        M, N, K,
        APPLY_RELU=apply_relu,
    )
    return y


# ─── Max Pool 2D ──────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def max_pool2d_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KH, KW, STRIDE_H, STRIDE_W, PAD_H, PAD_W,
    # strides (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    sy_n, sy_h, sy_w, sy_c,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: N_TOTAL = BATCH * OUT_H * OUT_W * C
    c_idx = offs % C
    tmp = offs // C
    ow_idx = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh_idx = tmp2 % OUT_H
    n_idx = tmp2 // OUT_H

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh_idx * STRIDE_H + kh - PAD_H
            iw = ow_idx * STRIDE_W + kw - PAD_W
            valid = mask & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            x_ptrs = X_ptr + n_idx * sx_n + ih * sx_h + iw * sx_w + c_idx * sx_c
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    y_ptrs = Y_ptr + n_idx * sy_n + oh_idx * sy_h + ow_idx * sy_w + c_idx * sy_c
    tl.store(y_ptrs, max_val, mask=mask)


def triton_max_pool2d(x_nhwc, kernel_size=3, stride=2, padding=0, ceil_mode=False):
    BATCH, IN_H, IN_W, C = x_nhwc.shape
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    if ceil_mode:
        OUT_H = math.ceil((IN_H + 2 * padding[0] - kernel_size[0]) / stride[0]) + 1
        OUT_W = math.ceil((IN_W + 2 * padding[1] - kernel_size[1]) / stride[1]) + 1
    else:
        OUT_H = (IN_H + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        OUT_W = (IN_W + 2 * padding[1] - kernel_size[1]) // stride[1] + 1

    y = torch.empty((BATCH, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)

    N_TOTAL = BATCH * OUT_H * OUT_W * C
    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    sy_n, sy_h, sy_w, sy_c = y.stride()

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    max_pool2d_kernel[grid](
        x_nhwc, y,
        BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
        kernel_size[0], kernel_size[1],
        stride[0], stride[1],
        padding[0], padding[1],
        sx_n, sx_h, sx_w, sx_c,
        sy_n, sy_h, sy_w, sy_c,
        N_TOTAL,
    )
    return y


# ─── Adaptive Avg Pool 2D ─────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    sx_n, sx_h, sx_w, sx_c,
    sy_n, sy_h, sy_w, sy_c,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    c_idx = offs % C
    tmp = offs // C
    ow_idx = tmp % OUT_W
    tmp2 = tmp // OUT_W
    oh_idx = tmp2 % OUT_H
    n_idx = tmp2 // OUT_H

    # Compute input window for this output position
    # start = floor(oh * IN_H / OUT_H), end = ceil((oh+1) * IN_H / OUT_H)
    ih_start = oh_idx * IN_H // OUT_H
    ih_end = (oh_idx + 1) * IN_H // OUT_H  # approximate for OUT_H=1
    iw_start = ow_idx * IN_W // OUT_W
    iw_end = (ow_idx + 1) * IN_W // OUT_W

    # For OUT_H=1, OUT_W=1: pool over entire spatial
    # We'll do a simple loop - for adaptive avg pool to (1,1)
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    count = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Since OUT_H=OUT_W=1 in our case, ih_start=0, ih_end=IN_H
    # We need to loop over all spatial positions
    for ih in range(IN_H):
        for iw in range(IN_W):
            valid = mask & (ih >= ih_start) & (ih < ih_end) & (iw >= iw_start) & (iw < iw_end)
            x_ptrs = X_ptr + n_idx * sx_n + ih * sx_h + iw * sx_w + c_idx * sx_c
            val = tl.load(x_ptrs, mask=valid, other=0.0)
            acc = acc + tl.where(valid, val, 0.0)
            count = count + tl.where(valid, 1.0, 0.0)

    result = acc / tl.maximum(count, 1.0)
    y_ptrs = Y_ptr + n_idx * sy_n + oh_idx * sy_h + ow_idx * sy_w + c_idx * sy_c
    tl.store(y_ptrs, result, mask=mask)


def triton_adaptive_avg_pool2d(x_nhwc, output_size=(1, 1)):
    BATCH, IN_H, IN_W, C = x_nhwc.shape
    OUT_H, OUT_W = output_size

    y = torch.empty((BATCH, OUT_H, OUT_W, C), device=x_nhwc.device, dtype=torch.float32)

    N_TOTAL = BATCH * OUT_H * OUT_W * C
    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    sy_n, sy_h, sy_w, sy_c = y.stride()

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x_nhwc, y,
        BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
        sx_n, sx_h, sx_w, sx_c,
        sy_n, sy_h, sy_w, sy_c,
        N_TOTAL,
    )
    return y


# ─── Cat (channel concat in NHWC) ─────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def cat_nhwc_kernel(
    A_ptr, B_ptr, Y_ptr,
    BATCH, H, W, CA, CB,
    sa_n, sa_h, sa_w, sa_c,
    sb_n, sb_h, sb_w, sb_c,
    sy_n, sy_h, sy_w, sy_c,
    N_TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    C_total = CA + CB
    c_idx = offs % C_total
    tmp = offs // C_total
    w_idx = tmp % W
    tmp2 = tmp // W
    h_idx = tmp2 % H
    n_idx = tmp2 // H

    # Load from A or B
    from_a = c_idx < CA
    c_a = tl.where(from_a, c_idx, 0)
    c_b = tl.where(from_a, 0, c_idx - CA)

    a_ptrs = A_ptr + n_idx * sa_n + h_idx * sa_h + w_idx * sa_w + c_a * sa_c
    b_ptrs = B_ptr + n_idx * sb_n + h_idx * sb_h + w_idx * sb_w + c_b * sb_c

    val_a = tl.load(a_ptrs, mask=mask & from_a, other=0.0)
    val_b = tl.load(b_ptrs, mask=mask & ~from_a, other=0.0)
    val = tl.where(from_a, val_a, val_b)

    y_ptrs = Y_ptr + n_idx * sy_n + h_idx * sy_h + w_idx * sy_w + c_idx * sy_c
    tl.store(y_ptrs, val, mask=mask)


def triton_cat_nhwc(a, b):
    """Concatenate two NHWC tensors along channel dim."""
    BATCH, H, W, CA = a.shape
    _, _, _, CB = b.shape
    C_total = CA + CB
    y = torch.empty((BATCH, H, W, C_total), device=a.device, dtype=torch.float32)

    N_TOTAL = BATCH * H * W * C_total
    sa_n, sa_h, sa_w, sa_c = a.stride()
    sb_n, sb_h, sb_w, sb_c = b.stride()
    sy_n, sy_h, sy_w, sy_c = y.stride()

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    cat_nhwc_kernel[grid](
        a, b, y,
        BATCH, H, W, CA, CB,
        sa_n, sa_h, sa_w, sa_c,
        sb_n, sb_h, sb_w, sb_c,
        sy_n, sy_h, sy_w, sy_c,
        N_TOTAL,
    )
    return y


# ─── Fire Module ──────────────────────────────────────────────────────────────

def fire_module(x_nhwc, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """
    returns: (N, H, W, C_ex1 + C_ex3)
    """
    # Squeeze: 1x1 conv + relu
    sq = conv2d_relu(x_nhwc, sq_w, sq_b, stride=1, padding=0, apply_relu=True)
    # Expand 1x1
    ex1 = conv2d_relu(sq, ex1_w, ex1_b, stride=1, padding=0, apply_relu=True)
    # Expand 3x3
    ex3 = conv2d_relu(sq, ex3_w, ex3_b, stride=1, padding=1, apply_relu=True)
    # Cat
    return triton_cat_nhwc(ex1, ex3)


# ─── Main kernel_function ─────────────────────────────────────────────────────

def kernel_function(
    p_features_0_weight, p_features_0_bias,
    p_features_3_squeeze_weight, p_features_3_squeeze_bias,
    p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,
    p_features_3_expand3x3_weight, p_features_3_expand3x3_bias,
    p_features_4_squeeze_weight, p_features_4_squeeze_bias,
    p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,
    p_features_4_expand3x3_weight, p_features_4_expand3x3_bias,
    p_features_5_squeeze_weight, p_features_5_squeeze_bias,
    p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,
    p_features_5_expand3x3_weight, p_features_5_expand3x3_bias,
    p_features_7_squeeze_weight, p_features_7_squeeze_bias,
    p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,
    p_features_7_expand3x3_weight, p_features_7_expand3x3_bias,
    p_features_8_squeeze_weight, p_features_8_squeeze_bias,
    p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,
    p_features_8_expand3x3_weight, p_features_8_expand3x3_bias,
    p_features_9_squeeze_weight, p_features_9_squeeze_bias,
    p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,
    p_features_9_expand3x3_weight, p_features_9_expand3x3_bias,
    p_features_10_squeeze_weight, p_features_10_squeeze_bias,
    p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,
    p_features_10_expand3x3_weight, p_features_10_expand3x3_bias,
    p_features_12_squeeze_weight, p_features_12_squeeze_bias,
    p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,
    p_features_12_expand3x3_weight, p_features_12_expand3x3_bias,
    p_classifier_1_weight, p_classifier_1_bias,
    x,
):
    # Convert input to NHWC
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (64, 512, 512, 3)

    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    x_nhwc = conv2d_relu(x_nhwc, p_features_0_weight, p_features_0_bias,
                         stride=2, padding=0, apply_relu=True)
    # -> (64, 253, 253, 96)

    # features[2]: MaxPool2d(3, stride=2, ceil_mode=True)
    x_nhwc = triton_max_pool2d(x_nhwc, kernel_size=3, stride=2, padding=0, ceil_mode=True)
    # -> (64, 126, 126, 96)

    # features[3]: Fire(96, 16, 64, 64)
    x_nhwc = fire_module(x_nhwc,
                         p_features_3_squeeze_weight, p_features_3_squeeze_bias,
                         p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,
    # -> (64, 126, 126, 128)

    # features[4]: Fire(128, 16, 64, 64)
    x_nhwc = fire_module(x_nhwc,
                         p_features_4_squeeze_weight, p_features_4_squeeze_bias,
                         p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,
    # -> (64, 126, 126, 128)

    # features[5]: Fire(128, 32, 128, 128)
    x_nhwc = fire_module(x_nhwc,
                         p_features_5_squeeze_weight, p_features_5_squeeze_bias,
                         p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,
    # -> (64, 126, 126, 256)

    # features[6]: MaxPool2d(3, stride=2, ceil_mode=True)
    x_nhwc = triton_max_pool2d(x_nhwc, kernel_size=3, stride=2, padding=0, ceil_mode=True)
    # -> (64, 63, 63, 256)

    # features[7]: Fire(256, 32, 128, 128)
    x_nhwc = fire_module(x_nhwc,
                         p_features_7_squeeze_weight, p_features_7_squeeze_bias,
                         p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,
    # -> (64, 63, 63, 256)

    # features[8]: Fire(256, 48, 192, 192)
    x_nhwc = fire_module(x_nhwc,
                         p_features_8_squeeze_weight, p_features_8_squeeze_bias,
                         p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,
    # -> (64, 63, 63, 384)

    # features[9]: Fire(384, 48, 192, 192)
    x_nhwc = fire_module(x_nhwc,
                         p_features_9_squeeze_weight, p_features_9_squeeze_bias,
                         p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,
    # -> (64, 63, 63, 384)

    # features[10]: Fire(384, 64, 256, 256)
    x_nhwc = fire_module(x_nhwc,
                         p_features_10_squeeze_weight, p_features_10_squeeze_bias,
                         p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,
    # -> (64, 63, 63, 512)

    # features[11]: MaxPool2d(3, stride=2, ceil_