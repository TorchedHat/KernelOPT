import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % channels
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
    key=['numel'],
)
@triton.jit
def bias_relu_nhwc_kernel(
    x_ptr,
    bias_ptr,
    out_ptr,
    numel,
    channels,
    BLOCK_SIZE: tl.constexpr,
):
    """For NHWC layout where channel is last dimension"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % channels
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
    key=['numel'],
)
@triton.jit
def maxpool2d_kernel(
    x_ptr,
    out_ptr,
    N, C, H, W,
    OH, OW,
    KH, KW,
    SH, SW,
    PH, PW,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Decode output index: layout is NCHW
    ow = offsets % OW
    oh = (offsets // OW) % OH
    c = (offsets // (OW * OH)) % C
    n = offsets // (OW * OH * C)

    n_mask = n < N
    c_mask = c < C
    oh_mask = oh < OH
    ow_mask = ow < OW
    valid = mask & n_mask & c_mask & oh_mask & ow_mask

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            in_bounds = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & valid
            idx = n * (C * H * W) + c * (H * W) + ih * W + iw
            val = tl.load(x_ptr + idx, mask=in_bounds, other=float('-inf'))
            max_val = tl.where(in_bounds, tl.maximum(max_val, val), max_val)

    tl.store(out_ptr + offsets, max_val, mask=valid)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr,
    out_ptr,
    N, C, H, W,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    """Adaptive avg pool to 1x1"""
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    c = offsets % C
    n = offsets // C

    n_mask = n < N
    c_mask = c < C
    valid = mask & n_mask & c_mask

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = H * W

    for h in range(H):
        for w in range(W):
            idx = n * (C * H * W) + c * (H * W) + h * W + w
            val = tl.load(x_ptr + idx, mask=valid, other=0.0)
            acc = acc + val

    avg = acc / count
    tl.store(out_ptr + offsets, avg, mask=valid)


def conv2d_bias_relu(x, weight, bias, stride=1, padding=0):
    """Conv2d followed by bias add and ReLU"""
    # Use PyTorch conv (cuDNN) then fuse bias+relu in Triton
    # Actually we need to implement conv in Triton per the rules
    # But conv is complex - let's use the implicit GEMM approach
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)
    
    OH = (H + 2 * padding[0] - KH) // stride[0] + 1
    OW = (W + 2 * padding[1] - KW) // stride[1] + 1
    
    out = torch.empty(N, C_out, OH, OW, device=x.device, dtype=x.dtype)
    
    # We'll implement this as a proper conv using Triton implicit GEMM
    # For now use the fused approach
    _conv2d_triton(x, weight, bias, out, N, C_in, H, W, C_out, OH, OW, KH, KW,
                   stride[0], stride[1], padding[0], padding[1])
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, C_in, H, W,
    C_out, OH, OW,
    KH, KW,
    SH, SW,
    PH, PW,
    M, K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    M = N * OH * OW (output spatial positions)
    N_out = C_out
    K = C_in * KH * KW
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index to (batch, out_h, out_w)
    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    batch = offs_m // (OH * OW)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K index to (in_c, kh, kw)
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        in_c = offs_k // (KH * KW)

        # Input coordinates
        ih = oh[:, None] * SH + kh[None, :] - PH
        iw = ow[:, None] * SW + kw[None, :] - PW

        valid_m = offs_m < M
        valid_k = offs_k < K
        in_bounds = (
            valid_m[:, None] &
            valid_k[None, :] &
            (ih >= 0) & (ih < H) &
            (iw >= 0) & (iw < W) &
            (batch[:, None] < N)
        )

        x_idx = batch[:, None] * (C_in * H * W) + in_c[None, :] * (H * W) + ih * W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=in_bounds, other=0.0)

        # Weight: [C_out, C_in, KH, KW] -> index [offs_n, in_c, kh, kw]
        w_idx = offs_n[None, :] * (C_in * KH * KW) + in_c[:, None] * (KH * KW) + kh[:, None] * KW + kw[:, None]
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < C_out)
        w_tile = tl.load(W_ptr + w_idx, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias and apply ReLU
    bias = tl.load(B_ptr + offs_n, mask=offs_n < C_out, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.where(acc > 0.0, acc, 0.0)

    # Store output: Y[batch, c_out, oh, ow]
    out_idx = batch[:, None] * (C_out * OH * OW) + offs_n[None, :] * (OH * OW) + oh[:, None] * OW + ow[:, None]
    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < C_out)
    tl.store(Y_ptr + out_idx, acc, mask=valid_out)


def _conv2d_triton(x, weight, bias, out, N, C_in, H, W, C_out, OH, OW, KH, KW, SH, SW, PH, PW):
    M = N * OH * OW
    K = C_in * KH * KW

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(C_out, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_relu_kernel[grid](
        x, weight, bias, out,
        N, C_in, H, W,
        C_out, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        M, K,
    )


def do_conv2d_relu(x, weight, bias, stride=1, padding=0):
    N, C_in, H, W = x.shape
    C_out, _, KH, KW = weight.shape
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(padding, int):
        padding = (padding, padding)

    OH = (H + 2 * padding[0] - KH) // stride[0] + 1
    OW = (W + 2 * padding[1] - KW) // stride[1] + 1

    out = torch.empty(N, C_out, OH, OW, device=x.device, dtype=x.dtype)
    _conv2d_triton(x, weight, bias, out, N, C_in, H, W, C_out, OH, OW, KH, KW,
                   stride[0], stride[1], padding[0], padding[1])
    return out


def do_maxpool2d(x, kernel_size=3, stride=2, ceil_mode=True):
    N, C, H, W = x.shape
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
        stride = (stride, stride)

    KH, KW = kernel_size
    SH, SW = stride
    PH, PW = 0, 0

    if ceil_mode:
        OH = (H + 2 * PH - KH + SH - 1) // SH + 1
        OW = (W + 2 * PW - KW + SW - 1) // SW + 1
    else:
        OH = (H + 2 * PH - KH) // SH + 1
        OW = (W + 2 * PW - KW) // SW + 1

    out = torch.empty(N, C, OH, OW, device=x.device, dtype=x.dtype)
    numel = N * C * OH * OW

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, out,
        N, C, H, W,
        OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        numel,
    )
    return out


def do_adaptive_avg_pool2d_1x1(x):
    N, C, H, W = x.shape
    out = torch.empty(N, C, 1, 1, device=x.device, dtype=x.dtype)
    numel = N * C

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        numel,
    )
    return out


def fire_module(x, sq_w, sq_b, ex1_w, ex1_b, ex3_w, ex3_b):
    """Fire module: squeeze -> expand1x1 + expand3x3 -> cat"""
    sq = do_conv2d_relu(x, sq_w, sq_b, stride=1, padding=0)
    ex1 = do_conv2d_relu(sq, ex1_w, ex1_b, stride=1, padding=0)
    ex3 = do_conv2d_relu(sq, ex3_w, ex3_b, stride=1, padding=1)
    return torch.cat([ex1, ex3], dim=1)


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
    # features[0]: Conv2d(3, 96, 7, stride=2) + ReLU
    x = do_conv2d_relu(x, p_features_0_weight, p_features_0_bias, stride=2, padding=0)

    # features[2]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = do_maxpool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    # features[3]: FireModule(96, 16, 64, 64)
    x = fire_module(
        x,
        p_features_3_squeeze_weight, p_features_3_squeeze_bias,
        p_features_3_expand1x1_weight, p_features_3_expand1x1_bias,
        p_features_3_expand3x3_weight, p_features_3_expand3x3_bias,
    )

    # features[4]: FireModule(128, 16, 64, 64)
    x = fire_module(
        x,
        p_features_4_squeeze_weight, p_features_4_squeeze_bias,
        p_features_4_expand1x1_weight, p_features_4_expand1x1_bias,
        p_features_4_expand3x3_weight, p_features_4_expand3x3_bias,
    )

    # features[5]: FireModule(128, 32, 128, 128)
    x = fire_module(
        x,
        p_features_5_squeeze_weight, p_features_5_squeeze_bias,
        p_features_5_expand1x1_weight, p_features_5_expand1x1_bias,
        p_features_5_expand3x3_weight, p_features_5_expand3x3_bias,
    )

    # features[6]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = do_maxpool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    # features[7]: FireModule(256, 32, 128, 128)
    x = fire_module(
        x,
        p_features_7_squeeze_weight, p_features_7_squeeze_bias,
        p_features_7_expand1x1_weight, p_features_7_expand1x1_bias,
        p_features_7_expand3x3_weight, p_features_7_expand3x3_bias,
    )

    # features[8]: FireModule(256, 48, 192, 192)
    x = fire_module(
        x,
        p_features_8_squeeze_weight, p_features_8_squeeze_bias,
        p_features_8_expand1x1_weight, p_features_8_expand1x1_bias,
        p_features_8_expand3x3_weight, p_features_8_expand3x3_bias,
    )

    # features[9]: FireModule(384, 48, 192, 192)
    x = fire_module(
        x,
        p_features_9_squeeze_weight, p_features_9_squeeze_bias,
        p_features_9_expand1x1_weight, p_features_9_expand1x1_bias,
        p_features_9_expand3x3_weight, p_features_9_expand3x3_bias,
    )

    # features[10]: FireModule(384, 64, 256, 256)
    x = fire_module(
        x,
        p_features_10_squeeze_weight, p_features_10_squeeze_bias,
        p_features_10_expand1x1_weight, p_features_10_expand1x1_bias,
        p_features_10_expand3x3_weight, p_features_10_expand3x3_bias,
    )

    # features[11]: MaxPool2d(3, stride=2, ceil_mode=True)
    x = do_maxpool2d(x, kernel_size=3, stride=2, ceil_mode=True)

    # features[12]: FireModule(512, 64, 256, 256)
    x = fire_module(
        x,
        p_features_12_squeeze_weight, p_features_12_squeeze_bias,
        p_features_12_expand1x1_weight, p_features_12_expand1x1_bias,
        p_features_12_expand3x3_weight, p_features_12_expand3x3_bias,
    )

    # classifier[0]: Dropout (p=0.0, no-op in inference)
    # classifier[1]: Conv2d(512, 1000, 1) + ReLU
    x = do_conv2d_relu(x, p_classifier_1_weight, p_classifier_1_bias, stride=1, padding=0)

    # classifier[3]: AdaptiveAvgPool2d(1, 1)
    x = do_adaptive_avg_pool2d_1x1(x)

    # flatten
    x = x.reshape(x.shape[0], -1)

    return x


def get_inputs():
    batch_size = 64
    return [
        torch.randn(96, 3, 7, 7, dtype=torch.float32, device='cuda'),      # p_features_0_weight
        torch.randn(96, dtype=torch.float32, device='cuda'),                 # p_features_0_bias
        torch.randn(16, 96, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_3_squeeze_weight
        torch.randn(16, dtype=torch.float32, device='cuda'),                 # p_features_3_squeeze_bias
        torch.randn(64, 16, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_3_expand1x1_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_3_expand1x1_bias
        torch.randn(64, 16, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_3_expand3x3_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_3_expand3x3_bias
        torch.randn(16, 128, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_4_squeeze_weight
        torch.randn(16, dtype=torch.float32, device='cuda'),                 # p_features_4_squeeze_bias
        torch.randn(64, 16, 1, 1, dtype=torch.float32, device='cuda'),      # p_features_4_expand1x1_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_4_expand1x1_bias
        torch.randn(64, 16, 3, 3, dtype=torch.float32, device='cuda'),      # p_features_4_expand3x3_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_4_expand3x3_bias
        torch.randn(32, 128, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_5_squeeze_weight
        torch.randn(32, dtype=torch.float32, device='cuda'),                 # p_features_5_squeeze_bias
        torch.randn(128, 32, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_5_expand1x1_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                # p_features_5_expand1x1_bias
        torch.randn(128, 32, 3, 3, dtype=torch.float32, device='cuda'),     # p_features_5_expand3x3_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                # p_features_5_expand3x3_bias
        torch.randn(32, 256, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_7_squeeze_weight
        torch.randn(32, dtype=torch.float32, device='cuda'),                 # p_features_7_squeeze_bias
        torch.randn(128, 32, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_7_expand1x1_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                # p_features_7_expand1x1_bias
        torch.randn(128, 32, 3, 3, dtype=torch.float32, device='cuda'),     # p_features_7_expand3x3_weight
        torch.randn(128, dtype=torch.float32, device='cuda'),                # p_features_7_expand3x3_bias
        torch.randn(48, 256, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_8_squeeze_weight
        torch.randn(48, dtype=torch.float32, device='cuda'),                 # p_features_8_squeeze_bias
        torch.randn(192, 48, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_8_expand1x1_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                # p_features_8_expand1x1_bias
        torch.randn(192, 48, 3, 3, dtype=torch.float32, device='cuda'),     # p_features_8_expand3x3_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                # p_features_8_expand3x3_bias
        torch.randn(48, 384, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_9_squeeze_weight
        torch.randn(48, dtype=torch.float32, device='cuda'),                 # p_features_9_squeeze_bias
        torch.randn(192, 48, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_9_expand1x1_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                # p_features_9_expand1x1_bias
        torch.randn(192, 48, 3, 3, dtype=torch.float32, device='cuda'),     # p_features_9_expand3x3_weight
        torch.randn(192, dtype=torch.float32, device='cuda'),                # p_features_9_expand3x3_bias
        torch.randn(64, 384, 1, 1, dtype=torch.float32, device='cuda'),     # p_features_10_squeeze_weight
        torch.randn(64, dtype=torch.float32, device='cuda'),                 # p_features_10_squeeze_bias
        torch.randn(256, 64, 1,