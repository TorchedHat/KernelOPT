import torch
import triton
import triton.language as tl
import math

# ============================================================
# Fused Conv2d + BatchNorm + ReLU kernel (implicit GEMM)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv_bn_relu_kernel(
    X_ptr, W_ptr, bias_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_idx in range(BLOCK_K_COUNT):
        k_off = k_idx * BLOCK_K
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw

        # X in NCHW: X[batch, in_c, ih, iw]
        x_ptrs = X_ptr + batch[:, None] * (IN_C * IN_H * IN_W) + in_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # W in NCHW: W[out_c, in_c, kh, kw]
        valid_n = offs_n[None, :] < OUT_C
        valid_k2 = offs_k[:, None] < TOTAL_K
        mask_w = valid_k2 & valid_n
        w_ptrs = W_ptr + offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) + in_c[:, None] * (KERNEL_H * KERNEL_W) + kh[:, None] * KERNEL_W + kw[:, None]
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add conv bias
    valid_n = offs_n < OUT_C
    bias = tl.load(bias_ptr + offs_n, mask=valid_n, other=0.0)
    acc = acc + bias[None, :]

    # BatchNorm: (x - mean) / sqrt(var + eps) * weight + bias
    bn_mean = tl.load(bn_mean_ptr + offs_n, mask=valid_n, other=0.0)
    bn_var = tl.load(bn_var_ptr + offs_n, mask=valid_n, other=1.0)
    bn_w = tl.load(bn_weight_ptr + offs_n, mask=valid_n, other=1.0)
    bn_b = tl.load(bn_bias_ptr + offs_n, mask=valid_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var + eps)
    acc = (acc - bn_mean[None, :]) * inv_std[None, :] * bn_w[None, :] + bn_b[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # Store: output in NCHW format
    # Y[batch, out_c, out_h, out_w] -> linear index: batch*(OUT_C*OUT_H*OUT_W) + out_c*(OUT_H*OUT_W) + out_h*OUT_W + out_w
    valid_m = offs_m[:, None] < M_TOTAL
    mask_out = valid_m & valid_n[None, :]

    y_ptrs = Y_ptr + batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def conv_bn_relu(x, weight, bias, bn_mean, bn_var, bn_weight, bn_bias,
                 stride=1, padding=1, eps=1e-5):
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = weight.shape
    OUT_H = (IN_H + 2 * padding - KERNEL_H) // stride + 1
    OUT_W = (IN_W + 2 * padding - KERNEL_W) // stride + 1

    y = torch.empty(BATCH, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W

    def grid(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )

    conv_bn_relu_kernel[grid](
        x, weight, bias,
        bn_mean, bn_var, bn_weight, bn_bias,
        y,
        BATCH=BATCH, IN_C=IN_C, IN_H=IN_H, IN_W=IN_W,
        OUT_C=OUT_C, OUT_H=OUT_H, OUT_W=OUT_W,
        KERNEL_H=KERNEL_H, KERNEL_W=KERNEL_W,
        STRIDE_H=stride, STRIDE_W=stride,
        PAD_H=padding, PAD_W=padding,
        eps=eps,
    )
    return y


# ============================================================
# Max Pool 2D kernel (2x2, stride 2)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH: tl.constexpr, C: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    N_TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: offs -> (batch, c, out_h, out_w)
    out_w = offs % OUT_W
    out_h = (offs // OUT_W) % OUT_H
    c = (offs // (OUT_W * OUT_H)) % C
    batch = offs // (OUT_W * OUT_H * C)

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h * STRIDE_H + kh
            iw = out_w * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_idx = batch * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(X_ptr + x_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(Y_ptr + offs, max_val, mask=mask)


def maxpool2d(x, kernel_size=2, stride=2):
    BATCH, C, IN_H, IN_W = x.shape
    OUT_H = (IN_H - kernel_size) // stride + 1
    OUT_W = (IN_W - kernel_size) // stride + 1
    y = torch.empty(BATCH, C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    N_TOTAL = BATCH * C * OUT_H * OUT_W

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, y,
        BATCH=BATCH, C=C,
        IN_H=IN_H, IN_W=IN_W,
        OUT_H=OUT_H, OUT_W=OUT_W,
        KH=kernel_size, KW=kernel_size,
        STRIDE_H=stride, STRIDE_W=stride,
        N_TOTAL=N_TOTAL,
    )
    return y


# ============================================================
# Global Average Pooling + Linear kernel (fused)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 256, 'BLOCK_HW': 256}, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 512}, num_warps=4),
        triton.Config({'BLOCK_C': 256, 'BLOCK_HW': 512}, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 256}, num_warps=4),
    ],
    key=['BATCH', 'C', 'HW'],
)
@triton.jit
def gap_kernel(
    X_ptr, OUT_ptr,
    BATCH: tl.constexpr, C: tl.constexpr, HW: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    # One program per (batch, channel_block)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    valid_c = offs_c < C

    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)

    for hw_off in range(0, HW, BLOCK_HW):
        offs_hw = hw_off + tl.arange(0, BLOCK_HW)
        valid_hw = offs_hw < HW
        # X[pid_b, offs_c, offs_hw] -> X[pid_b * C * HW + offs_c * HW + offs_hw]
        x_ptrs = X_ptr + pid_b * C * HW + offs_c[:, None] * HW + offs_hw[None, :]
        mask = valid_c[:, None] & valid_hw[None, :]
        vals = tl.load(x_ptrs, mask=mask, other=0.0)
        acc += tl.sum(vals, axis=1)

    acc = acc / HW

    out_ptrs = OUT_ptr + pid_b * C + offs_c
    tl.store(out_ptrs, acc, mask=valid_c)


def global_avg_pool(x):
    BATCH, C, H, W = x.shape
    HW = H * W
    out = torch.empty(BATCH, C, device=x.device, dtype=x.dtype)

    BLOCK_C = 256
    BLOCK_HW = 256

    grid = (BATCH, triton.cdiv(C, BLOCK_C))
    gap_kernel[grid](
        x, out,
        BATCH=BATCH, C=C, HW=HW,
        BLOCK_C=BLOCK_C, BLOCK_HW=BLOCK_HW,
    )
    return out


# ============================================================
# Linear kernel (GEMM: [BATCH, IN] x [OUT, IN]^T + bias)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 32, 'BLOCK_K': 128}, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)

        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_n[:, None] * K + offs_k[None, :], mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


def linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    y = torch.empty(M, N, device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, y,
        M=M, N=N, K=K,
    )
    return y


# ============================================================
# Main kernel_function
# ============================================================

def kernel_function(
    p_feature_extractor_0_0_weight,
    p_feature_extractor_0_0_bias,
    p_feature_extractor_0_1_weight,
    p_feature_extractor_0_1_bias,
    p_feature_extractor_0_3_weight,
    p_feature_extractor_0_3_bias,
    p_feature_extractor_0_4_weight,
    p_feature_extractor_0_4_bias,
    p_feature_extractor_1_0_weight,
    p_feature_extractor_1_0_bias,
    p_feature_extractor_1_1_weight,
    p_feature_extractor_1_1_bias,
    p_feature_extractor_1_3_weight,
    p_feature_extractor_1_3_bias,
    p_feature_extractor_1_4_weight,
    p_feature_extractor_1_4_bias,
    p_feature_extractor_2_0_weight,
    p_feature_extractor_2_0_bias,
    p_feature_extractor_2_1_weight,
    p_feature_extractor_2_1_bias,
    p_feature_extractor_2_3_weight,
    p_feature_extractor_2_3_bias,
    p_feature_extractor_2_4_weight,
    p_feature_extractor_2_4_bias,
    p_fc_weight,
    p_fc_bias,
    b_feature_extractor_0_1_running_mean,
    b_feature_extractor_0_1_running_var,
    b_feature_extractor_0_4_running_mean,
    b_feature_extractor_0_4_running_var,
    b_feature_extractor_1_1_running_mean,
    b_feature_extractor_1_1_running_var,
    b_feature_extractor_1_4_running_mean,
    b_feature_extractor_1_4_running_var,
    b_feature_extractor_2_1_running_mean,
    b_feature_extractor_2_1_running_var,
    b_feature_extractor_2_4_running_mean,
    b_feature_extractor_2_4_running_var,
    x,
):
    # Stage 0, block 0: Conv(3->64, 3x3, pad=1) + BN + ReLU
    x = conv_bn_relu(
        x,
        p_feature_extractor_0_0_weight,
        p_feature_extractor_0_0_bias,
        b_feature_extractor_0_1_running_mean,
        b_feature_extractor_0_1_running_var,
        p_feature_extractor_0_1_weight,
        p_feature_extractor_0_1_bias,
        stride=1, padding=1,
    )

    # Stage 0, block 1: Conv(64->64, 3x3, pad=1) + BN + ReLU
    x = conv_bn_relu(
        x,
        p_feature_extractor_0_3_weight,
        p_feature_extractor_0_3_bias,
        b_feature_extractor_0_4_running_mean,
        b_feature_extractor_0_4_running_var,
        p_feature_extractor_0_4_weight,
        p_feature_extractor_0_4_bias,
        stride=1, padding=1,
    )

    # MaxPool 2x2 stride 2
    x = maxpool2d(x, kernel_size=2, stride=2)

    # Stage 1, block 0: Conv(64->128, 3x3, pad=1) + BN + ReLU
    x = conv_bn_relu(
        x,
        p_feature_extractor_1_0_weight,
        p_feature_extractor_1_0_bias,
        b_feature_extractor_1_1_running_mean,
        b_feature_extractor_1_1_running_var,
        p_feature_extractor_1_1_weight,
        p_feature_extractor_1_1_bias,
        stride=1, padding=1,
    )

    # Stage 1, block 1: Conv(128->128, 3x3, pad=1) + BN + ReLU
    x = conv_bn_relu(
        x,
        p_feature_extractor_1_3_weight,
        p_feature_extractor_1_3_bias,
        b_feature_extractor_1_4_running_mean,
        b_feature_extractor_1_4_running_var,
        p_feature_extractor_1_4_weight,
        p_feature_extractor_1_4_bias,
        stride=1, padding=1,
    )

    # MaxPool 2x2 stride 2
    x = maxpool2d(x, kernel_size=2, stride=2)

    # Stage 2, block 0: Conv(128->256, 3x3, pad=1) + BN + ReLU
    x = conv_bn_relu(
        x,
        p_feature_extractor_2_0_weight,
        p_feature_extractor_2_0_bias,
        b_feature_extractor_2_1_running_mean,
        b_feature_extractor_2_1_running_var,
        p_feature_extractor_2_1_weight,
        p_feature_extractor_2_1_bias,
        stride=1, padding=1,
    )

    # Stage 2, block 1: Conv(256->256, 3x3, pad=1) + BN + ReLU
    x = conv_bn_relu(
        x,
        p_feature_extractor_2_3_weight,
        p_feature_extractor_2_3_bias,
        b_feature_extractor_2_4_running_mean,
        b_feature_extractor_2_4_running_var,
        p_feature_extractor_2_4_weight,
        p_feature_extractor_2_4_bias,
        stride=1, padding=1,
    )

    # MaxPool 2x2 stride 2
    x = maxpool2d(x, kernel_size=2, stride=2)

    # Global Average Pooling: mean over H, W -> [BATCH, 256]
    x = global_avg_pool(x)

    # Linear: [BATCH, 256] -> [BATCH, 10]
    x = linear(x, p_fc_weight, p_fc_bias)

    return x


def get_inputs():
    batch_size = 8
    input_channels = 3
    image_height, image_width = 224, 224
    stages = 3
    block_widths = [64, 128, 256]
    output_classes = 10

    p_feature_extractor_0_0_weight = torch.randn(64, 3, 3, 3)
    p_feature_extractor_0_0_bias = torch.randn(64)
    p_feature_extractor_0_1_weight = torch.randn(64)
    p_feature_extractor_0_1_bias = torch.randn(64)
    p_feature_extractor_0_3_weight = torch.randn(64, 64, 3, 3)
    p_feature_extractor_0_3_bias = torch.randn(64)
    p_feature_extractor_0_4_weight = torch.randn(64)
    p_feature_extractor_0_4_bias = torch.randn(64)
    p_feature_extractor_1_0_weight = torch.randn(128, 64, 3, 3)
    p_feature_extractor_1_0_bias = torch.randn(128)
    p_feature_extractor_1_1_weight = torch.randn(128)
    p_feature_extractor_1_1_bias = torch.randn(128)
    p_feature_extractor_1_3_weight = torch.randn(128, 128, 3, 3)
    p_feature_extractor_1_3_bias = torch.randn(128)
    p_feature_extractor_1_4_weight = torch.randn(128)
    p_feature_extractor_1_4_bias = torch.randn(128)
    p_feature_extractor_2_0_weight = torch.randn(256, 128, 3, 3)
    p_feature_extractor_2_0_bias = torch.randn(256)
    p_feature_extractor_2_1_weight = torch.randn(256)
    p_feature_extractor_2_1_bias = torch.randn(256)
    p_feature_extractor_2_3_weight = torch.randn(256, 256, 3, 3)
    p_feature_extractor_2_3_bias = torch.randn(256)
    p_feature_extractor_2_4_weight = torch.randn(256)
    p_feature_extractor_2_4_bias = torch.randn(256)
    p_fc_weight = torch.randn(10, 256)
    p_fc_bias = torch.randn(10)
    b_feature_extractor_0_1_running_mean = torch.zeros(64)
    b_feature_extractor_0_1_running_var = torch.ones(64)
    b_feature_extractor_0_4_running_mean = torch.zeros(64)
    b_feature_extractor_0_4_running_var = torch.ones(64)
    b_feature_extractor_1_1_running_mean = torch.zeros(128)
    b_feature_extractor_1_1_running_var = torch.ones(128)
    b_feature_extractor_1_4_running_mean = torch.zeros(128)
    b_feature_extractor_1_4_running_var = torch.ones(128)
    b_feature_extractor_2_1_running_mean = torch.zeros(256)
    b_feature_extractor_2_1_running_var = torch.ones(256)
    b_feature_extractor_2_4_running_mean = torch.zeros(256)
    b_feature_extractor_2_4_running_var = torch.ones(256)
    x = torch.rand(batch_size, input_channels, image_height, image_width)

    device = 'cuda'
    return [
        p_feature_extractor_0_0_weight.to(device),
        p_feature_extractor_0_0_bias.to(device),
        p_feature_extractor_0_1_weight.to(device),
        p_feature_extractor_0_1_bias.to(device),
        p_feature_extractor_0_3_weight.to(device),
        p_feature_extractor_0_3_bias.to(device),
        p_feature_extractor_0_4_weight.to(device),
        p_feature_extractor_0_4_bias.to(device),
        p_feature_extractor_1_0_weight.to(device),
        p_feature_extractor_1_0_bias.to(device),