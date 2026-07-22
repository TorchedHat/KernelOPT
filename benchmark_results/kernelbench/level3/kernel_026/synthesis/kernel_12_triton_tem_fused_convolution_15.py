import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Fused BatchNorm (inference) + optional ReLU  -- pointwise kernel
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    C, HW,
    apply_relu: tl.constexpr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    val = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    # channel index
    c = (offs // HW) % C
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c, mask=mask, other=1.0).to(tl.float32)
    gam  = tl.load(gamma_ptr + c, mask=mask, other=1.0).to(tl.float32)
    bet  = tl.load(beta_ptr  + c, mask=mask, other=0.0).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var + 1e-5)
    out = (val - mean) * inv_std * gam + bet
    if apply_relu:
        out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


def bn_relu(x, mean, var, gamma, beta, relu=True):
    B, C, H, W = x.shape
    N = B * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    bn_relu_kernel[grid](
        x, out, mean, var, gamma, beta,
        C, HW, relu, N,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Channel Shuffle  -- pointwise kernel
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    B, C, HW, groups,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # decode (b, c, hw)
    hw   = offs % HW
    tmp  = offs // HW
    c    = tmp % C
    b    = tmp // C
    # channel shuffle: c -> (c % cpg) * groups + (c // cpg)
    cpg  = C // groups
    g    = c // cpg
    r    = c % cpg
    c_new = r * groups + g
    src = b * (C * HW) + c_new * HW + hw
    val = tl.load(x_ptr + src, mask=mask, other=0.0)
    tl.store(out_ptr + offs, val, mask=mask)


def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    N = B * C * H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    channel_shuffle_kernel[grid](x, out, B, C, H * W, groups, N)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Max Pool 2D (3x3, stride=2, pad=1)  -- pointwise output kernel
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 2048}, num_warps=8),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def maxpool2d_kernel(
    x_ptr, out_ptr,
    B, C, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    N,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    ow   = offs % OW
    tmp  = offs // OW
    oh   = tmp % OH
    tmp2 = tmp // OH
    c    = tmp2 % C
    b    = tmp2 // C
    best = tl.full([BLOCK], float('-inf'), dtype=tl.float32)
    for kh in range(KH):
        for kw in range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            src = b * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(x_ptr + src, mask=valid, other=float('-inf'))
            best = tl.where(val > best, val, best)
    tl.store(out_ptr + offs, best, mask=mask)


def maxpool2d(x, kernel_size=3, stride=2, padding=1):
    B, C, IH, IW = x.shape
    OH = (IH + 2 * padding - kernel_size) // stride + 1
    OW = (IW + 2 * padding - kernel_size) // stride + 1
    N = B * C * OH * OW
    out = torch.empty(B, C, OH, OW, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    maxpool2d_kernel[grid](
        x, out, B, C, IH, IW, OH, OW,
        kernel_size, kernel_size, stride, stride, padding, padding, N,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Adaptive Average Pool 2D -> (1,1)  -- one program per (b,c)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=8),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=4),
        triton.Config({'BLOCK': 128},  num_warps=2),
    ],
    key=['HW'],
)
@triton.jit
def adaptive_avgpool_kernel(
    x_ptr, out_ptr,
    BC, HW,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)   # one program per (b,c) pair
    base = pid * HW
    acc = tl.zeros([1], dtype=tl.float32)
    for off in range(0, HW, BLOCK):
        offs = off + tl.arange(0, BLOCK)
        mask = offs < HW
        val = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(val, axis=0)
    tl.store(out_ptr + pid, (acc / HW).to(tl.float32))


def adaptive_avgpool(x):
    B, C, H, W = x.shape
    HW = H * W
    out = torch.empty(B, C, 1, 1, device=x.device, dtype=x.dtype)
    grid = (B * C,)
    adaptive_avgpool_kernel[grid](x, out.view(-1), B * C, HW)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Linear (GEMM): y = x @ W^T + b   shape: (M, K) x (N, K)^T -> (M, N)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2),
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
        x_tile = tl.load(
            x_ptr + offs_m[:, None] * K + offs_k[None, :],
            mask=(offs_m[:, None] < M) & (offs_k[None, :] < K),
            other=0.0,
        )
        w_tile = tl.load(
            w_ptr + offs_n[None, :] * K + offs_k[:, None],
            mask=(offs_n[None, :] < N) & (offs_k[:, None] < K),
            other=0.0,
        )
        acc = tl.dot(x_tile, w_tile, acc)
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + bias[None, :]
    tl.store(
        out_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def linear(x, weight, bias):
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](x, weight, bias, out, M, N, K)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Implicit GEMM Conv2d kernel (groups supported)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M_total', 'OUT_C', 'K_total'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    GROUPS: tl.constexpr,
    M_total, K_total,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    GROUP_IN_C  = IN_C  // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    ow  = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh  = tmp % OUT_H
    b   = tmp // OUT_H

    # which group does each output channel belong to?
    g_n = offs_n // GROUP_OUT_C   # group index for output channels

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for k in range(0, K_total, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # decode k: (kh, kw, in_c_local)
        in_c_local = offs_k % GROUP_IN_C
        kw_idx     = (offs_k // GROUP_IN_C) % KW
        kh_idx     = offs_k // (GROUP_IN_C * KW)

        ih = oh[:, None] * SH + kh_idx[None, :] - PH
        iw = ow[:, None] * SW + kw_idx[None, :] - PW

        # group for each m: determined by output channel group
        # For input: group of m is determined by which group the output belongs to
        # But m doesn't have a fixed group -- we need to handle groups carefully.
        # For grouped conv: output channel offs_n[j] belongs to group g = offs_n[j]//GROUP_OUT_C
        # The corresponding input channels are [g*GROUP_IN_C, (g+1)*GROUP_IN_C)
        # We tile over K = GROUP_IN_C * KH * KW for each group separately.
        # Since GROUPS can vary, we handle it by masking.
        # For each (m, n) pair, the valid k range is determined by group(n).
        # We load X with group offset based on g_n (group of output channel).
        # This is approximate for mixed groups in a tile -- use masking.

        g_n_bc = g_n[None, :]   # [1, BLOCK_N]
        in_c_abs = g_n_bc * GROUP_IN_C + in_c_local[:, None]  # [BLOCK_K, BLOCK_N] -- but we need [BLOCK_M, BLOCK_K]

        # For X: group is determined per output channel, but X tile is [BLOCK_M, BLOCK_K]
        # We need a single group per tile column. Use g_n for the n-dimension.
        # For simplicity, assume all offs_n in a tile belong to the same group (true when BLOCK_N <= GROUP_OUT_C)
        # Use the group of the first element in offs_n tile
        g0 = offs_n[0] // GROUP_OUT_C
        in_c_abs_x = g0 * GROUP_IN_C + in_c_local  # [BLOCK_K]

        valid_k_x = (offs_k < K_total) & (in_c_abs_x < IN_C)
        valid_hw  = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        valid_m   = offs_m < M_total

        x_ptrs = (
            b[:, None] * (IN_C * IN_H * IN_W)
        )
        mask_x = valid_m[:, None] & valid_k_x[None, :] & valid_hw
        x_tile = tl.load(X_ptr + x_ptrs, mask=mask_x, other=0.0)

        # W shape: [OUT_C, GROUP_IN_C, KH, KW]
        # w_ptrs: [BLOCK_K, BLOCK_N]
        w_ptrs = (
            offs_n[None, :] * (GROUP_IN_C * KH * KW)
        )
        mask_w = (offs_k[:, None] < K_total) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(W_ptr + w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    out_ptrs = (
        b[:, None] * (OUT_C * OUT_H * OUT_W)
    )
    mask_out = (offs_m[:, None] < M_total) & (offs_n[None, :] < OUT_C)
    tl.store(Out_ptr + out_ptrs, acc, mask=mask_out)


def conv2d(x, weight, stride=1, padding=0, groups=1):
    B, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    if isinstance(stride, int):
        SH = SW = stride
    else:
        SH, SW = stride
    if isinstance(padding, int):
        PH = PW = padding
    else:
        PH, PW = padding
    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1
    M_total = B * OUT_H * OUT_W
    GROUP_IN_C = IN_C // groups
    K_total = GROUP_IN_C * KH * KW
    out = torch.empty(B, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(M_total, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    conv2d_kernel[grid](
        x, weight, out,
        B, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        groups,
        M_total, K_total,
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ShuffleNet building blocks
# ─────────────────────────────────────────────────────────────────────────────
def shufflenet_unit(x, params, prefix, groups=3, has_shortcut_conv=False):
    """
    """
    shortcut = x

    # conv1 (group conv 1x1)
    w1 = params[f'p_{prefix}_conv1_weight']
    x = conv2d(x, w1, stride=1, padding=0, groups=groups)
    x = bn_relu(x,
                params[f'b_{prefix}_bn1_running_mean'],
                params[f'b_{prefix}_bn1_running_var'],
                params[f'p_{prefix}_bn1_weight'],
                params[f'p_{prefix}_bn1_bias'],
                relu=True)

    # conv2 (depthwise 3x3)
    w2 = params[f'p_{prefix}_conv2_weight']
    mid_c = w2.shape[0]
    x = conv2d(x, w2, stride=1, padding=1, groups=mid_c)
    x = bn_relu(x,
                params[f'b_{prefix}_bn2_running_mean'],
                params[f'b_{prefix}_bn2_running_var'],
                params[f'p_{prefix}_bn2_weight'],
                params[f'p_{prefix}_bn2_bias'],
                relu=False)

    # channel shuffle
    x = channel_shuffle(x, groups)

    # conv3 (group conv 1x1)
    w3 = params[f'p_{prefix}_conv3_weight']
    x = conv2d(x, w3, stride=1, padding=0, groups=groups)
    x = bn_relu(x,
                params[f'b_{prefix}_bn3_running_mean'],
                params[f'b_{prefix}_bn3_running_var'],
                params[f'p_{prefix}_bn3_weight'],
                params[f'p_{prefix}_bn3_bias'],
                relu=True)

    # shortcut
    if has_shortcut_conv:
        ws = params[f'p_{prefix}_shortcut_0_weight']
        shortcut = conv2d(shortcut, ws, stride=1, padding=0, groups=1)
        shortcut = bn_relu(shortcut,
                           params[f'b_{prefix}_shortcut_1_running_mean'],
                           params[f'b_{prefix}_shortcut_1_running_var'],
                           params[f'p_{prefix}_shortcut_1_weight'],
                           params[f'p_{prefix}_shortcut_1_bias'],
                           relu=False)

    x = x + shortcut
    return x


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_stage2_0_conv1_weight, p_stage2_0_bn1_weight, p_stage2_0_bn1_bias,
    p_stage2_0_conv2_weight, p_stage2_0_bn2_weight, p_stage2_0_bn2_bias,
    p_stage2_0_conv3_weight, p_stage2_0_bn3_weight, p_stage2_0_bn3_bias,
    p_stage2_0_shortcut_0_weight, p_stage2_0_shortcut_1_weight, p_stage2_0_shortcut_1_bias,
    p_stage2_1_conv1_weight, p_stage2_1_bn1_weight, p_stage2_1_bn1_bias,
    p_stage2_1_conv2_weight, p_stage2_1_bn2_weight, p_stage2_1_bn2_bias,
    p_stage2_1_conv3_weight, p_stage2_1_bn3_weight, p_stage2_1_bn3_bias,
    p_stage2_2_conv1_weight, p_stage2_2_bn1_weight, p_stage2_2_bn1_bias,
    p_stage2_2_conv2_weight, p_stage2_2_bn2_weight, p_stage2_2_bn2_bias,
    p_stage2_2_conv3_weight, p_stage2_2_bn3_weight, p_stage2_2_bn3_bias,
    p_stage3_0_conv1_weight, p_stage3_0_bn1_weight, p_stage3_0_bn1_bias,
    p_stage3_0_conv2_weight, p_stage3_0_bn2_weight, p_stage3_0_bn2_bias,
    p_stage3_0_conv3_weight, p_stage3_0_bn3_weight, p_stage3_0_bn3_bias,
    p_stage3_0_shortcut_0_weight, p_stage3_0_shortcut_1_weight, p_stage3_0_shortcut_1_bias,
    p_stage3_1_conv1_weight, p_stage3_1_bn1_weight, p_stage3_1_bn1_bias,
    p_stage3_1_conv2_weight, p_stage3_1_bn2_weight, p_stage3_1_bn2_bias,
    p_stage3_1_conv3_weight, p_stage3_1_bn3_weight, p_stage3_1_bn3_bias,
    p_stage3_2_conv1_weight, p_stage3_2_bn1_weight, p_stage3_2_bn1_bias,
    p_stage3_2_conv2_weight, p_stage3_2_bn2_weight, p_stage3_2_bn2_bias,
    p_stage3_2_conv3_weight, p_stage3_2_bn3_weight, p_stage3_2_bn3_bias,
    p_stage3_3_conv1_weight, p_stage3_3_bn1_weight, p_stage3_3_bn1_bias,
    p_stage3_3_conv2_weight, p_stage3_3_bn2_weight, p_stage3_3_bn2_bias,
    p_stage3_3_conv3_weight, p_stage3_3_bn3_weight, p_stage3_3_bn3_bias,
    p_stage3_4_conv1_weight, p_stage3_4_bn1_weight, p_stage3_4_bn1_bias,
    p_stage3_4_conv2_weight, p_stage3_4_bn2_weight, p_stage3_4_bn2_bias,
    p_stage3_4_conv3_weight, p_stage3_4_bn3_weight, p_stage3_4_bn3_bias,
    p_stage3_5_conv1_weight, p_stage3_5_bn1_weight, p_stage3_5_bn1_bias,
    p_stage3_5_conv2_weight, p_stage3_5_bn2_weight, p_stage3_5_bn2_bias,
    p_stage3_5_conv3_weight, p_stage3_5_bn3_weight, p_stage3_5_bn3_bias,
    p_stage3_6_conv1_weight, p_stage3_6_bn1_weight, p_stage3_6_bn1_bias,
    p_stage3_6_conv2_weight, p_stage3_6_bn2_weight, p_stage3_6_bn2_bias,