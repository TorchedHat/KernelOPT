import torch
import triton
import triton.language as tl
import math

# ─── Fused BN + optional activation (relu / hardtanh / none) ───────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def bn_act_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    act: tl.constexpr,   # 0=none, 1=relu, 2=hardtanh
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    # channel index: layout is (N, C, H, W) contiguous -> channel = (offs // HW) % C
    c = (offs // HW) % C

    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c, mask=mask, other=0.0).to(tl.float32)
    w    = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * w + b

    if act == 1:  # relu
        y = tl.where(y > 0.0, y, 0.0)
    elif act == 2:  # hardtanh [-6, 6]
        y = tl.where(y < -6.0, -6.0, y)
        y = tl.where(y > 6.0,  6.0,  y)

    tl.store(out_ptr + offs, y.to(tl.float32), mask=mask)


def bn_act(x, mean, var, weight, bias, eps=1e-5, act=0):
    """Fused BatchNorm + activation. act: 0=none, 1=relu, 2=hardtanh"""
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_act_kernel[grid](
        x, out,
        mean, var, weight, bias,
        numel, C, HW,
        eps, act,
    )
    return out


# ─── Fused BN + add residual + optional activation ─────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def bn_add_kernel(
    x_ptr, res_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    c = (offs // HW) % C

    x   = tl.load(x_ptr   + offs, mask=mask, other=0.0).to(tl.float32)
    res = tl.load(res_ptr  + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr   + c, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr    + c, mask=mask, other=0.0).to(tl.float32)
    w    = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * w + b + res

    tl.store(out_ptr + offs, y.to(tl.float32), mask=mask)


def bn_add(x, res, mean, var, weight, bias, eps=1e-5):
    N, C, H, W = x.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_add_kernel[grid](
        x, res, out,
        mean, var, weight, bias,
        numel, C, HW,
        eps,
    )
    return out


# ─── Fused adaptive avg pool + flatten ─────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
        triton.Config({'BLOCK_SIZE': 32},  num_warps=2),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def avg_pool_flatten_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    # one program per (n, c) pair
    pid = tl.program_id(0)
    nc_total = N * C
    if pid >= nc_total:
        return
    n = pid // C
    c = pid % C

    base = (n * C + c) * HW
    offs = tl.arange(0, BLOCK_SIZE)

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    # We need to sum HW elements; loop in chunks of BLOCK_SIZE
    # But BLOCK_SIZE may be > HW, so mask carefully
    # For simplicity, do a single load with mask (HW is small: 7*7=49)
    # We'll use a loop approach
    total = tl.zeros([1], dtype=tl.float32)
    count = 0
    # Unrolled: load all HW elements
    # Since HW can be up to 49 for 7x7, BLOCK_SIZE=64 covers it
    mask = offs < HW
    vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
    s = tl.sum(vals, axis=0)
    result = s / HW

    out_idx = n * C + c
    tl.store(out_ptr + out_idx, result)


def avg_pool_flatten(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, device=x.device, dtype=x.dtype)
    # Use a grid of N*C programs
    grid = (N * C,)
    avg_pool_flatten_kernel[grid](
        x, out,
        N, C, HW,
    )
    return out


# ─── Triton GEMM for final linear layer ────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

        acc = tl.dot(x_tile, w_tile, acc)

    # add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def triton_linear(x, weight, bias):
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


# ─── Conv2d wrapper (uses cuDNN) ────────────────────────────────────────────

import torch.nn.functional as F

def conv2d(x, weight, bias=None, stride=1, padding=0, groups=1):
    return F.conv2d(x, weight, bias=bias, stride=stride, padding=padding, groups=groups)


# ─── Main kernel_function ───────────────────────────────────────────────────

def kernel_function(
    p_conv1_weight,
    p_bn1_weight, p_bn1_bias,
    p_blocks_0_depthwise_conv_0_weight,
    p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
    p_blocks_0_project_conv_0_weight,
    p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
    p_blocks_1_expand_conv_0_weight,
    p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
    p_blocks_1_depthwise_conv_0_weight,
    p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
    p_blocks_1_project_conv_0_weight,
    p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
    p_blocks_2_expand_conv_0_weight,
    p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
    p_blocks_2_depthwise_conv_0_weight,
    p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
    p_blocks_2_project_conv_0_weight,
    p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
    p_blocks_3_expand_conv_0_weight,
    p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
    p_blocks_3_depthwise_conv_0_weight,
    p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
    p_blocks_3_project_conv_0_weight,
    p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
    p_blocks_4_expand_conv_0_weight,
    p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
    p_blocks_4_depthwise_conv_0_weight,
    p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
    p_blocks_4_project_conv_0_weight,
    p_blocks_4_project_conv_1_weight, p_blocks_4_project_conv_1_bias,
    p_blocks_5_expand_conv_0_weight,
    p_blocks_5_expand_conv_1_weight, p_blocks_5_expand_conv_1_bias,
    p_blocks_5_depthwise_conv_0_weight,
    p_blocks_5_depthwise_conv_1_weight, p_blocks_5_depthwise_conv_1_bias,
    p_blocks_5_project_conv_0_weight,
    p_blocks_5_project_conv_1_weight, p_blocks_5_project_conv_1_bias,
    p_blocks_6_expand_conv_0_weight,
    p_blocks_6_expand_conv_1_weight, p_blocks_6_expand_conv_1_bias,
    p_blocks_6_depthwise_conv_0_weight,
    p_blocks_6_depthwise_conv_1_weight, p_blocks_6_depthwise_conv_1_bias,
    p_blocks_6_project_conv_0_weight,
    p_blocks_6_project_conv_1_weight, p_blocks_6_project_conv_1_bias,
    p_blocks_7_expand_conv_0_weight,
    p_blocks_7_expand_conv_1_weight, p_blocks_7_expand_conv_1_bias,
    p_blocks_7_depthwise_conv_0_weight,
    p_blocks_7_depthwise_conv_1_weight, p_blocks_7_depthwise_conv_1_bias,
    p_blocks_7_project_conv_0_weight,
    p_blocks_7_project_conv_1_weight, p_blocks_7_project_conv_1_bias,
    p_blocks_8_expand_conv_0_weight,
    p_blocks_8_expand_conv_1_weight, p_blocks_8_expand_conv_1_bias,
    p_blocks_8_depthwise_conv_0_weight,
    p_blocks_8_depthwise_conv_1_weight, p_blocks_8_depthwise_conv_1_bias,
    p_blocks_8_project_conv_0_weight,
    p_blocks_8_project_conv_1_weight, p_blocks_8_project_conv_1_bias,
    p_blocks_9_expand_conv_0_weight,
    p_blocks_9_expand_conv_1_weight, p_blocks_9_expand_conv_1_bias,
    p_blocks_9_depthwise_conv_0_weight,
    p_blocks_9_depthwise_conv_1_weight, p_blocks_9_depthwise_conv_1_bias,
    p_blocks_9_project_conv_0_weight,
    p_blocks_9_project_conv_1_weight, p_blocks_9_project_conv_1_bias,
    p_blocks_10_expand_conv_0_weight,
    p_blocks_10_expand_conv_1_weight, p_blocks_10_expand_conv_1_bias,
    p_blocks_10_depthwise_conv_0_weight,
    p_blocks_10_depthwise_conv_1_weight, p_blocks_10_depthwise_conv_1_bias,
    p_blocks_10_project_conv_0_weight,
    p_blocks_10_project_conv_1_weight, p_blocks_10_project_conv_1_bias,
    p_blocks_11_expand_conv_0_weight,
    p_blocks_11_expand_conv_1_weight, p_blocks_11_expand_conv_1_bias,
    p_blocks_11_depthwise_conv_0_weight,
    p_blocks_11_depthwise_conv_1_weight, p_blocks_11_depthwise_conv_1_bias,
    p_blocks_11_project_conv_0_weight,
    p_blocks_11_project_conv_1_weight, p_blocks_11_project_conv_1_bias,
    p_blocks_12_expand_conv_0_weight,
    p_blocks_12_expand_conv_1_weight, p_blocks_12_expand_conv_1_bias,
    p_blocks_12_depthwise_conv_0_weight,
    p_blocks_12_depthwise_conv_1_weight, p_blocks_12_depthwise_conv_1_bias,
    p_blocks_12_project_conv_0_weight,
    p_blocks_12_project_conv_1_weight, p_blocks_12_project_conv_1_bias,
    p_conv2_weight,
    p_bn2_weight, p_bn2_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var,
    b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var,
    b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var,
    b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var,
    b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var,
    b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var,
    b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var,
    b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var,
    b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var,
    b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var,
    b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var,
    b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var,
    b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var,
    b_blocks_4_project_conv_1_running_mean, b_blocks_4_project_conv_1_running_var,
    b_blocks_5_expand_conv_1_running_mean, b_blocks_5_expand_conv_1_running_var,
    b_blocks_5_depthwise_conv_1_running_mean, b_blocks_5_depthwise_conv_1_running_var,
    b_blocks_5_project_conv_1_running_mean, b_blocks_5_project_conv_1_running_var,
    b_blocks_6_expand_conv_1_running_mean, b_blocks_6_expand_conv_1_running_var,
    b_blocks_6_depthwise_conv_1_running_mean, b_blocks_6_depthwise_conv_1_running_var,
    b_blocks_6_project_conv_1_running_mean, b_blocks_6_project_conv_1_running_var,
    b_blocks_7_expand_conv_1_running_mean, b_blocks_7_expand_conv_1_running_var,
    b_blocks_7_depthwise_conv_1_running_mean, b_blocks_7_depthwise_conv_1_running_var,
    b_blocks_7_project_conv_1_running_mean, b_blocks_7_project_conv_1_running_var,
    b_blocks_8_expand_conv_1_running_mean, b_blocks_8_expand_conv_1_running_var,
    b_blocks_8_depthwise_conv_1_running_mean, b_blocks_8_depthwise_conv_1_running_var,
    b_blocks_8_project_conv_1_running_mean, b_blocks_8_project_conv_1_running_var,
    b_blocks_9_expand_conv_1_running_mean, b_blocks_9_expand_conv_1_running_var,
    b_blocks_9_depthwise_conv_1_running_mean, b_blocks_9_depthwise_conv_1_running_var,
    b_blocks_9_project_conv_1_running_mean, b_blocks_9_project_conv_1_running_var,
    b_blocks_10_expand_conv_1_running_mean, b_blocks_10_expand_conv_1_running_var,
    b_blocks_10_depthwise_conv_1_running_mean, b_blocks_10_depthwise_conv_1_running_var,
    b_blocks_10_project_conv_1_running_mean, b_blocks_10_project_conv_1_running_var,
    b_blocks_11_expand_conv_1_running_mean, b_blocks_11_expand_conv_1_running_var,
    b_blocks_11_depthwise_conv_1_running_mean, b_blocks_11_depthwise_conv_1_running_var,
    b_blocks_11_project_conv_1_running_mean, b_blocks_11_project_conv_1_running_var,
    b_blocks_12_expand_conv_1_running_mean, b_blocks_12_expand_conv_1_running_var,
    b_blocks_12_depthwise_conv_1_running_mean, b_blocks_12_depthwise_conv_1_running_var,
    b_blocks_12_project_conv_1_running_mean, b_blocks_12_project_conv_1_running_var,
    b_bn2_running_mean, b_bn2_running_var,
    x,
):
    # ── Initial conv + BN + ReLU ──────────────────────────────────────────
    x = conv2d(x, p_conv1_weight, stride=2, padding=1)
    x = bn_act(x, b_bn1_running_mean, b_bn1_running_var,
               p_bn1_weight, p_bn1_bias, act=1)

    # ── Block 0: MBConv1 (32->16, stride=1, expand=1) ─────────────────────
    # depthwise conv (groups=32)
    x = conv2d(x, p_blocks_0_depthwise_conv_0_weight, stride=1, padding=1, groups=32)
    x = bn_act(x, b_blocks_0_depthwise_conv_1_running_mean,
               b_blocks_0_depthwise_conv_1_running_var,
               p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias, act=2)
    # project conv
    x = conv2d(x, p_blocks_0_project_conv_0_weight, stride=1, padding=0)
    x = bn_act(x, b_blocks_0_project_conv_1_running_mean,
               b_blocks_0_project_conv_1_running_var,
               p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias, act=0)

    # ── Block 1: MBConv6 (16->24, stride=2, expand=6) ─────────────────────
    x = conv2d(x, p_blocks_1_expand_conv_0_weight, stride=1, padding=0)
    x = bn_act(x, b_blocks_1_expand_conv_1_running_mean,
               b_blocks_1_expand_conv_1_running_var,
               p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias, act=2)
    x = conv2d(x, p_blocks_1_depthwise_conv_0_weight, stride=2, padding=1, groups=96)
    x = bn_act(x, b_blocks_1_depthwise_conv_1_running_mean,
               b_blocks_1_depthwise_conv_1_running_var,
               p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias, act=2)
    x = conv2d(x, p_blocks_1_project_conv_0_weight, stride=1, padding=0)
    x = bn_act(x, b_blocks_1_project_conv_1_running_mean,
               b_blocks_1_project_conv_1_running_var,
               p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias, act=0)

    # ── Block 2: MBConv6 (24->24, stride=1, expand=6) -- has residual ──────
    residual = x
    x = conv2d(x, p_blocks_2_expand_conv_0_weight, stride=1, padding=0)
    x = bn_act(x, b_blocks_2_expand_conv_1_running_mean,
               b_blocks_2_expand_conv_1_running_var,
               p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias, act=2)
    x = conv2d(x, p_blocks_2_depthwise_conv_0_weight, stride=1, padding=1, groups=144)
    x = bn_act(x, b_blocks_2_depthwise_conv_1_running_mean,
               b_blocks_2_depthwise_conv_1_running_var,
               p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias, act=2)
    x = conv2d(x, p_blocks_2_project_conv_0_weight, stride=1, padding=0)
    x = bn_add(x, residual,
               b_blocks_2_project_conv_1_running_mean,
               b_blocks_2_project_conv_1_running_var,

    # ── Block 3: MBConv6 (24->40, stride=2, expand=6, k=5) ────────────────
    x = conv2d(x, p_blocks_3_expand_conv_0_weight, stride=1, padding=0)
    x = bn_act(x, b_blocks_3_expand_conv_1_running_mean,
               b_blocks_3_expand_conv_1_running_var,
               p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias, act=2)
    x = conv2d(x, p_blocks_3_depthwise_conv_0_weight,