import torch
import triton
import triton.language as tl
import torch.nn.functional as F

# ─── Fused BN + optional activation (relu / hardtanh / none) ───────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_act_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps,
    act: tl.constexpr,   # 0=none, 1=relu, 2=hardtanh6
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # decode channel index
    c = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c, mask=mask, other=1.0).to(tl.float32)
    w    = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    y = x_norm * w + b

    if act == 1:
        y = tl.maximum(y, 0.0)
    elif act == 2:
        y = tl.minimum(tl.maximum(y, 0.0), 6.0)

    tl.store(out_ptr + offsets, y.to(tl.float32), mask=mask)


def fused_bn_act(x, weight, bias, running_mean, running_var, eps=1e-5, act=0):
    """act: 0=none, 1=relu, 2=hardtanh(0,6)"""
    out = torch.empty_like(x)
    numel = x.numel()
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3] if x.ndim == 4 else 1

    # make contiguous
    x_c = x.contiguous()

    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_act_kernel[grid](
        x_c, out,
        running_mean, running_var, weight, bias,
        numel, C, HW,
        eps,
        act,
    )
    return out


# ─── Fused Linear (GEMM) with bias ─────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

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
    tl.store(out_ptrs, acc, mask=out_mask)


def triton_linear(x, weight, bias):
    """x: (M, K), weight: (N, K), bias: (N,) -> (M, N)"""
    M, K = x.shape
    N = weight.shape[0]
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, weight, bias, out,
        M, N, K,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
    )
    return out


# ─── Adaptive Average Pool 2D (global) ─────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_HW': 256}, num_warps=8),
        triton.Config({'BLOCK_HW': 32},  num_warps=2),
    ],
    key=['HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_HW: tl.constexpr,
):
    # one program per (n, c)
    pid = tl.program_id(0)
    n = pid // C
    c = pid % C

    base = n * C * HW + c * HW
    acc = tl.zeros((1,), dtype=tl.float32)

    for hw_off in range(0, HW, BLOCK_HW):
        offs = hw_off + tl.arange(0, BLOCK_HW)
        mask = offs < HW
        vals = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        acc += tl.sum(vals, axis=0)

    tl.store(out_ptr + pid, (acc / HW).to(tl.float32))


def triton_adaptive_avg_pool(x):
    """x: (N, C, H, W) -> (N, C, 1, 1)"""
    N, C, H, W = x.shape
    HW = H * W
    x_c = x.contiguous()
    out = torch.empty((N, C, 1, 1), device=x.device, dtype=x.dtype)
    grid = (N * C,)
    adaptive_avg_pool_kernel[grid](
        x_c, out.view(N * C),
        N, C, HW,
    )
    return out


# ─── Conv2d wrapper (uses cuDNN) ────────────────────────────────────────────

def apply_conv2d(x, weight, stride=1, padding=0, groups=1):
    return F.conv2d(x, weight, bias=None, stride=stride, padding=padding, groups=groups)


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
    # ── Initial stem ──────────────────────────────────────────────────────
    x = apply_conv2d(x, p_conv1_weight, stride=2, padding=1)
    x = fused_bn_act(x, p_bn1_weight, p_bn1_bias,
                     b_bn1_running_mean, b_bn1_running_var, act=1)

    # ── Block 0: MBConv1 (32->16, k=3, s=1, expand=1) ────────────────────
    # depthwise
    x0 = apply_conv2d(x, p_blocks_0_depthwise_conv_0_weight, stride=1, padding=1, groups=32)
    x0 = fused_bn_act(x0, p_blocks_0_depthwise_conv_1_weight, p_blocks_0_depthwise_conv_1_bias,
                      b_blocks_0_depthwise_conv_1_running_mean, b_blocks_0_depthwise_conv_1_running_var, act=2)
    # project
    x0 = apply_conv2d(x0, p_blocks_0_project_conv_0_weight, stride=1, padding=0)
    x0 = fused_bn_act(x0, p_blocks_0_project_conv_1_weight, p_blocks_0_project_conv_1_bias,
                      b_blocks_0_project_conv_1_running_mean, b_blocks_0_project_conv_1_running_var, act=0)
    x = x0  # no residual for block 0 (in_c != out_c)

    # ── Block 1: MBConv6 (16->24, k=3, s=2, expand=6) ────────────────────
    # expand
    x1 = apply_conv2d(x, p_blocks_1_expand_conv_0_weight, stride=1, padding=0)
    x1 = fused_bn_act(x1, p_blocks_1_expand_conv_1_weight, p_blocks_1_expand_conv_1_bias,
                      b_blocks_1_expand_conv_1_running_mean, b_blocks_1_expand_conv_1_running_var, act=2)
    # depthwise
    x1 = apply_conv2d(x1, p_blocks_1_depthwise_conv_0_weight, stride=2, padding=1, groups=96)
    x1 = fused_bn_act(x1, p_blocks_1_depthwise_conv_1_weight, p_blocks_1_depthwise_conv_1_bias,
                      b_blocks_1_depthwise_conv_1_running_mean, b_blocks_1_depthwise_conv_1_running_var, act=2)
    # project
    x1 = apply_conv2d(x1, p_blocks_1_project_conv_0_weight, stride=1, padding=0)
    x1 = fused_bn_act(x1, p_blocks_1_project_conv_1_weight, p_blocks_1_project_conv_1_bias,
                      b_blocks_1_project_conv_1_running_mean, b_blocks_1_project_conv_1_running_var, act=0)
    x = x1  # no residual (stride=2)

    # ── Block 2: MBConv6 (24->24, k=3, s=1, expand=6) ────────────────────
    residual = x
    # expand
    x2 = apply_conv2d(x, p_blocks_2_expand_conv_0_weight, stride=1, padding=0)
    x2 = fused_bn_act(x2, p_blocks_2_expand_conv_1_weight, p_blocks_2_expand_conv_1_bias,
                      b_blocks_2_expand_conv_1_running_mean, b_blocks_2_expand_conv_1_running_var, act=2)
    # depthwise
    x2 = apply_conv2d(x2, p_blocks_2_depthwise_conv_0_weight, stride=1, padding=1, groups=144)
    x2 = fused_bn_act(x2, p_blocks_2_depthwise_conv_1_weight, p_blocks_2_depthwise_conv_1_bias,
                      b_blocks_2_depthwise_conv_1_running_mean, b_blocks_2_depthwise_conv_1_running_var, act=2)
    # project
    x2 = apply_conv2d(x2, p_blocks_2_project_conv_0_weight, stride=1, padding=0)
    x2 = fused_bn_act(x2, p_blocks_2_project_conv_1_weight, p_blocks_2_project_conv_1_bias,
                      b_blocks_2_project_conv_1_running_mean, b_blocks_2_project_conv_1_running_var, act=0)
    x = x2 + residual

    # ── Block 3: MBConv6 (24->40, k=5, s=2, expand=6) ────────────────────
    # expand
    x3 = apply_conv2d(x, p_blocks_3_expand_conv_0_weight, stride=1, padding=0)
    x3 = fused_bn_act(x3, p_blocks_3_expand_conv_1_weight, p_blocks_3_expand_conv_1_bias,
                      b_blocks_3_expand_conv_1_running_mean, b_blocks_3_expand_conv_1_running_var, act=2)
    # depthwise k=5, s=2, pad=2
    x3 = apply_conv2d(x3, p_blocks_3_depthwise_conv_0_weight, stride=2, padding=2, groups=144)
    x3 = fused_bn_act(x3, p_blocks_3_depthwise_conv_1_weight, p_blocks_3_depthwise_conv_1_bias,
                      b_blocks_3_depthwise_conv_1_running_mean, b_blocks_3_depthwise_conv_1_running_var, act=2)
    # project
    x3 = apply_conv2d(x3, p_blocks_3_project_conv_0_weight, stride=1, padding=0)
    x3 = fused_bn_act(x3, p_blocks_3_project_conv_1_weight, p_blocks_3_project_conv_1_bias,
                      b_blocks_3_project_conv_1_running_mean, b_blocks_3_project_conv_1_running_var, act=0)
    x = x3  # no residual (stride=2)

    # ── Block 4: MBConv6 (40->40, k=5, s=1, expand=6) ────────────────────
    residual = x
    # expand
    x4 = apply_conv2d(x, p_blocks_4_expand_conv_0_weight, stride=1, padding=0)
    x4 = fused_bn_act(x4, p_blocks_4_expand_conv_1_weight, p_blocks_4_expand_conv_1_bias,
                      b_blocks_4_expand_conv_1_running_mean, b_blocks_4_expand_conv_1_running_var, act=2)
    # depthwise k=5, s=1, pad=2
    x4 = apply_conv2d(x4, p_blocks_4_depthwise_conv_0_weight, stride=1, padding=2, groups=240)
    x4 = fused_bn_act(x4, p_blocks_4_depthwise_conv_1_weight, p_blocks_4_depthwise_conv_1_bias,
                      b_blocks_4_depthwise_conv_1_running_mean, b_blocks_4_depthwise_conv_1_running_var, act=2)
    # project
    x4 = apply_conv2d(x4, p_blocks_4_project_conv_0_weight, stride=1, padding=0