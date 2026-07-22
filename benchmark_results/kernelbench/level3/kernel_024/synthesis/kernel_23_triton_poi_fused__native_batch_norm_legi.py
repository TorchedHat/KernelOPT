import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import math

# ─── Fused BN + ReLU pointwise kernel ────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    numel, C, HW,
    eps,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # channel index: layout is (N, C, H, W) -> flat index // HW % C
    c_idx = (offsets // HW) % C

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c_idx, mask=mask, other=1.0).to(tl.float32)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0).to(tl.float32)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    if do_relu:
        out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out.to(tl.float32), mask=mask)


# ─── Fused sigmoid pointwise kernel ──────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + tl.exp(-x))
    tl.store(out_ptr + offsets, out, mask=mask)


# ─── ReLU pointwise kernel ────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def relu_kernel(x_ptr, out_ptr, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    out = tl.maximum(x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ─── Adaptive avg pool 2d -> (1,1) kernel ─────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 16}, num_warps=2),
        triton.Config({'BLOCK_C': 8},  num_warps=2),
    ],
    key=['N', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool_kernel(
    x_ptr, out_ptr,
    N, C, HW,
    BLOCK_C: tl.constexpr,
):
    # one program per (n, c_block)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)

    c_offsets = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offsets < C

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    base = pid_n * C * HW + c_offsets * HW

    for hw in range(HW):
        ptrs = base + hw
        vals = tl.load(x_ptr + ptrs, mask=c_mask, other=0.0).to(tl.float32)
        acc += vals

    acc = acc / HW

    out_ptrs = pid_n * C + c_offsets
    tl.store(out_ptr + out_ptrs, acc, mask=c_mask)


# ─── Multiply (SE attention) kernel ──────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=4),
    ],
    key=['numel', 'C', 'HW'],
)
@triton.jit
def se_mul_kernel(
    feat_ptr, scale_ptr, out_ptr,
    numel, C, HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # scale is (N, C, 1, 1) -> index by (n, c)
    n_idx = offsets // (C * HW)
    c_idx = (offsets // HW) % C
    scale_idx = n_idx * C + c_idx

    feat  = tl.load(feat_ptr  + offsets,   mask=mask, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + scale_idx, mask=mask, other=0.0).to(tl.float32)

    tl.store(out_ptr + offsets, feat * scale, mask=mask)


def apply_bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    N, C, H, W = x.shape
    numel = x.numel()
    HW = H * W
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias,
        out,
        numel, C, HW,
        eps,
        do_relu,
    )
    return out


def apply_sigmoid(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    sigmoid_kernel[grid](x, out, numel)
    return out


def apply_relu(x):
    numel = x.numel()
    out = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def apply_adaptive_avg_pool(x):
    N, C, H, W = x.shape
    HW = H * W
    out = torch.empty(N, C, device=x.device, dtype=x.dtype)
    BLOCK_C = min(64, triton.next_power_of_2(C))
    grid = (N, triton.cdiv(C, BLOCK_C))
    adaptive_avg_pool_kernel[grid](x, out, N, C, HW)
    return out.view(N, C, 1, 1)


def apply_se_mul(feat, scale):
    # feat: (N, C, H, W), scale: (N, C, 1, 1)
    N, C, H, W = feat.shape
    HW = H * W
    numel = feat.numel()
    out = torch.empty_like(feat)
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    se_mul_kernel[grid](feat, scale, out, numel, C, HW)
    return out


def conv2d_wrap(x, w, stride=1, padding=0, groups=1):
    """Use torch.conv2d - this is the low-level call, not F.conv2d"""
    return torch.conv2d(x, w, bias=None, stride=stride, padding=padding, groups=groups)


def mbconv_block(x, weights, bn_stats, block_idx):
    """
    """
    w = weights
    s = bn_stats

    # Expansion conv (1x1)
    x_exp = conv2d_wrap(x, w['expand'], stride=1, padding=0)
    x_exp = apply_bn_relu(x_exp, s['expand_mean'], s['expand_var'],
                          s['expand_w'], s['expand_b'], do_relu=True)

    # Depthwise conv (3x3)
    C_exp = x_exp.shape[1]
    x_dw = conv2d_wrap(x_exp, w['dw'], stride=w['stride'], padding=1, groups=C_exp)
    x_dw = apply_bn_relu(x_dw, s['dw_mean'], s['dw_var'],
                         s['dw_w'], s['dw_b'], do_relu=True)

    # SE
    x_se = apply_adaptive_avg_pool(x_dw)
    x_se = conv2d_wrap(x_se, w['se_reduce'], stride=1, padding=0)
    x_se = apply_relu(x_se)
    x_se = conv2d_wrap(x_se, w['se_expand'], stride=1, padding=0)
    x_se = apply_sigmoid(x_se)
    x_dw = apply_se_mul(x_dw, x_se)

    # Output conv (1x1)
    x_out = conv2d_wrap(x_dw, w['project'], stride=1, padding=0)
    x_out = apply_bn_relu(x_out, s['proj_mean'], s['proj_var'],
                          s['proj_w'], s['proj_b'], do_relu=False)

    return x_out


# ─── Linear kernel (for final FC) ────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=8, num_stages=2),
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

    # Add bias
    b = tl.load(b_ptr + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
    acc = acc + b[None, :]

    out_ptrs = out_ptr + offs_m[:, None] * N + offs_n[None, :]
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc, mask=out_mask)


def apply_linear(x, weight, bias):
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


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_mbconv1_0_weight, p_mbconv1_1_weight, p_mbconv1_1_bias,
    p_mbconv1_3_weight, p_mbconv1_4_weight, p_mbconv1_4_bias,
    p_mbconv1_7_weight, p_mbconv1_9_weight,
    p_mbconv1_11_weight, p_mbconv1_12_weight, p_mbconv1_12_bias,
    p_mbconv2_0_weight, p_mbconv2_1_weight, p_mbconv2_1_bias,
    p_mbconv2_3_weight, p_mbconv2_4_weight, p_mbconv2_4_bias,
    p_mbconv2_7_weight, p_mbconv2_9_weight,
    p_mbconv2_11_weight, p_mbconv2_12_weight, p_mbconv2_12_bias,
    p_mbconv3_0_weight, p_mbconv3_1_weight, p_mbconv3_1_bias,
    p_mbconv3_3_weight, p_mbconv3_4_weight, p_mbconv3_4_bias,
    p_mbconv3_7_weight, p_mbconv3_9_weight,
    p_mbconv3_11_weight, p_mbconv3_12_weight, p_mbconv3_12_bias,
    p_mbconv4_0_weight, p_mbconv4_1_weight, p_mbconv4_1_bias,
    p_mbconv4_3_weight, p_mbconv4_4_weight, p_mbconv4_4_bias,
    p_mbconv4_7_weight, p_mbconv4_9_weight,
    p_mbconv4_11_weight, p_mbconv4_12_weight, p_mbconv4_12_bias,
    p_mbconv5_0_weight, p_mbconv5_1_weight, p_mbconv5_1_bias,
    p_mbconv5_3_weight, p_mbconv5_4_weight, p_mbconv5_4_bias,
    p_mbconv5_7_weight, p_mbconv5_9_weight,
    p_mbconv5_11_weight, p_mbconv5_12_weight, p_mbconv5_12_bias,
    p_conv_final_weight, p_bn_final_weight, p_bn_final_bias,
    p_fc_weight, p_fc_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_mbconv1_1_running_mean, b_mbconv1_1_running_var,
    b_mbconv1_4_running_mean, b_mbconv1_4_running_var,
    b_mbconv1_12_running_mean, b_mbconv1_12_running_var,
    b_mbconv2_1_running_mean, b_mbconv2_1_running_var,
    b_mbconv2_4_running_mean, b_mbconv2_4_running_var,
    b_mbconv2_12_running_mean, b_mbconv2_12_running_var,
    b_mbconv3_1_running_mean, b_mbconv3_1_running_var,
    b_mbconv3_4_running_mean, b_mbconv3_4_running_var,
    b_mbconv3_12_running_mean, b_mbconv3_12_running_var,
    b_mbconv4_1_running_mean, b_mbconv4_1_running_var,
    b_mbconv4_4_running_mean, b_mbconv4_4_running_var,
    b_mbconv4_12_running_mean, b_mbconv4_12_running_var,
    b_mbconv5_1_running_mean, b_mbconv5_1_running_var,
    b_mbconv5_4_running_mean, b_mbconv5_4_running_var,
    b_mbconv5_12_running_mean, b_mbconv5_12_running_var,
    b_bn_final_running_mean, b_bn_final_running_var,
    x,
):
    # ── Initial conv + BN + ReLU ──────────────────────────────────────────────
    x = conv2d_wrap(x, p_conv1_weight, stride=2, padding=1)
    x = apply_bn_relu(x, b_bn1_running_mean, b_bn1_running_var,
                      p_bn1_weight, p_bn1_bias, do_relu=True)

    # ── MBConv1: 32->96, stride=1, expand=3 ───────────────────────────────────
    # expand conv
    x1 = conv2d_wrap(x, p_mbconv1_0_weight, stride=1, padding=0)
    x1 = apply_bn_relu(x1, b_mbconv1_1_running_mean, b_mbconv1_1_running_var,
                       p_mbconv1_1_weight, p_mbconv1_1_bias, do_relu=True)
    # depthwise
    x1 = conv2d_wrap(x1, p_mbconv1_3_weight, stride=1, padding=1, groups=96)
    x1 = apply_bn_relu(x1, b_mbconv1_4_running_mean, b_mbconv1_4_running_var,
                       p_mbconv1_4_weight, p_mbconv1_4_bias, do_relu=True)
    # SE
    x1_se = apply_adaptive_avg_pool(x1)
    x1_se = conv2d_wrap(x1_se, p_mbconv1_7_weight, stride=1, padding=0)
    x1_se = apply_relu(x1_se)
    x1_se = conv2d_wrap(x1_se, p_mbconv1_9_weight, stride=1, padding=0)
    x1_se = apply_sigmoid(x1_se)
    x1 = apply_se_mul(x1, x1_se)
    # project
    x1 = conv2d_wrap(x1, p_mbconv1_11_weight, stride=1, padding=0)
    x1 = apply_bn_relu(x1, b_mbconv1_12_running_mean, b_mbconv1_12_running_var,
                       p_mbconv1_12_weight, p_mbconv1_12_bias, do_relu=False)

    # ── MBConv2: 96->144, stride=2, expand=6 ──────────────────────────────────
    x2 = conv2d_wrap(x1, p_mbconv2_0_weight, stride=1, padding=0)
    x2 = apply_bn_relu(x2, b_mbconv2_1_running_mean, b_mbconv2_1_running_var,
                       p_mbconv2_1_weight, p_mbconv2_1_bias, do_relu=True)
    x2 = conv2d_wrap(x2, p_mbconv2_3_weight, stride=2, padding=1, groups=576)
    x2 = apply_bn_relu(x2, b_mbconv2_4_running_mean, b_mbconv2_4_running_var,
                       p_mbconv2_4_weight, p_mbconv2_4_bias, do_relu=True)
    x2_se = apply_adaptive_avg_pool(x2)
    x2_se = conv2d_wrap(x2_se, p_mbconv2_7_weight, stride=1, padding=0)
    x2_se = apply_relu(x2_se)
    x2_se = conv2d_wrap(x2_se, p_mbconv2_9_weight, stride=1, padding=0)
    x2_se = apply_sigmoid(x2_se)
    x2 = apply_se_mul(x2, x2_se)
    x2 = conv2d_wrap(x2, p_mbconv2_11_weight, stride=1, padding=0)
    x2 = apply_bn_relu(x2, b_mbconv2_12_running_mean, b_mbconv2_12_running_var,
                       p_mbconv2_12_weight, p_mbconv2_12_bias, do_relu=False)

    # ── MBConv3: 144->192, stride=2, expand=6 ─────────────────────────────────
    x3 = conv2d_wrap(x2, p_mbconv3_0_weight, stride=1, padding=0)
    x3 = apply_bn_relu(x3, b_mbconv3_1_running_mean, b_mbconv3_1_running_var,
                       p_mbconv3_1_weight, p_mbconv3_1_bias, do_relu=True)
    x3 = conv2d_wrap(x3, p_mbconv3_3_weight, stride=2, padding=1, groups=864)
    x3 = apply_bn_relu(x3, b_mbconv3_4_running_mean, b_mbconv3_4_running_var,
                       p_mbconv3_4_weight, p_mbconv3_4_bias, do_relu=True)
    x3_se = apply_adaptive_avg_pool(x3)
    x3_se = conv2d_wrap(x3_se, p_mbconv3_7_weight, stride=1, padding=0)
    x3_se = apply_relu(x3_se)
    x3_se = conv2d_wrap(x3_se, p_mbconv3_9_weight, stride=1, padding=0)
    x3_se = apply_sigmoid(x3_se)
    x3 = apply_se_mul(x3, x3_se)
    x3 = conv2d_wrap(x3, p_mbconv3_11_weight, stride=1, padding=0)
    x3 = apply_bn_relu(x3, b_mbconv3_12_running_mean, b_mbconv3_12_running_var,
                       p_mbconv3_12_weight, p_mbconv3_12_bias, do_relu=False)

    # ── MBConv4: 192->288, stride=2, expand=6 ─────────────────────────────────
    x4 = conv2d_wrap(x3, p_mbconv4_0_weight, stride=1, padding=0)
    x4 = apply_bn_relu(x4, b_mbconv4_1_running_mean, b_mbconv4_1_running_var,
                       p_mbconv4_1_weight, p_mbconv4_1_bias, do_relu=True)
    x4 = conv2d_wrap(x4, p_mbconv4_3_weight, stride=2, padding=1, groups=1152)
    x4 = apply_bn_relu(x4, b_mbconv4_4_running_mean, b_mbconv4_4_running_var,
                       p_mbconv4_4_weight, p_mbconv4_4_bias, do_relu=True)
    x4_se = apply_adaptive_avg_pool(x4)
    x4_se = conv2d_wrap(x4_se, p_mbconv4_7_weight, stride=1, padding=0)
    x4_se = apply_relu(x4_se)
    x4_se = conv2d_wrap(x4_se, p_mbconv4_9_weight, stride=1, padding=0)
    x4_se = apply_sigmoid(x4_se)
    x4 = apply_se_mul(x4, x4_se)
    x4 = conv2d_wrap(x4, p_mbconv4_11_weight, stride=1, padding=0)
    x4 = apply_bn_relu(x4, b_mbconv4_12_running_mean, b_mbconv4_12_running_var,
                       p_mbconv4_12_weight, p_mbconv4_12_bias, do_relu=False)

    # ── MBConv5: 288->384, stride=1, expand=6 ─────────────────────────────────
    x5 = conv2d_wrap(x4, p_mbconv5_0_weight, stride=1, padding=0)
    x5 = apply_bn_relu(x5, b_mbconv5_1_running_mean, b_mbconv5_1_running_var,
                       p_mbconv5_1_weight, p_mbconv5_1_bias, do_relu=True)
    x5 = conv2d_wrap(x5, p_mbconv5_3_weight, stride=1, padding=1, groups=1728)
    x5 = apply_bn_relu(x5,