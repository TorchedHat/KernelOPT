import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math

# ─── Fused BN + ReLU pointwise kernel ────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=4),
        triton.Config({'BLOCK': 128},  num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, mean_ptr, var_ptr, weight_ptr, bias_ptr,
    out_ptr,
    C, HW,
    N,  # total elements = B*C*HW
    eps,
    do_relu: tl.constexpr,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    # channel index
    c = (offs // HW) % C
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(mean_ptr + c, mask=mask, other=0.0).to(tl.float32)
    var  = tl.load(var_ptr  + c, mask=mask, other=1.0).to(tl.float32)
    w    = tl.load(weight_ptr + c, mask=mask, other=1.0).to(tl.float32)
    b    = tl.load(bias_ptr   + c, mask=mask, other=0.0).to(tl.float32)
    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    if do_relu:
        out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


# ─── Fused ReLU pointwise kernel ─────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=4),
        triton.Config({'BLOCK': 128},  num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def relu_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


# ─── Sigmoid pointwise kernel ─────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK': 1024}, num_warps=4),
        triton.Config({'BLOCK': 512},  num_warps=4),
        triton.Config({'BLOCK': 256},  num_warps=4),
        triton.Config({'BLOCK': 128},  num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def sigmoid_kernel(x_ptr, out_ptr, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(x_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    out = 1.0 / (1.0 + libdevice.exp(-x))
    tl.store(out_ptr + offs, out.to(tl.float32), mask=mask)


# ─── Adaptive Average Pool 2D (global) kernel ────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 32}, num_warps=4),
        triton.Config({'BLOCK_C': 16}, num_warps=2),
        triton.Config({'BLOCK_C': 8},  num_warps=2),
    ],
    key=['B', 'C', 'HW'],
)
@triton.jit
def adaptive_avg_pool2d_kernel(
    x_ptr, out_ptr,
    B, C, HW,
    BLOCK_C: tl.constexpr,
):
    # Each program handles one (batch, channel_block)
    pid_b = tl.program_id(0)
    pid_c = tl.program_id(1)
    c_offs = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = c_offs < C

    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    base = pid_b * C * HW + c_offs * HW
    for hw in range(HW):
        vals = tl.load(x_ptr + base + hw, mask=c_mask, other=0.0).to(tl.float32)
        acc += vals
    acc = acc / HW
    out_offs = pid_b * C + c_offs
    tl.store(out_ptr + out_offs, acc.to(tl.float32), mask=c_mask)


# ─── Implicit GEMM Conv2d kernel ──────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    B, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    GROUPS,
    M, N, K,
    # strides for X (NCHW)
    sx_n, sx_c, sx_h, sx_w,
    # strides for W (NCHW: out_c, in_c, kh, kw)
    sw_o, sw_i, sw_h, sw_w,
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

    # Group info
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
    group_idx = offs_n // GROUP_OUT_C  # which group each output channel belongs to

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # K = GROUP_IN_C * KH * KW
    for k in range(0, K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c_local, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        ic_local = tmp_k // KH  # local channel within group

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        # For grouped conv, each output channel group uses different input channels
        # group_idx is per output channel (BLOCK_N), ic_local is per K
        # For simplicity with groups=1 or depthwise, handle carefully
        # ic_global = group_idx[None, :] * GROUP_IN_C + ic_local[:, None]  -- but we need [M,K] for X
        # X tile: [BLOCK_M, BLOCK_K]
        # For groups=1: ic_global = ic_local
        ic_global_x = ic_local  # shape [BLOCK_K]

        valid_x = (
            (batch_idx[:, None] < B) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < K)
        )
        x_ptrs = (X_ptr +
                  batch_idx[:, None] * sx_n +
                  ic_global_x[None, :] * sx_c +
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # W tile: [BLOCK_K, BLOCK_N]
        # w[out_c, ic_local, kh, kw]
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * sw_o +
                  ic_local[:, None] * sw_i +
                  kh_idx[:, None] * sw_h +
                  kw_idx[:, None] * sw_w)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output [BLOCK_M, BLOCK_N] -> out[batch, out_c, out_h, out_w]
    valid_out = (
        (batch_idx[:, None] < B) &
        (out_h_idx[:, None] < OUT_H) &
        (out_w_idx[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )
    out_ptrs = (Out_ptr +
                batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
                offs_n[None, :] * (OUT_H * OUT_W) +
                out_h_idx[:, None] * OUT_W +
                out_w_idx[:, None])
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


# ─── Depthwise Conv2d kernel ──────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_BC': 64,  'BLOCK_HW': 64},  num_warps=4),
        triton.Config({'BLOCK_BC': 128, 'BLOCK_HW': 32},  num_warps=4),
        triton.Config({'BLOCK_BC': 32,  'BLOCK_HW': 128}, num_warps=4),
        triton.Config({'BLOCK_BC': 64,  'BLOCK_HW': 32},  num_warps=2),
    ],
    key=['BC', 'OUT_HW'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    B, C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BC, OUT_HW,
    BLOCK_BC: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_bc = tl.program_id(0)
    pid_hw = tl.program_id(1)

    offs_bc = pid_bc * BLOCK_BC + tl.arange(0, BLOCK_BC)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    # Decode bc -> (batch, channel)
    c_idx = offs_bc % C
    b_idx = offs_bc // C

    # Decode hw -> (out_h, out_w)
    out_w_idx = offs_hw % OUT_W
    out_h_idx = offs_hw // OUT_W

    mask_bc = offs_bc < BC
    mask_hw = offs_hw < OUT_HW

    # For each (b,c) and (oh,ow), compute depthwise conv
    # Output: [BLOCK_BC, BLOCK_HW]
    acc = tl.zeros([BLOCK_BC, BLOCK_HW], dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h_idx[None, :] * STRIDE_H + kh - PAD_H
            iw = out_w_idx[None, :] * STRIDE_W + kw - PAD_W

            valid = (
                mask_bc[:, None] &
                mask_hw[None, :] &
                (b_idx[:, None] < B) &
                (ih >= 0) & (ih < IN_H) &
                (iw >= 0) & (iw < IN_W)
            )

            x_ptrs = (X_ptr +
                      b_idx[:, None] * (C * IN_H * IN_W) +
                      c_idx[:, None] * (IN_H * IN_W) +
            x_vals = tl.load(x_ptrs, mask=valid, other=0.0).to(tl.float32)

            # Weight: [C, 1, KH, KW] -> index [c, 0, kh, kw]
            w_ptrs = W_ptr + c_idx * (KH * KW) + kh * KW + kw
            w_vals = tl.load(w_ptrs, mask=mask_bc, other=0.0).to(tl.float32)

            acc += x_vals * w_vals[:, None]

    out_ptrs = (Out_ptr +
                b_idx[:, None] * (C * OUT_H * OUT_W) +
                c_idx[:, None] * (OUT_H * OUT_W) +
                out_h_idx[None, :] * OUT_W +
                out_w_idx[None, :])
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_bc[:, None] & mask_hw[None, :])


# ─── Linear kernel (GEMM: [M,K] x [K,N] -> [M,N]) ───────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=2),
    ],
    key=['M_dim', 'N_dim', 'K_dim'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M_dim, N_dim, K_dim,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    for k in range(0, K_dim, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M_dim) & (offs_k[None, :] < K_dim)
        w_mask = (offs_k[:, None] < K_dim) & (offs_n[None, :] < N_dim)
        x_tile = tl.load(X_ptr + offs_m[:, None] * K_dim + offs_k[None, :], mask=x_mask, other=0.0)
        w_tile = tl.load(W_ptr + offs_k[:, None] * N_dim + offs_n[None, :], mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N_dim, other=0.0).to(tl.float32)
    acc += bias[None, :]

    out_mask = (offs_m[:, None] < M_dim) & (offs_n[None, :] < N_dim)
    tl.store(Out_ptr + offs_m[:, None] * N_dim + offs_n[None, :], acc.to(tl.float32), mask=out_mask)


# ─── Helper functions ─────────────────────────────────────────────────────────

def triton_conv2d(x, w, stride=(1,1), padding=(0,0), groups=1):
    """Implicit GEMM conv2d using Triton kernel."""
    B, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = w.shape
    SH, SW = stride
    PH, PW = padding
    OUT_H = (IN_H + 2*PH - KH) // SH + 1
    OUT_W = (IN_W + 2*PW - KW) // SW + 1

    if groups > 1 and OUT_C == IN_C and groups == IN_C:
        # Depthwise convolution
        out = torch.empty(B, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
        BC = B * IN_C
        OUT_HW = OUT_H * OUT_W
        grid = lambda meta: (
            triton.cdiv(BC, meta['BLOCK_BC']),
            triton.cdiv(OUT_HW, meta['BLOCK_HW']),
        )
        depthwise_conv2d_kernel[grid](
            x, w, out,
            B, IN_C, IN_H, IN_W,
            OUT_H, OUT_W,
            KH, KW,
            SH, SW,
            PH, PW,
            BC, OUT_HW,
        )
        return out
    else:
        # Standard conv via implicit GEMM
        out = torch.empty(B, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)
        GROUP_IN_C = IN_C // groups
        K = GROUP_IN_C * KH * KW
        M = B * OUT_H * OUT_W
        N = OUT_C

        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )
        conv2d_implicit_gemm_kernel[grid](
            x, w, out,
            B, IN_C, IN_H, IN_W,
            OUT_C, OUT_H, OUT_W,
            KH, KW,
            SH, SW,
            PH, PW,
            groups,
            M, N, K,
            x.stride(0), x.stride(1), x.stride(2), x.stride(3),
            w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        )
        return out


def triton_bn_relu(x, mean, var, weight, bias, eps=1e-5, do_relu=True):
    out = torch.empty_like(x)
    N = x.numel()
    C = x.shape[1]
    HW = x.shape[2] * x.shape[3] if x.ndim == 4 else 1
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    bn_relu_kernel[grid](
        x, mean, var, weight, bias, out,
        C, HW, N, eps, do_relu,
    )
    return out


def triton_relu(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    relu_kernel[grid](x, out, N)
    return out


def triton_sigmoid(x):
    out = torch.empty_like(x)
    N = x.numel()
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK']),)
    sigmoid_kernel[grid](x, out, N)
    return out


def triton_adaptive_avg_pool2d(x):
    """Global average pooling: [B,C,H,W] -> [B,C,1,1]"""
    B, C, H, W = x.shape
    HW = H * W
    out = torch.empty(B, C, 1, 1, device=x.device, dtype=x.dtype)
    grid = lambda meta: (B, triton.cdiv(C, meta['BLOCK_C']))
    adaptive_avg_pool2d_kernel[grid](
        x, out,
        B, C, HW,
    )
    return out


def triton_linear(x, weight, bias):
    """x: [M, K], weight: [N, K], bias: [N] -> [M, N]"""
    M, K = x.shape
    N = weight.shape[0]
    # weight is [N, K], we need [K, N] for GEMM
    w_t = weight.t().contiguous()
    out = torch.empty(M, N, device=x.device, dtype=x.dtype)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    linear_kernel[grid](
        x, w_t, bias, out,
        M, N, K,
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
    # conv2d: [2,3,224,224] -> [2,32