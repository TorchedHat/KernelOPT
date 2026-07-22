import torch
import triton
import triton.language as tl


import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

# ─────────────────────────────────────────────────────────────────────────────
# Fused BN + ReLU pointwise kernel
# ─────────────────────────────────────────────────────────────────────────────
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
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # channel index: layout is (N, C, H, W) contiguous -> channel = (offset // HW) % C
    c_idx = (offsets // HW) % C

    mean = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var  = tl.load(var_ptr  + c_idx, mask=mask, other=1.0)
    w    = tl.load(weight_ptr + c_idx, mask=mask, other=1.0)
    b    = tl.load(bias_ptr   + c_idx, mask=mask, other=0.0)

    x_norm = (x - mean) / libdevice.sqrt(var + eps)
    out = x_norm * w + b
    out = tl.where(out > 0.0, out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# General implicit-GEMM convolution kernel (NHWC layout internally)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    # tensor dims
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    GROUPS,
    # strides for X (NHWC)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (Co, Ci, KH, KW) -> stored as (Co, KH, KW, Ci) for NHWC
    sw_o, sw_kh, sw_kw, sw_i,
    # strides for Out (NHWC)
    so_n, so_h, so_w, so_c,
    # problem size for autotune key
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    # group info
    GROUP_IN_C  = IN_C  // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
    group = offs_n // GROUP_OUT_C  # which group each output channel belongs to
    local_oc = offs_n % GROUP_OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = GROUP_IN_C * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < TOTAL_K

        # decode K -> (in_c_local, kh, kw)
        in_c_local = offs_k % GROUP_IN_C
        kh_idx = (offs_k // GROUP_IN_C) % KH
        kw_idx = offs_k // (GROUP_IN_C * KH)

        # input spatial coords
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        # global in_c
        in_c_global = in_c_local  # group offset added via batch_idx group

        valid_m = (batch_idx < BATCH) & (out_h_idx < OUT_H) & (out_w_idx < OUT_W)
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_x = valid_m[:, None] & valid_ih & valid_iw & k_mask[None, :]

        # For grouped conv: in_c_global = group * GROUP_IN_C + in_c_local
        # group for each M element: group_m = offs_n // GROUP_OUT_C -> but M and N are independent
        # For depthwise (groups=IN_C), each output channel c uses input channel c
        # We handle this by: for each (m, k) pair, the group is determined by the output channel
        # But M and N are tiled independently... this is tricky for grouped conv.
        # For simplicity, handle groups=1 and groups=IN_C (depthwise) cases.
        # For depthwise: GROUP_IN_C=1, so in_c_local=0 always, and in_c_global = oc (the output channel)
        # We'll compute x_ptr with group offset based on N tile's group
        # This only works correctly when BLOCK_N fits within one group or groups=1
        # For depthwise conv, GROUP_OUT_C=1, so each N tile has exactly one group.

        # Compute group for each output channel in offs_n
        grp_n = offs_n // GROUP_OUT_C  # shape [BLOCK_N]
        # in_c_global for each (m, k, n) = grp_n[n] * GROUP_IN_C + in_c_local[k]
        # X index: batch_idx[m] * sx_n + ih[m,k] * sx_h + iw[m,k] * sx_w + (grp_n[n]*GROUP_IN_C + in_c_local[k]) * sx_c
        # This requires a 3D indexing which tl.dot can't handle directly.
        # For groups=1: grp_n=0 always, so in_c_global = in_c_local
        # For depthwise (GROUP_IN_C=1): in_c_local=0, in_c_global = grp_n[n]
        # We need separate handling. Let's use a workaround:
        # Load X with in_c = in_c_local (for groups=1 this is correct)
        # For depthwise, we need per-N-element channel offset which breaks tl.dot.
        # Solution: for depthwise, BLOCK_N=1 effectively (or handle separately).
        # Here we implement the general case assuming groups=1 or depthwise with BLOCK_N tile within one group.

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: (OUT_C, GROUP_IN_C, KH, KW) in OIHW, stored as sw_o, sw_i, sw_kh, sw_kw
        # For depthwise: each oc uses in_c=oc, kh, kw
        # w_ptrs shape: [BLOCK_K, BLOCK_N]
        w_ptrs = (W_ptr
        valid_w = k_mask[:, None] & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output (NHWC)
    valid_out = ((batch_idx < BATCH) & (out_h_idx < OUT_H) & (out_w_idx < OUT_W))[:, None] & (offs_n[None, :] < OUT_C)
    out_ptrs = (Out_ptr
    tl.store(out_ptrs, acc.to(tl.float32), mask=valid_out)


# ─────────────────────────────────────────────────────────────────────────────
# Linear kernel (GEMM): out[M, N] = x[M, K] @ W[N, K]^T + bias[N]
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=2),
    ],
    key=['M_lin', 'N_lin', 'K_lin'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M_lin, N_lin, K_lin,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K_lin, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K_lin

        x_ptrs = X_ptr + offs_m[:, None] * K_lin + offs_k[None, :]
        x_mask = (offs_m[:, None] < M_lin) & k_mask[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # W is (N, K), we load W[offs_n, offs_k]
        w_ptrs = W_ptr + offs_n[None, :] * K_lin + offs_k[:, None]
        w_mask = (offs_n[None, :] < N_lin) & k_mask[:, None]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N_lin, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = Out_ptr + offs_m[:, None] * N_lin + offs_n[None, :]
    out_mask = (offs_m[:, None] < M_lin) & (offs_n[None, :] < N_lin)
    tl.store(out_ptrs, acc, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Avg pool 2D kernel (global average pool for 7x7 -> 1x1)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 64}, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_warps=8),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 64}, num_warps=2),
    ],
    key=['N', 'C', 'HW', 'KH', 'KW'],
)
@triton.jit
def avg_pool2d_kernel(
    X_ptr, Out_ptr,
    N, C, H, W,
    KH, KW,
    OUT_H, OUT_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides (NCHW)
    sx_n, sx_c, sx_h, sx_w,
    so_n, so_c, so_h, so_w,
    HW,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    out_w_idx = offs_hw % OUT_W
    out_h_idx = offs_hw // OUT_W

    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)
    count = KH * KW

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h_idx[None, :] * STRIDE_H + kh - PAD_H
            iw = out_w_idx[None, :] * STRIDE_W + kw - PAD_W

            valid = ((offs_c[:, None] < C)
                     & (ih >= 0) & (ih < H)
                     & (iw >= 0) & (iw < W))

            x_ptrs = (X_ptr
            x_val = tl.load(x_ptrs, mask=valid, other=0.0)
            acc = acc + x_val

    acc = acc / count

    out_hw = out_h_idx[None, :] * OUT_W + out_w_idx[None, :]
    out_ptrs = (Out_ptr
    # Fix: use proper strides
    out_ptrs2 = (Out_ptr
    out_mask = (offs_c[:, None] < C) & (offs_hw[None, :] < HW)
    tl.store(out_ptrs2, acc, mask=out_mask)


def run_conv2d(x_nhwc, w, OUT_H, OUT_W, stride_h, stride_w, pad_h, pad_w, groups):
    """Run implicit GEMM conv2d. x_nhwc: (N,H,W,C), w: (Co,Ci,KH,KW)"""
    N, H, W, IN_C = x_nhwc.shape
    OUT_C, _, KH, KW = w.shape

    M = N * OUT_H * OUT_W
    K_total = (IN_C // groups) * KH * KW

    out = torch.empty((N, OUT_H, OUT_W, OUT_C), device=x_nhwc.device, dtype=torch.float32)

    # strides for x_nhwc (N, H, W, C)
    sx_n, sx_h, sx_w, sx_c = x_nhwc.stride()
    # strides for w (Co, Ci, KH, KW)
    sw_o, sw_i, sw_kh, sw_kw = w.stride()
    # strides for out (N, H, W, C)
    so_n, so_h, so_w, so_c = out.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w, out,
        N, IN_C, H, W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        groups,
        sx_n, sx_h, sx_w, sx_c,
        sw_o, sw_kh, sw_kw, sw_i,
        so_n, so_h, so_w, so_c,
        M, OUT_C, K_total,
    )
    return out  # (N, OUT_H, OUT_W, OUT_C)


def run_bn_relu(x_nchw, mean, var, weight, bias, eps=1e-5):
    """Fused BN+ReLU on NCHW tensor."""
    N, C, H, W = x_nchw.shape
    numel = N * C * H * W
    HW = H * W
    out = torch.empty_like(x_nchw)

    def grid(meta):
        return (triton.cdiv(numel, meta['BLOCK_SIZE']),)

    bn_relu_kernel[grid](
        x_nchw, out,
        mean, var, weight, bias,
        numel, C, HW,
        eps,
    )
    return out


def run_avg_pool2d(x, kh, kw, stride_h, stride_w, pad_h, pad_w):
    """Average pooling on NCHW tensor."""
    N, C, H, W = x.shape
    OUT_H = (H + 2 * pad_h - kh) // stride_h + 1
    OUT_W = (W + 2 * pad_w - kw) // stride_w + 1
    HW_out = OUT_H * OUT_W

    out = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    sx_n, sx_c, sx_h, sx_w = x.stride()
    so_n, so_c, so_h, so_w = out.stride()

    def grid(meta):
        return (
            N,
            triton.cdiv(C, meta['BLOCK_C']),
            triton.cdiv(HW_out, meta['BLOCK_HW']),
        )

    avg_pool2d_kernel[grid](
        x, out,
        N, C, H, W,
        kh, kw,
        OUT_H, OUT_W,
        stride_h, stride_w,
        pad_h, pad_w,
        sx_n, sx_c, sx_h, sx_w,
        so_n, so_c, so_h, so_w,
        HW_out,
    )
    return out


def run_linear(x, weight, bias):
    """Linear layer: x @ weight.T + bias"""
    M_lin = x.shape[0]
    N_lin = weight.shape[0]
    K_lin = weight.shape[1]

    out = torch.empty((M_lin, N_lin), device=x.device, dtype=torch.float32)

    def grid(meta):
        return (
            triton.cdiv(M_lin, meta['BLOCK_M']),
            triton.cdiv(N_lin, meta['BLOCK_N']),
        )

    linear_kernel[grid](
        x, weight, bias, out,
        M_lin, N_lin, K_lin,
    )
    return out


def conv_bn_relu(x_nhwc, w, bn_mean, bn_var, bn_weight, bn_bias,
    """Conv2d (implicit GEMM) + BN + ReLU. x_nhwc: NHWC, w: OIHW. Returns NHWC."""
    # Run conv (output is NHWC)
    out_nhwc = run_conv2d(x_nhwc, w, OUT_H, OUT_W, stride_h, stride_w, pad_h, pad_w, groups)
    # Convert to NCHW for BN
    out_nchw = out_nhwc.permute(0, 3, 1, 2).contiguous()
    # BN + ReLU
    out_nchw = run_bn_relu(out_nchw, bn_mean, bn_var, bn_weight, bn_bias)
    # Back to NHWC
    return out_nchw.permute(0, 2, 3, 1).contiguous()


def kernel_function(
    p_model_0_0_weight, p_model_0_1_weight, p_model_0_1_bias,
    p_model_1_0_weight, p_model_1_1_weight, p_model_1_1_bias,
    p_model_1_3_weight, p_model_1_4_weight, p_model_1_4_bias,
    p_model_2_0_weight, p_model_2_1_weight, p_model_2_1_bias,
    p_model_2_3_weight, p_model_2_4_weight, p_model_2_4_bias,
    p_model_3_0_weight, p_model_3_1_weight, p_model_3_1_bias,
    p_model_3_3_weight, p_model_3_4_weight, p_model_3_4_bias,
    p_model_4_0_weight, p_model_4_1_weight, p_model_4_1_bias,
    p_model_4_3_weight, p_model_4_4_weight, p_model_4_4_bias,
    p_model_5_0_weight, p_model_5_1_weight, p_model_5_1_bias,
    p_model_5_3_weight, p_model_5_4_weight, p_model_5_4_bias,
    p_model_6_0_weight, p_model_6_1_weight, p_model_6_1_bias,
    p_model_6_3_weight, p_model_6_4_weight, p_model_6_4_bias,
    p_model_7_0_weight, p_model_7_1_weight, p_model_7_1_bias,
    p_model_7_3_weight, p_model_7_4_weight, p_model_7_4_bias,
    p_model_8_0_weight, p_model_8_1_weight, p_model_8_1_bias,
    p_model_8_3_weight, p_model_8_4_weight, p_model_8_4_bias,
    p_model_9_0_weight, p_model_9_1_weight, p_model_9_1_bias,
    p_model_9_3_weight, p_model_9_4_weight, p_model_9_4_bias,
    p_model_10_0_weight, p_model_10_1_weight, p_model_10_1_bias,
    p_model_10_3_weight, p_model_10_4_weight, p_model_10_4_bias,
    p_model_11_0_weight, p_model_11_1_weight, p_model_11_1_bias,
    p_model_11_3_weight, p_model_11_4_weight, p_model_11_4_bias,
    p_model_12_0_weight, p_model_12_1_weight, p_model_12_1_bias,
    p_model_12_3_weight, p_model_12_4_weight, p_model_12_4_bias,
    p_model_13_0_weight, p_model_13_1_weight, p_model_13_1_bias,
    p_model_13_3_weight, p_model_13_4_weight, p_model_13_4_bias,
    p_fc_weight, p_fc_bias,
    b_model_0_1_running_mean, b_model_0_1_running_var,
    b_model_1_1_running_mean, b_model_1_1_running_var,
    b_model_1_4_running_mean, b_model_1_4_running_var,
    b_model_2_1_running_mean, b_model_2_1_running_var,
    b_model_2_4_running_mean, b_model_2_4_running_var,
    b_model_3_1_running_mean, b_model_3_1_running_var,
    b_model_3_4_running_mean, b_model_3_4_running_var,
    b_model_4_1_running_mean, b_model_4_1_running_var,
    b_model_4_4_running_mean, b_model_4_4_running_var,
    b_model_5_1_running_mean, b_model_5_1_running_var,
    b_model_5_4_running_mean, b_model_5_4_running_var,
    b_model_6_1_running_mean, b_model_6_1_running_var,