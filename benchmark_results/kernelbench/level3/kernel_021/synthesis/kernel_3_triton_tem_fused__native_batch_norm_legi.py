import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: 1x1 conv (expand) + BN + ReLU6
# M = N*H*W = 10*224*224 = 501760, K = 112, N_out = 672
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N_OUT', 'K_IN'],
)
@triton.jit
def expand_conv_bn_relu6_kernel(
    X_ptr, W_ptr,
    BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    Y_ptr,
    M, N_OUT, K_IN,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K_IN, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_m = offs_m[:, None] < M
        mask_k = k_offs[None, :] < K_IN
        mask_n = offs_n[None, :] < N_OUT
        mask_kn = k_offs[:, None] < K_IN

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=mask_m & mask_k, other=0.0)

        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w = tl.load(w_ptrs, mask=mask_kn & mask_n, other=0.0)

        acc = tl.dot(x, w, acc, allow_tf32=False)

    # BN + ReLU6 epilogue
    bn_mean = tl.load(BN_mean_ptr + offs_n, mask=offs_n < N_OUT, other=0.0)
    bn_var  = tl.load(BN_var_ptr  + offs_n, mask=offs_n < N_OUT, other=1.0)
    bn_w    = tl.load(BN_weight_ptr + offs_n, mask=offs_n < N_OUT, other=1.0)
    bn_b    = tl.load(BN_bias_ptr   + offs_n, mask=offs_n < N_OUT, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    out = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]
    out = tl.maximum(out, 0.0)
    out = tl.minimum(out, 6.0)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, out.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Depthwise conv 5x5 stride=2 pad=2 + BN + ReLU6
# Input: (10, 672, 224, 224) in NHWC layout after expand
# Output: (10, 672, 112, 112)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_HW': 32}, num_stages=2, num_warps=8),
    ],
    key=['C', 'OUT_H', 'OUT_W', 'BATCH'],
)
@triton.jit
def depthwise_conv_bn_relu6_kernel(
    X_ptr, W_ptr,
    BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # X strides: NHWC layout
    s_xn, s_xh, s_xw, s_xc,
    # W strides: (C, 1, KH, KW) -> (C, KH, KW)
    s_wc, s_wkh, s_wkw,
    # Y strides: NHWC layout
    s_yn, s_yh, s_yw, s_yc,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles BLOCK_HW output spatial positions and BLOCK_C channels
    pid_hw = tl.program_id(0)
    pid_c  = tl.program_id(1)
    pid_n  = tl.program_id(2)

    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)
    offs_c  = pid_c  * BLOCK_C  + tl.arange(0, BLOCK_C)

    out_w_idx = offs_hw % OUT_W
    out_h_idx = offs_hw // OUT_W

    mask_hw = offs_hw < (OUT_H * OUT_W)
    mask_c  = offs_c  < C

    acc = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h_idx * STRIDE_H + kh - PAD_H
            iw = out_w_idx * STRIDE_W + kw - PAD_W

            valid_h = (ih >= 0) & (ih < IN_H)
            valid_w = (iw >= 0) & (iw < IN_W)
            valid_spatial = valid_h & valid_w

            # Load input: shape [BLOCK_HW, BLOCK_C]
            x_ptrs = (X_ptr
            x_mask = (mask_hw[:, None] & mask_c[None, :] & valid_spatial[:, None])
            x = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Load weight: shape [BLOCK_C] -> broadcast to [BLOCK_HW, BLOCK_C]
            w_ptrs = W_ptr + offs_c * s_wc + kh * s_wkh + kw * s_wkw
            w = tl.load(w_ptrs, mask=mask_c, other=0.0)

            acc += x * w[None, :]

    # BN + ReLU6
    bn_mean = tl.load(BN_mean_ptr + offs_c, mask=mask_c, other=0.0)
    bn_var  = tl.load(BN_var_ptr  + offs_c, mask=mask_c, other=1.0)
    bn_w    = tl.load(BN_weight_ptr + offs_c, mask=mask_c, other=1.0)
    bn_b    = tl.load(BN_bias_ptr   + offs_c, mask=mask_c, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    out = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]
    out = tl.maximum(out, 0.0)
    out = tl.minimum(out, 6.0)

    mask_out = mask_hw[:, None] & mask_c[None, :]
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, out.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: 1x1 project conv + BN (no activation)
# M = N*H*W = 10*112*112 = 125440, K = 672, N_out = 192
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N_OUT', 'K_IN'],
)
@triton.jit
def project_conv_bn_kernel(
    X_ptr, W_ptr,
    BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    Y_ptr,
    M, N_OUT, K_IN,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_ym, stride_yn,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K_IN, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_m = offs_m[:, None] < M
        mask_k = k_offs[None, :] < K_IN
        mask_kn = k_offs[:, None] < K_IN
        mask_n = offs_n[None, :] < N_OUT

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x = tl.load(x_ptrs, mask=mask_m & mask_k, other=0.0)

        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w = tl.load(w_ptrs, mask=mask_kn & mask_n, other=0.0)

        acc = tl.dot(x, w, acc, allow_tf32=False)

    # BN epilogue (no activation)
    bn_mean = tl.load(BN_mean_ptr + offs_n, mask=offs_n < N_OUT, other=0.0)
    bn_var  = tl.load(BN_var_ptr  + offs_n, mask=offs_n < N_OUT, other=1.0)
    bn_w    = tl.load(BN_weight_ptr + offs_n, mask=offs_n < N_OUT, other=1.0)
    bn_b    = tl.load(BN_bias_ptr   + offs_n, mask=offs_n < N_OUT, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    out = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
    y_ptrs = Y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn
    tl.store(y_ptrs, out.to(tl.float32), mask=mask_out)


def kernel_function(
):
    device = x.device
    dtype = x.dtype
    eps = 1e-5

    BATCH = x.shape[0]  # 10
    IN_C = x.shape[1]   # 112
    IN_H = x.shape[2]   # 224
    IN_W = x.shape[3]   # 224
    EXP_C = 672
    OUT_C = 192
    KH = KW = 5
    STRIDE = 2
    PAD = 2
    OUT_H = (IN_H + 2 * PAD - KH) // STRIDE + 1  # 112
    OUT_W = (IN_W + 2 * PAD - KW) // STRIDE + 1  # 112

    # ── Step 1: Expand 1x1 conv ──────────────────────────────────────────────
    # x: (10, 112, 224, 224) -> reshape to (501760, 112) in NHWC-like order
    # We need x in (M, K) = (N*H*W, C_in) layout
    # x is NCHW: (10, 112, 224, 224)
    # Reinterpret as (10, 224, 224, 112) NHWC then flatten to (501760, 112)
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (10, 224, 224, 112)
    M_expand = BATCH * IN_H * IN_W  # 501760

    # Weight: (672, 112, 1, 1) -> (672, 112) -> transpose to (112, 672) for GEMM
    W_expand = p_expand_conv_0_weight.view(EXP_C, IN_C)  # (672, 112)

    # Output of expand conv: (M_expand, EXP_C) in NHWC layout
    expand_out = torch.empty((M_expand, EXP_C), device=device, dtype=dtype)

    grid_expand = (
        triton.cdiv(M_expand, 128),
        triton.cdiv(EXP_C, 64),
    )

    expand_conv_bn_relu6_kernel[grid_expand](
        x_nhwc.view(M_expand, IN_C),  # (501760, 112)
        b_expand_conv_1_running_mean,
        b_expand_conv_1_running_var,
        p_expand_conv_1_weight,
        p_expand_conv_1_bias,
        expand_out,
        M_expand, EXP_C, IN_C,
        IN_C, 1,        # stride_wn, stride_wk  (W is (672,112), row-major: stride_wn=112, stride_wk=1)
        eps,
    )
    # expand_out: (501760, 672) in NHWC layout -> (10, 224, 224, 672)

    # ── Step 2: Depthwise conv 5x5 stride=2 pad=2 ───────────────────────────
    # Input: (10, 224, 224, 672) NHWC
    # Weight: (672, 1, 5, 5) -> (672, 5, 5)
    expand_nhwc = expand_out.view(BATCH, IN_H, IN_W, EXP_C)  # (10, 224, 224, 672)

    W_dw = p_depthwise_conv_0_weight.view(EXP_C, KH, KW)  # (672, 5, 5)

    dw_out = torch.empty((BATCH, OUT_H, OUT_W, EXP_C), device=device, dtype=dtype)

    BLOCK_C_DW = 64
    BLOCK_HW_DW = 64
    grid_dw = (
        triton.cdiv(OUT_H * OUT_W, BLOCK_HW_DW),
        triton.cdiv(EXP_C, BLOCK_C_DW),
        BATCH,
    )

    depthwise_conv_bn_relu6_kernel[grid_dw](
        expand_nhwc,
        W_dw,
        b_depthwise_conv_1_running_mean,
        b_depthwise_conv_1_running_var,
        p_depthwise_conv_1_weight,
        p_depthwise_conv_1_bias,
        dw_out,
        BATCH, EXP_C, IN_H, IN_W, OUT_H, OUT_W,
        KH, KW,
        STRIDE, STRIDE,
        PAD, PAD,
        # X strides (NHWC): n, h, w, c
        expand_nhwc.stride(0), expand_nhwc.stride(1), expand_nhwc.stride(2), expand_nhwc.stride(3),
        # W strides: (C, KH, KW)
        W_dw.stride(0), W_dw.stride(1), W_dw.stride(2),
        # Y strides (NHWC): n, h, w, c
        dw_out.stride(0), dw_out.stride(1), dw_out.stride(2), dw_out.stride(3),
        eps,
    )
    # dw_out: (10, 112, 112, 672) NHWC

    # ── Step 3: Project 1x1 conv + BN ────────────────────────────────────────
    M_proj = BATCH * OUT_H * OUT_W  # 125440
    dw_flat = dw_out.view(M_proj, EXP_C)  # (125440, 672)

    W_proj = p_project_conv_0_weight.view(OUT_C, EXP_C)  # (192, 672)

    proj_out = torch.empty((M_proj, OUT_C), device=device, dtype=dtype)

    grid_proj = (
        triton.cdiv(M_proj, 128),
        triton.cdiv(OUT_C, 64),
    )

    project_conv_bn_kernel[grid_proj](
        dw_flat,
        W_proj,
        b_project_conv_1_running_mean,
        b_project_conv_1_running_var,
        p_project_conv_1_weight,
        p_project_conv_1_bias,
        proj_out,
        M_proj, OUT_C, EXP_C,
        eps,
    )
    # proj_out: (125440, 192) -> (10, 112, 112, 192) NHWC -> (10, 192, 112, 112) NCHW
    result = proj_out.view(BATCH, OUT_H, OUT_W, OUT_C).permute(0, 3, 1, 2).contiguous()
    return result


def get_inputs():
    torch.manual_seed(42)
    p_expand_conv_0_weight = torch.randn(672, 112, 1, 1)
    p_expand_conv_1_weight = torch.randn(672)
    p_expand_conv_1_bias = torch.randn(672)
    p_depthwise_conv_0_weight = torch.randn(672, 1, 5, 5)
    p_depthwise_conv_1_weight = torch.randn(672)
    p_depthwise_conv_1_bias = torch.randn(672)
    p_project_conv_0_weight = torch.randn(192, 672, 1, 1)
    p_project_conv_1_weight = torch.randn(192)
    p_project_conv_1_bias = torch.randn(192)
    b_expand_conv_1_running_mean = torch.randn(672)
    b_expand_conv_1_running_var = torch.rand(672) + 0.1
    b_depthwise_conv_1_running_mean = torch.randn(672)
    b_depthwise_conv_1_running_var = torch.rand(672) + 0.1
    b_project_conv_1_running_mean = torch.randn(192)
    b_project_conv_1_running_var = torch.rand(192) + 0.1
    x = torch.rand(10, 112, 224, 224)
    return [
        p_expand_conv_0_weight,
        p_expand_conv_1_weight,
        p_expand_conv_1_bias,
        p_depthwise_conv_0_weight,
        p_depthwise_conv_1_weight,
        p_depthwise_conv_1_bias,
        p_project_conv_0_weight,
        p_project_conv_1_weight,
        p_project_conv_1_bias,
        b_expand_conv_1_running_mean,
        b_expand_conv_1_running_var,
        b_depthwise_conv_1_running_mean,
        b_depthwise_conv_1_running_var,
        b_project_conv_1_running_mean,
        b_project_conv_1_running_var,
        x,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    # Move to CUDA
    inputs_cuda = [t.cuda() for t in inputs]

    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        # Build reference model
        in_channels = 112
        out_channels = 192
        kernel_size = 5
        stride = 2
        expand_ratio = 6
        hidden_dim = in_channels * expand_ratio  # 672

        # Manually construct reference using the same weights
        (p_expand_conv_0_weight, p_expand_conv_1_weight, p_expand_conv_1_bias,
         p_depthwise_conv_0_weight, p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
         p_project_conv_0_weight, p_project_conv_1_weight, p_project_conv_1_bias,
         b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
         b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
         b_project_conv_1_running_mean, b_project_conv_1_running_var,
         x) = inputs_cuda

        # Expand conv + BN + ReLU6
        import torch.nn.functional as F
        out = F.conv2d(x, p_expand_conv_0_weight, bias=None, stride=1, padding=0)
        out = F.batch_norm(out, b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
                           p_expand_conv_1_weight, p_expand_conv_1_bias, training=False, eps=1e-5)
        out = out.clamp(0.0, 6.0)

        # Depthwise conv + BN + ReLU6
        out = F.conv2d(out, p_depthwise_conv_0_weight, bias=None, stride=2, padding=2,
                       groups=hidden_dim)
        out = F.batch_norm(out, b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
                           p_depthwise_conv_1_weight, p_depthwise_conv_1_bias, training=False, eps=1e-5)
        out = out.clamp(0.0, 6.0)

        # Project conv + BN
        out = F.conv2d(out, p_project_conv_0_weight, bias=None, stride=1, padding=0)
        ref = F.batch_norm(out, b_project_conv_1_running_mean, b_project_conv_1_running_var,
                           p_project_conv_1_weight, p_project_conv_1_bias, training=False, eps=1e-5)

        max_diff = (output