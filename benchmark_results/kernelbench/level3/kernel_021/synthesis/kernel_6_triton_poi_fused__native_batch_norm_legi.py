import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
import math

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: 1x1 conv (expand) + BN + ReLU6
# Treats as GEMM: (N*H*W, Cin) x (Cin, Cout) -> (N*H*W, Cout)
# Then fuse BN + clamp in epilogue
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bn_relu6_kernel(
    X_ptr, W_ptr, Out_ptr,
    BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    eps: tl.constexpr,
    APPLY_RELU6: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        # Load X tile [BLOCK_M, BLOCK_K]
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

    # BN + ReLU6 epilogue
    bn_mean = tl.load(BN_mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    bn_var = tl.load(BN_var_ptr + offs_n, mask=offs_n < N, other=1.0)
    bn_w = tl.load(BN_weight_ptr + offs_n, mask=offs_n < N, other=1.0)
    bn_b = tl.load(BN_bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    out = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]

    if APPLY_RELU6:
        out = tl.maximum(out, 0.0)
        out = tl.minimum(out, 6.0)

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out.to(tl.float32), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Depthwise conv 5x5 stride=2 pad=2 + BN + ReLU6
# Each program handles one (batch, channel, out_h, out_w) output element
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_HW': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_C': 256, 'BLOCK_HW': 4}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 64}, num_stages=2, num_warps=4),
    ],
    key=['C', 'OUT_H', 'OUT_W', 'IN_H', 'IN_W'],
)
@triton.jit
def depthwise_conv5x5_bn_relu6_kernel(
    X_ptr, W_ptr, Out_ptr,
    BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wc, stride_wkh, stride_wkw,
    stride_on, stride_oc, stride_oh, stride_ow,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    pid_n = tl.program_id(2)
    pid_c = tl.program_id(1)
    pid_hw = tl.program_id(0)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    out_h = offs_hw // OUT_W
    out_w = offs_hw % OUT_W

    acc = tl.zeros((BLOCK_HW, BLOCK_C), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h * STRIDE_H + kh - PAD_H
            iw = out_w * STRIDE_W + kw - PAD_W

            valid_h = (ih >= 0) & (ih < IN_H)
            valid_w = (iw >= 0) & (iw < IN_W)
            valid_hw = valid_h & valid_w

            # Load input [BLOCK_HW, BLOCK_C]
            x_ptrs = (X_ptr
            x_mask = (valid_hw[:, None]) & (offs_c[None, :] < C)
            x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

            # Load weight [BLOCK_C] (depthwise: one filter per channel)
            w_ptrs = W_ptr + offs_c * stride_wc + kh * stride_wkh + kw * stride_wkw
            w_mask = offs_c < C
            w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

            acc += x_tile * w_tile[None, :]

    # BN + ReLU6
    bn_mean = tl.load(BN_mean_ptr + offs_c, mask=offs_c < C, other=0.0)
    bn_var = tl.load(BN_var_ptr + offs_c, mask=offs_c < C, other=1.0)
    bn_w = tl.load(BN_weight_ptr + offs_c, mask=offs_c < C, other=1.0)
    bn_b = tl.load(BN_bias_ptr + offs_c, mask=offs_c < C, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    out = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]
    out = tl.maximum(out, 0.0)
    out = tl.minimum(out, 6.0)

    # Store [BLOCK_HW, BLOCK_C]
    out_ptrs = (Out_ptr
    out_mask = (offs_hw[:, None] < OUT_H * OUT_W) & (offs_c[None, :] < C)
    tl.store(out_ptrs, out.to(tl.float32), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: 1x1 project conv + BN (no ReLU6)
# Same GEMM structure as kernel 1
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bn_kernel(
    X_ptr, W_ptr, Out_ptr,
    BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_om, stride_on,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = W_ptr + k_offs[:, None] * stride_wk + offs_n[None, :] * stride_wn
        w_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

    bn_mean = tl.load(BN_mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    bn_var = tl.load(BN_var_ptr + offs_n, mask=offs_n < N, other=1.0)
    bn_w = tl.load(BN_weight_ptr + offs_n, mask=offs_n < N, other=1.0)
    bn_b = tl.load(BN_bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    out = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, out.to(tl.float32), mask=out_mask)


def kernel_function(
):
    N, Cin, H, W = x.shape  # 10, 112, 224, 224
    hidden_dim = 672
    out_channels = 192
    eps = 1e-5

    # ── Step 1: Expand 1x1 conv + BN + ReLU6 ──────────────────────────────
    # x: [N, 112, H, W] -> [N, 672, H, W]
    # Treat as GEMM: (N*H*W, 112) x (112, 672) -> (N*H*W, 672)
    # x is [N, Cin, H, W] in NCHW. We need (N*H*W, Cin).
    # Reshape x to [N*H*W, Cin] with strides.
    # x.stride() = [Cin*H*W, H*W, W, 1]
    # For row i = (n, h, w): x[n, :, h, w] -> x_ptr + n*Cin*H*W + h*W + w + c*H*W
    # This is NOT contiguous in the M dimension for NCHW layout.
    # Better: permute x to [N, H, W, Cin] (NHWC) then reshape to [N*H*W, Cin].

    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H, W, Cin]
    M1 = N * H * W  # 501760
    K1 = Cin        # 112
    N1 = hidden_dim # 672

    # Weight: [672, 112, 1, 1] -> [112, 672] (K x N)
    W1 = p_expand_conv_0_weight.view(N1, K1).t().contiguous()  # [112, 672]

    expand_out = torch.empty((M1, N1), device=x.device, dtype=torch.float32)

    grid1 = lambda meta: (
        triton.cdiv(M1, meta['BLOCK_M']),
        triton.cdiv(N1, meta['BLOCK_N']),
    )
    conv1x1_bn_relu6_kernel[grid1](
        x_nhwc.view(M1, K1), W1, expand_out,
        b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
        p_expand_conv_1_weight, p_expand_conv_1_bias,
        M1, N1, K1,
        eps,
    )
    # expand_out: [M1, N1] = [N*H*W, 672]
    # Reshape to [N, H, W, 672] then permute to [N, 672, H, W]
    expand_out_nchw = expand_out.view(N, H, W, hidden_dim).permute(0, 3, 1, 2).contiguous()
    # expand_out_nchw: [10, 672, 224, 224]

    # ── Step 2: Depthwise conv 5x5 stride=2 pad=2 + BN + ReLU6 ───────────
    # Input: [N, 672, 224, 224], Output: [N, 672, 112, 112]
    KH, KW = 5, 5
    STRIDE_H, STRIDE_W = 2, 2
    PAD_H, PAD_W = 2, 2
    OUT_H = (H + 2 * PAD_H - KH) // STRIDE_H + 1  # 112
    OUT_W = (W + 2 * PAD_W - KW) // STRIDE_W + 1  # 112

    dw_out = torch.empty((N, hidden_dim, OUT_H, OUT_W), device=x.device, dtype=torch.float32)

    # Weight: [672, 1, 5, 5] -> strides for [C, KH, KW]
    dw_weight = p_depthwise_conv_0_weight.view(hidden_dim, KH, KW)

    BLOCK_C = 64
    BLOCK_HW = 16
    # We'll use the autotune kernel
    grid2 = lambda meta: (
        triton.cdiv(OUT_H * OUT_W, meta['BLOCK_HW']),
        triton.cdiv(hidden_dim, meta['BLOCK_C']),
        N,
    )
    depthwise_conv5x5_bn_relu6_kernel[grid2](
        expand_out_nchw, dw_weight, dw_out,
        b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
        p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
        N, hidden_dim, H, W, OUT_H, OUT_W,
        expand_out_nchw.stride(0), expand_out_nchw.stride(1),
        expand_out_nchw.stride(2), expand_out_nchw.stride(3),
        dw_weight.stride(0), dw_weight.stride(1), dw_weight.stride(2),
        dw_out.stride(0), dw_out.stride(1), dw_out.stride(2), dw_out.stride(3),
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        KH, KW,
        eps,
    )
    # dw_out: [N, 672, 112, 112]

    # ── Step 3: Project 1x1 conv + BN (no ReLU6) ──────────────────────────
    # Input: [N, 672, 112, 112] -> [N, 192, 112, 112]
    M3 = N * OUT_H * OUT_W  # 10 * 112 * 112 = 125440
    K3 = hidden_dim          # 672
    N3 = out_channels        # 192

    # Permute dw_out to NHWC for contiguous M dimension
    dw_out_nhwc = dw_out.permute(0, 2, 3, 1).contiguous()  # [N, 112, 112, 672]

    # Weight: [192, 672, 1, 1] -> [672, 192]
    W3 = p_project_conv_0_weight.view(N3, K3).t().contiguous()  # [672, 192]

    proj_out = torch.empty((M3, N3), device=x.device, dtype=torch.float32)

    grid3 = lambda meta: (
        triton.cdiv(M3, meta['BLOCK_M']),
        triton.cdiv(N3, meta['BLOCK_N']),
    )
    conv1x1_bn_kernel[grid3](
        dw_out_nhwc.view(M3, K3), W3, proj_out,
        b_project_conv_1_running_mean, b_project_conv_1_running_var,
        p_project_conv_1_weight, p_project_conv_1_bias,
        M3, N3, K3,
        K3, 1,
        N3, 1,
        N3, 1,
        eps,
    )
    # proj_out: [M3, N3] = [N*112*112, 192]
    result = proj_out.view(N, OUT_H, OUT_W, out_channels).permute(0, 3, 1, 2).contiguous()
    return result


def get_inputs():
    torch.manual_seed(42)
    p_expand_conv_0_weight = torch.randn(672, 112, 1, 1)
    p_expand_conv_1_weight = torch.ones(672)
    p_expand_conv_1_bias = torch.zeros(672)
    p_depthwise_conv_0_weight = torch.randn(672, 1, 5, 5)
    p_depthwise_conv_1_weight = torch.ones(672)
    p_depthwise_conv_1_bias = torch.zeros(672)
    p_project_conv_0_weight = torch.randn(192, 672, 1, 1)
    p_project_conv_1_weight = torch.ones(192)
    p_project_conv_1_bias = torch.zeros(192)
    b_expand_conv_1_running_mean = torch.zeros(672)
    b_expand_conv_1_running_var = torch.ones(672)
    b_depthwise_conv_1_running_mean = torch.zeros(672)
    b_depthwise_conv_1_running_var = torch.ones(672)
    b_project_conv_1_running_mean = torch.zeros(192)
    b_project_conv_1_running_var = torch.ones(192)
    x = torch.rand(10, 112, 224, 224)
    return [
        p_expand_conv_0_weight, p_expand_conv_1_weight, p_expand_conv_1_bias,
        p_depthwise_conv_0_weight, p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
        p_project_conv_0_weight, p_project_conv_1_weight, p_project_conv_1_bias,
        b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
        b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
        b_project_conv_1_running_mean, b_project_conv_1_running_var,
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
        import torch.nn.functional as F

        # Reference implementation
        (p_expand_conv_0_weight, p_expand_conv_1_weight, p_expand_conv_1_bias,
         p_depthwise_conv_0_weight, p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
         p_project_conv_0_weight, p_project_conv_1_weight, p_project_conv_1_bias,
         b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
         b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
         b_project_conv_1_running_mean, b_project_conv_1_running_var,
         x) = inputs_cuda

        # Step 1: expand conv + BN + ReLU6
        out1 = F.conv2d(x, p_expand_conv_0_weight, bias=None, stride=1, padding=0)
        out1 = F.batch_norm(out1, b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
                            p_expand_conv_1_weight, p_expand_conv_1_bias, training=False, eps=1e-5)
        out1 = out1.clamp(0.0, 6.0)

        # Step 2: depthwise conv + BN + ReLU6
        out2 = F.conv2d(out1, p_depthwise_conv_0_weight, bias=None, stride=2, padding=2,
                        groups=672)
        out2 = F.batch_norm(out2, b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
                            p_depthwise_conv_1_weight, p_depthwise_conv_1_bias, training=False, eps=1e-5)
        out2 = out2.clamp(0.0, 6.0)

        # Step 3: project conv + BN
        out3 = F.conv2d(out2, p_project_conv_0_weight, bias=None, stride=1, padding=0)
        ref = F.batch_norm(out3, b_project_conv_1_running_mean, b_