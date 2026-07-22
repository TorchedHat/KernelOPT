import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Conv2d (3x3, pad=1) + BN + ReLU  (fused)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'IN_H', 'IN_W', 'KH', 'KW', 'STRIDE_H', 'STRIDE_W', 'PAD_H', 'PAD_W'],
)
@triton.jit
def conv_bn_relu_kernel(
    X_ptr, W_ptr,
    BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, OUT_C: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # X is NHWC: shape [BATCH, IN_H, IN_W, IN_C]
    # stride: [IN_H*IN_W*IN_C, IN_W*IN_C, IN_C, 1]
    s_xn = IN_H * IN_W * IN_C
    s_xh = IN_W * IN_C
    s_xw = IN_C
    # s_xc = 1

    # W is [OUT_C, IN_C, KH, KW] -> we access as [KH, KW, IN_C, OUT_C] logically
    # W stored as [OUT_C, KH, KW, IN_C] after reshape? No, keep original [OUT_C, IN_C, KH, KW]
    # stride: [IN_C*KH*KW, KH*KW, KW, 1]
    s_woc = IN_C * KH * KW
    s_wic = KH * KW
    s_wkh = KW
    # s_wkw = 1

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        ic_idx = tmp_k // KH

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            (offs_m[:, None] < M_TOTAL) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * s_xn +
                  ic_idx[None, :])
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # W: [OUT_C, IN_C, KH, KW], load tile [BLOCK_K, BLOCK_N]
        # offs_k -> (ic, kh, kw), offs_n -> out_c
        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_woc +
                  ic_idx[:, None] * s_wic +
                  kh_idx[:, None] * s_wkh +
                  kw_idx[:, None])
        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Apply BN: (acc - mean) / sqrt(var + eps) * weight + bias
    mask_n = offs_n < OUT_C
    bn_mean = tl.load(BN_mean_ptr + offs_n, mask=mask_n, other=0.0)
    bn_var  = tl.load(BN_var_ptr  + offs_n, mask=mask_n, other=1.0)
    bn_w    = tl.load(BN_weight_ptr + offs_n, mask=mask_n, other=1.0)
    bn_b    = tl.load(BN_bias_ptr   + offs_n, mask=mask_n, other=0.0)

    eps = 1e-5
    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    acc = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]

    if DO_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store: Y is NHWC [BATCH, OUT_H, OUT_W, OUT_C]
    s_yn = OUT_H * OUT_W * OUT_C
    s_yh = OUT_W * OUT_C
    s_yw = OUT_C

    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * s_yn +
              out_h_idx[:, None] * s_yh +
              out_w_idx[:, None] * s_yw +
              offs_n[None, :])
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Conv2d (3x3, pad=1) + BN  (no relu, result added later)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'IN_H', 'IN_W', 'KH', 'KW', 'STRIDE_H', 'STRIDE_W', 'PAD_H', 'PAD_W'],
)
@triton.jit
def conv_bn_kernel(
    X_ptr, W_ptr,
    BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, OUT_C: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    s_xn = IN_H * IN_W * IN_C
    s_xh = IN_W * IN_C
    s_xw = IN_C

    s_woc = IN_C * KH * KW
    s_wic = KH * KW
    s_wkh = KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        ic_idx = tmp_k // KH

        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            (offs_m[:, None] < M_TOTAL) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * s_xn +
                  ic_idx[None, :])
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_woc +
                  ic_idx[:, None] * s_wic +
                  kh_idx[:, None] * s_wkh +
                  kw_idx[:, None])
        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    mask_n = offs_n < OUT_C
    bn_mean = tl.load(BN_mean_ptr + offs_n, mask=mask_n, other=0.0)
    bn_var  = tl.load(BN_var_ptr  + offs_n, mask=mask_n, other=1.0)
    bn_w    = tl.load(BN_weight_ptr + offs_n, mask=mask_n, other=1.0)
    bn_b    = tl.load(BN_bias_ptr   + offs_n, mask=mask_n, other=0.0)

    eps = 1e-5
    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    acc = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]

    s_yn = OUT_H * OUT_W * OUT_C
    s_yh = OUT_W * OUT_C
    s_yw = OUT_C

    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr +
              batch_idx[:, None] * s_yn +
              out_h_idx[:, None] * s_yh +
              out_w_idx[:, None] * s_yw +
              offs_n[None, :])
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Add + ReLU (pointwise, fused)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['N'],
)
@triton.jit
def add_relu_kernel(
    A_ptr, B_ptr, C_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    c = tl.maximum(a + b, tl.zeros((BLOCK_SIZE,), dtype=tl.float32))
    tl.store(C_ptr + offs, c, mask=mask)


def cdiv(a, b):
    return (a + b - 1) // b


def launch_conv_bn_relu(x_nhwc, w, bn_mean, bn_var, bn_weight, bn_bias,
    batch = x_nhwc.shape[0]
    in_h  = x_nhwc.shape[1]
    in_w  = x_nhwc.shape[2]
    in_c  = x_nhwc.shape[3]
    out_c = w.shape[0]
    out_h = (in_h + 2 * pad_h - kh) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kw) // stride_w + 1

    y = torch.empty((batch, out_h, out_w, out_c), device=x_nhwc.device, dtype=torch.float32)

    M = batch * out_h * out_w
    grid = (cdiv(M, 128), cdiv(out_c, 64))  # will be overridden by autotune

    # Use autotune-friendly launch
    conv_bn_relu_kernel[lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(out_c, meta['BLOCK_N']))](
        x_nhwc, w,
        bn_mean, bn_var, bn_weight, bn_bias,
        y,
        batch, in_c, out_c,
        in_h, in_w,
        out_h, out_w,
        kh, kw,
        stride_h, stride_w,
        pad_h, pad_w,
        do_relu,
    )
    return y


def launch_conv_bn(x_nhwc, w, bn_mean, bn_var, bn_weight, bn_bias,
    batch = x_nhwc.shape[0]
    in_h  = x_nhwc.shape[1]
    in_w  = x_nhwc.shape[2]
    in_c  = x_nhwc.shape[3]
    out_c = w.shape[0]
    out_h = (in_h + 2 * pad_h - kh) // stride_h + 1
    out_w = (in_w + 2 * pad_w - kw) // stride_w + 1

    y = torch.empty((batch, out_h, out_w, out_c), device=x_nhwc.device, dtype=torch.float32)

    M = batch * out_h * out_w

    conv_bn_kernel[lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(out_c, meta['BLOCK_N']))](
        x_nhwc, w,
        bn_mean, bn_var, bn_weight, bn_bias,
        y,
        batch, in_c, out_c,
        in_h, in_w,
        out_h, out_w,
        kh, kw,
        stride_h, stride_w,
        pad_h, pad_w,
    )
    return y


def kernel_function(
    p_conv1_weight,
    p_bn1_weight,
    p_bn1_bias,
    p_conv2_weight,
    p_bn2_weight,
    p_bn2_bias,
    p_downsample_0_weight,
    p_downsample_1_weight,
    p_downsample_1_bias,
    b_bn1_running_mean,
    b_bn1_running_var,
    b_bn2_running_mean,
    b_bn2_running_var,
    b_downsample_1_running_mean,
    b_downsample_1_running_var,
    x,
):
    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # ── Step 1: conv1 (3x3, pad=1) + bn1 + relu ──────────────────────────────
    out1 = launch_conv_bn_relu(
        x_nhwc, p_conv1_weight,
        b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias,
        3, 3, 1, 1, 1, 1, True,
    )

    # ── Step 2: conv2 (3x3, pad=1) + bn2 (no relu yet) ───────────────────────
    out2 = launch_conv_bn(
        out1, p_conv2_weight,
        b_bn2_running_mean, b_bn2_running_var, p_bn2_weight, p_bn2_bias,
        3, 3, 1, 1, 1, 1,
    )

    # ── Step 3: downsample conv (1x1, no pad) + bn_ds ─────────────────────────
    out_ds = launch_conv_bn(
        x_nhwc, p_downsample_0_weight,
        b_downsample_1_running_mean, b_downsample_1_running_var,
        p_downsample_1_weight, p_downsample_1_bias,
        1, 1, 1, 1, 0, 0,
    )

    # ── Step 4: add + relu ────────────────────────────────────────────────────
    # out2 and out_ds are both NHWC [batch, H, W, C]
    N_total = out2.numel()
    out_final = torch.empty_like(out2)

    add_relu_kernel[lambda meta: (cdiv(N_total, meta['BLOCK_SIZE']),)](
        out2, out_ds, out_final,
        N_total,
    )

    # Convert back to NCHW
    return out_final.permute(0, 3, 1, 2).contiguous()


def get_inputs():
    torch.manual_seed(0)
    p_conv1_weight       = torch.randn(64, 3, 3, 3,  device='cuda')
    p_bn1_weight         = torch.ones(64,             device='cuda')
    p_bn1_bias           = torch.zeros(64,            device='cuda')
    p_conv2_weight       = torch.randn(64, 64, 3, 3, device='cuda')
    p_bn2_weight         = torch.ones(64,             device='cuda')
    p_bn2_bias           = torch.zeros(64,            device='cuda')
    p_downsample_0_weight = torch.randn(64, 3, 1, 1, device='cuda')
    p_downsample_1_weight = torch.ones(64,            device='cuda')
    p_downsample_1_bias   = torch.zeros(64,           device='cuda')
    b_bn1_running_mean    = torch.zeros(64,           device='cuda')
    b_bn1_running_var     = torch.ones(64,            device='cuda')
    b_bn2_running_mean    = torch.zeros(64,           device='cuda')
    b_bn2_running_var     = torch.ones(64,            device='cuda')
    b_downsample_1_running_mean = torch.zeros(64,     device='cuda')
    b_downsample_1_running_var  = torch.ones(64,      device='cuda')
    x = torch.randn(10, 3, 224, 224, device='cuda')
    return [
        p_conv1_weight, p_bn1_weight, p_bn1_bias,
        p_conv2_weight, p_bn2_weight, p_bn2_bias,
        p_downsample_0_weight, p_downsample_1_weight, p_downsample_1_bias,
        b_bn1_running_mean, b_bn1_running_var,
        b_bn2_running_mean, b_bn2_running_var,
        b_downsample_1_running_mean, b_downsample_1_running_var,
        x,
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        (p_conv1_weight, p_bn1_weight, p_bn1_bias,
         p_conv2_weight, p_bn2_weight, p_bn2_bias,
         p_downsample_0_weight, p_downsample_1_weight, p_downsample_1_bias,
         b_bn1_running_mean, b_bn1_running_var,
         b_bn2_running_mean, b_bn2_running_var,
         b_downsample_1_running_mean, b_downsample_1_running_var,
         x) = inputs

        # Reference using PyTorch eager
        import torch.nn.functional as F

        def ref_forward(x):
            # conv1 + bn1 + relu
            out = F.conv2d(x, p_conv1_weight, None, stride=1, padding=1)
            out = F.batch_norm(out, b_bn1_running_mean, b_bn1_running_var,
                               p_bn1_weight, p_bn1_bias, training=False, eps=1e-5)
            out = F.relu(out)
            # conv2 + bn2
            out = F.conv2d(out, p_conv2_weight, None, stride=1, padding=1)
            out = F.batch_norm(out, b_bn2_running_mean, b_bn2_running_var,
                               p_bn2_weight, p_bn2_bias, training=False, eps=1e-5)
            # downsample
            identity = F.conv2d(x, p_downsample_0_weight, None, stride=1, padding=0)
            identity = F.batch_norm(identity, b_downsample_1_running_mean, b_downsample_1_running_var,
                                    p_downsample_1_weight, p_downsample_1_bias, training=False, eps=1e-5)
            out = out + identity
            out = F.relu(out)
            return out

        ref = ref_forward(x)
        torch.cuda.synchronize()

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")