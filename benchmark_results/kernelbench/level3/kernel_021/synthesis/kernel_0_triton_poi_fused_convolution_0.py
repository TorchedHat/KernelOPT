import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

# ─── Kernel 1: 1x1 conv + BN + ReLU6 (expand: 112->672) ───────────────────────
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
    stride_wn, stride_wk,
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
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc, allow_tf32=False)

    # BN + ReLU6
    bn_mean = tl.load(BN_mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    bn_var = tl.load(BN_var_ptr + offs_n, mask=offs_n < N, other=1.0)
    bn_w = tl.load(BN_weight_ptr + offs_n, mask=offs_n < N, other=1.0)
    bn_b = tl.load(BN_bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    out = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]
    out = tl.minimum(tl.maximum(out, 0.0), 6.0)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out, mask=out_mask)


# ─── Kernel 2: Depthwise conv 5x5 stride=2 pad=2 + BN + ReLU6 ────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_P': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_P': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_P': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 128, 'BLOCK_P': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_P': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_P': 64}, num_stages=3, num_warps=8),
    ],
    key=['C', 'OUT_H', 'OUT_W', 'IN_H', 'IN_W'],
)
@triton.jit
def depthwise_conv_bn_relu6_kernel(
    X_ptr, W_ptr, Out_ptr,
    BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_P: tl.constexpr,
):
    # Each program handles BLOCK_C channels and BLOCK_P output spatial positions
    pid_c = tl.program_id(0)
    pid_p = tl.program_id(1)
    pid_n = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    out_h = offs_p // OUT_W
    out_w = offs_p % OUT_W

    c_mask = offs_c < C
    p_mask = offs_p < (OUT_H * OUT_W)

    # Load BN params
    bn_mean = tl.load(BN_mean_ptr + offs_c, mask=c_mask, other=0.0)
    bn_var = tl.load(BN_var_ptr + offs_c, mask=c_mask, other=1.0)
    bn_w = tl.load(BN_weight_ptr + offs_c, mask=c_mask, other=1.0)
    bn_b = tl.load(BN_bias_ptr + offs_c, mask=c_mask, other=0.0)
    inv_std = 1.0 / tl.sqrt(bn_var + eps)

    # acc: [BLOCK_P, BLOCK_C]
    acc = tl.zeros((BLOCK_P, BLOCK_C), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h * stride_h + kh - pad_h
            iw = out_w * stride_w + kw - pad_w

            valid_h = (ih >= 0) & (ih < IN_H)
            valid_w = (iw >= 0) & (iw < IN_W)
            valid_p = p_mask & valid_h & valid_w

            # Load weights: W[c, 0, kh, kw] -> shape [BLOCK_C]
            w_idx = offs_c * KH * KW + kh * KW + kw
            w = tl.load(W_ptr + w_idx, mask=c_mask, other=0.0)

            # Load input: X[n, c, ih, iw] -> [BLOCK_P, BLOCK_C]
            x_idx = (pid_n * C * IN_H * IN_W +
                     offs_c[None, :] * IN_H * IN_W +
                     ih[:, None] * IN_W +
                     iw[:, None])
            x = tl.load(X_ptr + x_idx,
                        mask=valid_p[:, None] & c_mask[None, :],
                        other=0.0)

            acc += x * w[None, :]

    # Apply BN + ReLU6
    out = (acc - bn_mean[None, :]) * inv_std[None, :] * bn_w[None, :] + bn_b[None, :]
    out = tl.minimum(tl.maximum(out, 0.0), 6.0)

    # Store: Out[n, c, out_h, out_w]
    out_idx = (pid_n * C * OUT_H * OUT_W +
               offs_c[None, :] * OUT_H * OUT_W +
               out_h[:, None] * OUT_W +
               out_w[:, None])
    tl.store(Out_ptr + out_idx,
             out,
             mask=valid_p[:, None] & c_mask[None, :])


# ─── Kernel 3: 1x1 conv + BN (project: 672->192, no activation) ───────────────
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
    stride_wn, stride_wk,
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
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc, allow_tf32=False)

    # BN (no activation)
    bn_mean = tl.load(BN_mean_ptr + offs_n, mask=offs_n < N, other=0.0)
    bn_var = tl.load(BN_var_ptr + offs_n, mask=offs_n < N, other=1.0)
    bn_w = tl.load(BN_weight_ptr + offs_n, mask=offs_n < N, other=1.0)
    bn_b = tl.load(BN_bias_ptr + offs_n, mask=offs_n < N, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    out = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on, out, mask=out_mask)


def kernel_function(
):
    batch, in_c, H, W = x.shape  # 10, 112, 224, 224
    expand_c = 672
    out_c = 192
    eps = 1e-5

    # ── Step 1: Expand conv 1x1 (112->672) + BN + ReLU6 ──────────────────────
    # Reshape x to [M, K] where M = batch*H*W, K = in_c
    M1 = batch * H * W  # 10 * 224 * 224 = 501760
    K1 = in_c           # 112
    N1 = expand_c       # 672

    # x is [10, 112, 224, 224] in NCHW -> need [M, K] = [501760, 112]
    # Reorder to NHWC: [10, 224, 224, 112]
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [10, 224, 224, 112]
    x_2d = x_nhwc.reshape(M1, K1)  # [501760, 112]

    # Weight: [672, 112, 1, 1] -> [672, 112] -> treat as [N, K]
    w_expand = p_expand_conv_0_weight.reshape(N1, K1)  # [672, 112]

    # Output: [M1, N1] = [501760, 672]
    out1 = torch.empty((M1, N1), device=x.device, dtype=torch.float32)

    grid1 = lambda meta: (triton.cdiv(M1, meta['BLOCK_M']), triton.cdiv(N1, meta['BLOCK_N']))
    conv1x1_bn_relu6_kernel[grid1](
        x_2d, w_expand, out1,
        b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
        p_expand_conv_1_weight, p_expand_conv_1_bias,
        M1, N1, K1,
        K1, 1,   # stride_wn, stride_wk (w is [N, K], so stride_wn=K, stride_wk=1)
        eps,
    )
    # out1 is [501760, 672] in NHWC layout -> reshape to [10, 224, 224, 672]
    # Then convert to NCHW [10, 672, 224, 224] for depthwise conv
    out1_nchw = out1.reshape(batch, H, W, expand_c).permute(0, 3, 1, 2).contiguous()

    # ── Step 2: Depthwise conv 5x5 stride=2 pad=2 + BN + ReLU6 ──────────────
    OUT_H2 = (H + 2*2 - 5) // 2 + 1  # (224 + 4 - 5) // 2 + 1 = 112
    OUT_W2 = (W + 2*2 - 5) // 2 + 1  # 112

    out2 = torch.empty((batch, expand_c, OUT_H2, OUT_W2), device=x.device, dtype=torch.float32)

    # Weight for depthwise: [672, 1, 5, 5] -> flatten to [672, 25]
    w_dw = p_depthwise_conv_0_weight.reshape(expand_c, 25)  # [672, 25]

    BLOCK_C = 32
    BLOCK_P = 64
    grid2 = lambda meta: (
        triton.cdiv(expand_c, meta['BLOCK_C']),
        triton.cdiv(OUT_H2 * OUT_W2, meta['BLOCK_P']),
        batch,
    )
    depthwise_conv_bn_relu6_kernel[grid2](
        out1_nchw, p_depthwise_conv_0_weight, out2,
        b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
        p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
        batch, expand_c, H, W, OUT_H2, OUT_W2,
        eps,
    )

    # ── Step 3: Project conv 1x1 (672->192) + BN ──────────────────────────────
    M3 = batch * OUT_H2 * OUT_W2  # 10 * 112 * 112 = 125440
    K3 = expand_c                  # 672
    N3 = out_c                     # 192

    # out2 is [10, 672, 112, 112] NCHW -> NHWC [10, 112, 112, 672] -> [M3, K3]
    out2_nhwc = out2.permute(0, 2, 3, 1).contiguous()
    out2_2d = out2_nhwc.reshape(M3, K3)

    # Weight: [192, 672, 1, 1] -> [192, 672]
    w_proj = p_project_conv_0_weight.reshape(N3, K3)

    out3 = torch.empty((M3, N3), device=x.device, dtype=torch.float32)

    grid3 = lambda meta: (triton.cdiv(M3, meta['BLOCK_M']), triton.cdiv(N3, meta['BLOCK_N']))
    conv1x1_bn_kernel[grid3](
        out2_2d, w_proj, out3,
        b_project_conv_1_running_mean, b_project_conv_1_running_var,
        p_project_conv_1_weight, p_project_conv_1_bias,
        M3, N3, K3,
        K3, 1,
        K3, 1,
        N3, 1,
        eps,
    )

    # Reshape to [10, 192, 112, 112]
    result = out3.reshape(batch, OUT_H2, OUT_W2, out_c).permute(0, 3, 1, 2).contiguous()
    return result


def get_inputs():
    batch_size = 10
    in_channels = 112
    out_channels = 192
    expand_ratio = 6
    hidden_dim = in_channels * expand_ratio  # 672

    p_expand_conv_0_weight = torch.randn(hidden_dim, in_channels, 1, 1)
    p_expand_conv_1_weight = torch.ones(hidden_dim)
    p_expand_conv_1_bias = torch.zeros(hidden_dim)
    p_depthwise_conv_0_weight = torch.randn(hidden_dim, 1, 5, 5)
    p_depthwise_conv_1_weight = torch.ones(hidden_dim)
    p_depthwise_conv_1_bias = torch.zeros(hidden_dim)
    p_project_conv_0_weight = torch.randn(out_channels, hidden_dim, 1, 1)
    p_project_conv_1_weight = torch.ones(out_channels)
    p_project_conv_1_bias = torch.zeros(out_channels)
    b_expand_conv_1_running_mean = torch.zeros(hidden_dim)
    b_expand_conv_1_running_var = torch.ones(hidden_dim)
    b_depthwise_conv_1_running_mean = torch.zeros(hidden_dim)
    b_depthwise_conv_1_running_var = torch.ones(hidden_dim)
    b_project_conv_1_running_mean = torch.zeros(out_channels)
    b_project_conv_1_running_var = torch.ones(out_channels)
    x = torch.rand(batch_size, in_channels, 224, 224)

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
    inputs = [t.cuda() for t in inputs]

    output = kernel_function(*inputs)
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

        (p_expand_conv_0_weight, p_expand_conv_1_weight, p_expand_conv_1_bias,
         p_depthwise_conv_0_weight, p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
         p_project_conv_0_weight, p_project_conv_1_weight, p_project_conv_1_bias,
         b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
         b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
         b_project_conv_1_running_mean, b_project_conv_1_running_var,
         x) = inputs

        # Expand conv
        expand_conv_out = torch.nn.functional.conv2d(x, p_expand_conv_0_weight, bias=None, stride=1, padding=0)
        # BN
        expand_bn_out = torch.nn.functional.batch_norm(
            expand_conv_out,
            b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
            p_expand_conv_1_weight, p_expand_conv_1_bias,
            training=False, eps=1e-5
        )
        # ReLU6
        expand_act = torch.clamp(expand_bn_out, 0.0, 6.0)

        # Depthwise conv
        dw_conv_out = torch.nn.functional.conv2d(
            expand_act, p_depthwise_conv_0_weight, bias=None,
            stride=2, padding=2, groups=hidden_dim
        )
        # BN
        dw_bn_out = torch.nn.functional.batch_norm(
            dw_conv_out,
            b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
            p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
            training=False, eps=1e-5
        )
        # ReLU6
        dw_act = torch.clamp(dw_bn_out, 0.0, 6.0)

        # Project conv
        proj_conv_out = torch.nn.functional.conv2d(dw_act, p_project_conv_0_weight, bias=None, stride=1, padding=0)
        # BN
        ref = torch.nn.functional.batch_norm(
            proj_conv_out,
            b_project_conv_1_running_mean, b_project_conv_1_running_var,
            p_project_conv_1_weight, p_project_conv_1_bias,
            training=False, eps=1e-5
        )

        print(f"Output shape: {output.shape}, Ref shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")