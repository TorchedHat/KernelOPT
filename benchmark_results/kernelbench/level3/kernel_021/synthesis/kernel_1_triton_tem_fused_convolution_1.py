import torch
import triton
import triton.language as tl

# ─── Kernel 1: 1x1 GEMM (expand conv: M=N*H*W, K=112, N=672) ───────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bn_relu6_kernel(
    A_ptr, B_ptr, Out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(a, b, acc, allow_tf32=False)

    # BN + ReLU6 epilogue
    mean = tl.load(mean_ptr + offs_n, mask=mask_n, other=0.0)
    var  = tl.load(var_ptr  + offs_n, mask=mask_n, other=0.0)
    w    = tl.load(weight_ptr + offs_n, mask=mask_n, other=1.0)
    b_   = tl.load(bias_ptr   + offs_n, mask=mask_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(var[None, :] + eps)
    out = (acc - mean[None, :]) * inv_std * w[None, :] + b_[None, :]
    out = tl.maximum(out, 0.0)
    out = tl.minimum(out, 6.0)

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


# ─── Kernel 2: Depthwise conv 5x5 stride=2 pad=2 + BN + ReLU6 ──────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 64},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 32},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 16, 'BLOCK_HW': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 32, 'BLOCK_HW': 128}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 64, 'BLOCK_HW': 64},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_C': 128,'BLOCK_HW': 32},  num_stages=2, num_warps=8),
    ],
    key=['C', 'OUT_H', 'OUT_W', 'IN_H', 'IN_W'],
)
@triton.jit
def depthwise_conv_bn_relu6_kernel(
    X_ptr, W_ptr, Out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    N, C, IN_H, IN_W, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_h: tl.constexpr, stride_w: tl.constexpr,
    pad_h: tl.constexpr, pad_w: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_HW: tl.constexpr,
):
    # Each program handles BLOCK_C channels and BLOCK_HW output spatial positions
    pid_c  = tl.program_id(0)
    pid_hw = tl.program_id(1)
    pid_n  = tl.program_id(2)

    offs_c  = pid_c  * BLOCK_C  + tl.arange(0, BLOCK_C)
    offs_hw = pid_hw * BLOCK_HW + tl.arange(0, BLOCK_HW)

    mask_c  = offs_c  < C
    mask_hw = offs_hw < (OUT_H * OUT_W)

    out_h = offs_hw // OUT_W
    out_w = offs_hw %  OUT_W

    # Load BN params
    mean = tl.load(mean_ptr   + offs_c, mask=mask_c, other=0.0)
    var  = tl.load(var_ptr    + offs_c, mask=mask_c, other=0.0)
    w    = tl.load(weight_ptr + offs_c, mask=mask_c, other=1.0)
    b_   = tl.load(bias_ptr   + offs_c, mask=mask_c, other=0.0)
    inv_std = 1.0 / tl.sqrt(var + eps)

    # acc shape: [BLOCK_C, BLOCK_HW]
    acc = tl.zeros((BLOCK_C, BLOCK_HW), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h[None, :] * stride_h + kh - pad_h
            iw = out_w[None, :] * stride_w + kw - pad_w

            valid = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
            valid = valid & mask_hw[None, :] & mask_c[:, None]

            x_idx = pid_n * C * IN_H * IN_W + offs_c[:, None] * IN_H * IN_W + ih * IN_W + iw
            x_val = tl.load(X_ptr + x_idx, mask=valid, other=0.0)

            # Weight: [C, 1, KH, KW] -> index [c, 0, kh, kw]
            w_idx = offs_c * KH * KW + kh * KW + kw
            w_val = tl.load(W_ptr + w_idx, mask=mask_c, other=0.0)

            acc += x_val * w_val[:, None]

    # BN + ReLU6
    out = (acc - mean[:, None]) * (inv_std[:, None] * w[:, None]) + b_[:, None]
    out = tl.maximum(out, 0.0)
    out = tl.minimum(out, 6.0)

    # Store: output shape [N, C, OUT_H, OUT_W]
    out_idx = pid_n * C * OUT_H * OUT_W + offs_c[:, None] * OUT_H * OUT_W + out_h[None, :] * OUT_W + out_w[None, :]
    tl.store(Out_ptr + out_idx, out.to(Out_ptr.dtype.element_ty),
             mask=mask_c[:, None] & mask_hw[None, :])


# ─── Kernel 3: 1x1 GEMM (project conv: M=N*H*W, K=672, N=192) + BN ─────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_bn_kernel(
    A_ptr, B_ptr, Out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(a, b, acc, allow_tf32=False)

    # BN epilogue (no activation)
    mean = tl.load(mean_ptr   + offs_n, mask=mask_n, other=0.0)
    var  = tl.load(var_ptr    + offs_n, mask=mask_n, other=0.0)
    w    = tl.load(weight_ptr + offs_n, mask=mask_n, other=1.0)
    b_   = tl.load(bias_ptr   + offs_n, mask=mask_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(var[None, :] + eps)
    out = (acc - mean[None, :]) * inv_std * w[None, :] + b_[None, :]

    out_ptrs = Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, out.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(
):
    device = x.device
    dtype  = x.dtype
    eps    = 1e-5

    batch, in_c, H, W = x.shape  # 10, 112, 224, 224
    expand_c = 672
    out_c    = 192
    KH, KW   = 5, 5
    stride_dw = 2
    pad_dw    = 2
    OUT_H = (H + 2 * pad_dw - KH) // stride_dw + 1  # 112
    OUT_W = (W + 2 * pad_dw - KW) // stride_dw + 1  # 112

    # ── Step 1: Expand 1x1 conv + BN + ReLU6 ──────────────────────────────
    # x: [10, 112, 224, 224] -> reshape to [10*224*224, 112]
    # weight: [672, 112, 1, 1] -> [112, 672]
    M1 = batch * H * W   # 501760
    K1 = in_c             # 112
    N1 = expand_c         # 672

    x_2d = x.permute(0, 2, 3, 1).reshape(M1, K1).contiguous()  # [M1, K1]
    w1   = p_expand_conv_0_weight.reshape(N1, K1).t().contiguous()  # [K1, N1]

    expand_out = torch.empty((M1, N1), device=device, dtype=dtype)

    grid1 = (triton.cdiv(M1, 128), triton.cdiv(N1, 64))
    gemm_bn_relu6_kernel[grid1](
        x_2d, w1, expand_out,
        b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
        p_expand_conv_1_weight, p_expand_conv_1_bias,
        M1, N1, K1,
        x_2d.stride(0), x_2d.stride(1),
        w1.stride(0), w1.stride(1),
        expand_out.stride(0), expand_out.stride(1),
        eps,
    )
    # expand_out: [M1, N1] = [501760, 672], represents [10, 224, 224, 672]

    # ── Step 2: Depthwise conv 5x5 stride=2 pad=2 + BN + ReLU6 ───────────
    # Input: [10, 672, 224, 224] (need NCHW for depthwise)
    # expand_out is [10, 224, 224, 672] -> permute to [10, 672, 224, 224]
    expand_nchw = expand_out.reshape(batch, H, W, expand_c).permute(0, 3, 1, 2).contiguous()

    dw_out = torch.empty((batch, expand_c, OUT_H, OUT_W), device=device, dtype=dtype)

    # Weight: [672, 1, 5, 5] -> keep as is
    dw_weight = p_depthwise_conv_0_weight.reshape(expand_c, KH * KW).contiguous()  # [672, 25]
    # Actually keep original shape for indexing
    dw_weight_orig = p_depthwise_conv_0_weight.contiguous()  # [672, 1, 5, 5]

    BLOCK_C_dw  = 32
    BLOCK_HW_dw = 64
    grid2 = (
        triton.cdiv(expand_c, BLOCK_C_dw),
        triton.cdiv(OUT_H * OUT_W, BLOCK_HW_dw),
        batch,
    )
    depthwise_conv_bn_relu6_kernel[grid2](
        expand_nchw, dw_weight_orig, dw_out,
        b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
        p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
        batch, expand_c, H, W, OUT_H, OUT_W,
        KH, KW,
        stride_dw, stride_dw,
        pad_dw, pad_dw,
        eps,
    )

    # ── Step 3: Project 1x1 conv + BN ─────────────────────────────────────
    # dw_out: [10, 672, 112, 112] -> [10*112*112, 672]
    M3 = batch * OUT_H * OUT_W  # 125440
    K3 = expand_c               # 672
    N3 = out_c                  # 192

    dw_2d = dw_out.permute(0, 2, 3, 1).reshape(M3, K3).contiguous()  # [M3, K3]
    w3    = p_project_conv_0_weight.reshape(N3, K3).t().contiguous()  # [K3, N3]

    proj_out = torch.empty((M3, N3), device=device, dtype=dtype)

    grid3 = (triton.cdiv(M3, 128), triton.cdiv(N3, 64))
    gemm_bn_kernel[grid3](
        dw_2d, w3, proj_out,
        b_project_conv_1_running_mean, b_project_conv_1_running_var,
        p_project_conv_1_weight, p_project_conv_1_bias,
        M3, N3, K3,
        dw_2d.stride(0), dw_2d.stride(1),
        w3.stride(0), w3.stride(1),
        proj_out.stride(0), proj_out.stride(1),
        eps,
    )

    # Reshape to [10, 192, 112, 112]
    result = proj_out.reshape(batch, OUT_H, OUT_W, out_c).permute(0, 3, 1, 2).contiguous()
    return result


def get_inputs():
    batch_size   = 10
    in_channels  = 112
    out_channels = 192
    expand_ratio = 6
    hidden_dim   = in_channels * expand_ratio  # 672

    p_expand_conv_0_weight    = torch.randn(hidden_dim, in_channels, 1, 1)
    p_expand_conv_1_weight    = torch.randn(hidden_dim)
    p_expand_conv_1_bias      = torch.randn(hidden_dim)
    p_depthwise_conv_0_weight = torch.randn(hidden_dim, 1, 5, 5)
    p_depthwise_conv_1_weight = torch.randn(hidden_dim)
    p_depthwise_conv_1_bias   = torch.randn(hidden_dim)
    p_project_conv_0_weight   = torch.randn(out_channels, hidden_dim, 1, 1)
    p_project_conv_1_weight   = torch.randn(out_channels)
    p_project_conv_1_bias     = torch.randn(out_channels)
    b_expand_conv_1_running_mean    = torch.zeros(hidden_dim)
    b_expand_conv_1_running_var     = torch.ones(hidden_dim)
    b_depthwise_conv_1_running_mean = torch.zeros(hidden_dim)
    b_depthwise_conv_1_running_var  = torch.ones(hidden_dim)
    b_project_conv_1_running_mean   = torch.zeros(out_channels)
    b_project_conv_1_running_var    = torch.ones(out_channels)
    x = torch.randn(batch_size, in_channels, 224, 224)

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

        (p_expand_conv_0_weight, p_expand_conv_1_weight, p_expand_conv_1_bias,
         p_depthwise_conv_0_weight, p_depthwise_conv_1_weight, p_depthwise_conv_1_bias,
         p_project_conv_0_weight, p_project_conv_1_weight, p_project_conv_1_bias,
         b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
         b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
         b_project_conv_1_running_mean, b_project_conv_1_running_var,
         x) = inputs_cuda

        # Reference computation
        import torch.nn.functional as F

        # Expand conv + BN + ReLU6
        out1 = F.conv2d(x, p_expand_conv_0_weight, bias=None, stride=1, padding=0)
        out1 = F.batch_norm(out1, b_expand_conv_1_running_mean, b_expand_conv_1_running_var,
                            p_expand_conv_1_weight, p_expand_conv_1_bias, training=False, eps=1e-5)
        out1 = out1.clamp(0.0, 6.0)

        # Depthwise conv + BN + ReLU6
        out2 = F.conv2d(out1, p_depthwise_conv_0_weight, bias=None, stride=2, padding=2,
                        groups=672)
        out2 = F.batch_norm(out2, b_depthwise_conv_1_running_mean, b_depthwise_conv_1_running_var,
                            p_depthwise_conv_1_weight, p_depthwise_conv_1_bias, training=False, eps=1e-5)
        out2 = out2.clamp(0.0, 6.0)

        # Project conv + BN
        out3 = F.conv2d(out2, p_project_conv_0_weight, bias=None, stride=1, padding=0)
        ref  = F.batch_norm(out3, b_project_conv_1_running_mean, b_project_conv_1_running_var,
                            p_project_conv_1_weight, p_project_conv_1_bias, training=False, eps=1e-5)

        print(f"Output shape: {output.shape}, Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={(output - ref).abs().max().item():.4e}"
        print("Kernel executed successfully.")