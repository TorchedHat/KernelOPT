import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

# ─── Kernel 1: Implicit GEMM Conv2d ───────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, K,
    # strides for X (NHWC layout)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (OUT_C, IN_C, KH, KW) -> stored as (OUT_C, KH, KW, IN_C) for coalescing
    sw_oc, sw_kh, sw_kw, sw_ic,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KW
        kh_idx = offs_k // (IN_C * KW)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m < M
        valid_k = offs_k < K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        mask_x = valid_m[:, None] & valid_k[None, :] & valid_spatial

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: (OUT_C, KH, KW, IN_C) layout
        mask_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias_mask = offs_n < OUT_C
    bias = tl.load(B_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store: output in (N, OUT_C, OUT_H, OUT_W) NCHW layout
    # Y[batch, oc, oh, ow] = Y[batch * OUT_C*OUT_H*OUT_W + oc * OUT_H*OUT_W + oh * OUT_W + ow]
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ─── Kernel 2: GroupNorm mean/var reduction ────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_R': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_R': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_R': 2048}, num_stages=3, num_warps=8),
    ],
    key=['R'],
)
@triton.jit
def groupnorm_mean_var_kernel(
    X_ptr, mean_ptr, var_ptr,
    N, G, C_per_G, HW,
    R,  # = C_per_G * HW
    BLOCK_R: tl.constexpr,
):
    # Each program handles one (batch, group) pair
    pid = tl.program_id(0)
    batch = pid // G
    group = pid % G

    # Base pointer for this (batch, group)
    # X layout: (N, C, H, W) = (N, G, C_per_G, HW)
    # X[batch, group, c_in_group, hw] = X_ptr + batch*(G*C_per_G*HW) + group*(C_per_G*HW) + c_in_group*HW + hw
    base = batch * (G * C_per_G * HW) + group * (C_per_G * HW)

    mean_acc = tl.zeros((), dtype=tl.float32)
    m2_acc = tl.zeros((), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)

    for r_start in range(0, R, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R
        # Map r -> (c_in_group, hw_idx)
        c_in_g = offs_r // HW
        hw_idx = offs_r % HW
        # In NCHW: X[batch, group*C_per_G + c_in_g, h, w]
        # = X_ptr + batch*(G*C_per_G*HW) + (group*C_per_G + c_in_g)*HW + hw_idx
        ptrs = X_ptr + batch * (G * C_per_G * HW) + (group * C_per_G + c_in_g) * HW + hw_idx
        vals = tl.load(ptrs, mask=mask_r, other=0.0).to(tl.float32)

        # Welford online update
        new_count = count + tl.sum(mask_r.to(tl.float32), axis=0)
        delta = vals - mean_acc
        mean_acc = mean_acc + tl.sum(tl.where(mask_r, delta / tl.maximum(new_count, 1.0), 0.0), axis=0)
        delta2 = vals - mean_acc
        m2_acc = m2_acc + tl.sum(tl.where(mask_r, delta * delta2, 0.0), axis=0)
        count = new_count

    var_acc = m2_acc / tl.maximum(count, 1.0)

    tl.store(mean_ptr + pid, mean_acc)
    tl.store(var_ptr + pid, var_acc)


# ─── Kernel 3: GroupNorm normalize + scale + MaxPool + Clamp ──────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 4, 'BLOCK_P': 64}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 8, 'BLOCK_P': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 4, 'BLOCK_P': 32}, num_stages=2, num_warps=2),
        triton.Config({'BLOCK_C': 8, 'BLOCK_P': 64}, num_stages=2, num_warps=8),
    ],
    key=['N', 'C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def groupnorm_scale_maxpool_clamp_kernel(
    X_ptr, mean_ptr, var_ptr,
    gn_weight_ptr, gn_bias_ptr, scale_ptr,
    Y_ptr,
    N, C, G, C_per_G,
    IN_H, IN_W,
    OUT_H, OUT_W,
    POOL_K,
    clamp_min, clamp_max,
    eps,
    BLOCK_C: tl.constexpr,
    BLOCK_P: tl.constexpr,
):
    # Each program handles a tile of (batch, channel_block, pool_out_pos)
    pid_n = tl.program_id(0)
    pid_c = tl.program_id(1)
    pid_p = tl.program_id(2)

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    offs_p = pid_p * BLOCK_P + tl.arange(0, BLOCK_P)

    # Pool output position -> (out_h, out_w)
    out_h = offs_p // OUT_W
    out_w = offs_p % OUT_W

    mask_c = offs_c < C
    mask_p = offs_p < (OUT_H * OUT_W)

    # Load per-channel stats
    group_idx = offs_c // C_per_G  # shape [BLOCK_C]
    mean_ptrs = mean_ptr + pid_n * G + group_idx
    var_ptrs = var_ptr + pid_n * G + group_idx
    mean_vals = tl.load(mean_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    var_vals = tl.load(var_ptrs, mask=mask_c, other=0.0).to(tl.float32)
    inv_std = 1.0 / tl.sqrt(var_vals + eps)

    gn_w = tl.load(gn_weight_ptr + offs_c, mask=mask_c, other=1.0).to(tl.float32)
    gn_b = tl.load(gn_bias_ptr + offs_c, mask=mask_c, other=0.0).to(tl.float32)
    scale = tl.load(scale_ptr + offs_c, mask=mask_c, other=1.0).to(tl.float32)

    # Combined scale: gn_w * inv_std * scale
    combined_scale = gn_w * inv_std * scale  # [BLOCK_C]
    combined_bias = gn_b * scale             # [BLOCK_C]

    # MaxPool: for each output position, find max over POOL_K x POOL_K window
    # Initialize accumulator
    pool_max = tl.full((BLOCK_C, BLOCK_P), float('-inf'), dtype=tl.float32)

    for ki in range(POOL_K):
        for kj in range(POOL_K):
            ih = out_h[None, :] * POOL_K + ki  # [1, BLOCK_P]
            iw = out_w[None, :] * POOL_K + kj  # [1, BLOCK_P]

            valid_spatial = (ih < IN_H) & (iw < IN_W) & mask_p[None, :]

            # X[pid_n, c, ih, iw] in NCHW
            x_ptrs = (X_ptr
            x_vals = tl.load(x_ptrs, mask=mask_c[:, None] & valid_spatial, other=float('-inf')).to(tl.float32)

            # Apply GroupNorm + scale
            normed = x_vals * combined_scale[:, None] + combined_bias[:, None]

            pool_max = tl.maximum(pool_max, normed)

    # Clamp
    pool_max = tl.maximum(pool_max, clamp_min)
    pool_max = tl.minimum(pool_max, clamp_max)

    # Store Y[pid_n, c, out_h, out_w]
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, pool_max.to(tl.float32), mask=mask_c[:, None] & mask_p[None, :])


def kernel_function(p_conv_weight, p_conv_bias, p_group_norm_weight, p_group_norm_bias, p_scale, x):
    # x: (128, 8, 128, 128) NCHW
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = p_conv_weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1

    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)

    # Convert weight to (OUT_C, KH, KW, IN_C) for coalesced access
    w_okhwc = p_conv_weight.permute(0, 2, 3, 1).contiguous()  # (OUT_C, KH, KW, IN_C)

    # Allocate conv output in NCHW
    conv_out = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=torch.float32)

    M = N * OUT_H * OUT_W
    K = IN_C * KH * KW

    grid_conv = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv2d_implicit_gemm_kernel[grid_conv](
        x_nhwc, w_okhwc, p_conv_bias, conv_out,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M, K,
        # strides for x_nhwc (N, H, W, C)
        IN_H * IN_W * IN_C, IN_W * IN_C, IN_C, 1,
        # strides for w_okhwc (OUT_C, KH, KW, IN_C)
        KH * KW * IN_C, KW * IN_C, IN_C, 1,
    )

    # GroupNorm parameters
    G = 16  # num_groups
    C_per_G = OUT_C // G
    HW = OUT_H * OUT_W
    R = C_per_G * HW

    mean_buf = torch.empty((N * G,), device=x.device, dtype=torch.float32)
    var_buf = torch.empty((N * G,), device=x.device, dtype=torch.float32)

    grid_gn_stats = (N * G,)
    groupnorm_mean_var_kernel[grid_gn_stats](
        conv_out, mean_buf, var_buf,
        N, G, C_per_G, HW,
        R,
    )

    # MaxPool params
    POOL_K = 4
    POOL_OUT_H = OUT_H // POOL_K  # 126 // 4 = 31
    POOL_OUT_W = OUT_W // POOL_K

    clamp_min = 0.0
    clamp_max = 1.0
    eps = 1e-5

    # scale: (64, 1, 1) -> flatten to (64,)
    scale_flat = p_scale.view(OUT_C)

    y_out = torch.empty((N, OUT_C, POOL_OUT_H, POOL_OUT_W), device=x.device, dtype=torch.float32)

    grid_fused = lambda meta: (
        N,
        triton.cdiv(OUT_C, meta['BLOCK_C']),
        triton.cdiv(POOL_OUT_H * POOL_OUT_W, meta['BLOCK_P']),
    )

    groupnorm_scale_maxpool_clamp_kernel[grid_fused](
        conv_out, mean_buf, var_buf,
        p_group_norm_weight, p_group_norm_bias, scale_flat,
        y_out,
        N, OUT_C, G, C_per_G,
        OUT_H, OUT_W,
        POOL_OUT_H, POOL_OUT_W,
        POOL_K,
        clamp_min, clamp_max,
        eps,
    )

    return y_out


def get_inputs():
    batch_size = 128
    in_channels = 8
    out_channels = 64
    height, width = 128, 128
    kernel_size = 3
    num_groups = 16
    scale_shape = (out_channels, 1, 1)
    maxpool_kernel_size = 4

    p_conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda')
    p_conv_bias = torch.randn(out_channels, device='cuda')
    p_group_norm_weight = torch.ones(out_channels, device='cuda')
    p_group_norm_bias = torch.zeros(out_channels, device='cuda')
    p_scale = torch.ones(scale_shape, device='cuda')
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')
    return [p_conv_weight, p_conv_bias, p_group_norm_weight, p_group_norm_bias, p_scale, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_conv_weight, p_conv_bias, p_group_norm_weight, p_group_norm_bias, p_scale, x = inputs

        import torch.nn as nn
        import torch.nn.functional as F

        # Reference computation
        conv_ref = F.conv2d(x, p_conv_weight, p_conv_bias)
        # GroupNorm
        gn = nn.GroupNorm(16, 64, eps=1e-5).to('cuda')
        gn.weight.data = p_group_norm_weight
        gn.bias.data = p_group_norm_bias
        gn_out = gn(conv_ref)
        scaled = gn_out * p_scale
        pooled = F.max_pool2d(scaled, kernel_size=4)
        ref = torch.clamp(pooled, 0.0, 1.0)

        print(f"Output shape: {output.shape}, ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.6f}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")