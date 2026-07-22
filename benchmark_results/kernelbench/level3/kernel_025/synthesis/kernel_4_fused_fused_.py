import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice

# ─── Kernel 1: Group Conv 1x1 (conv1: 240->120, groups=3) + BN + ReLU ───────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def group_conv1x1_bn_relu_kernel(
    x_ptr, w_ptr, out_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    M, N, K,  # M=batch*H*W, N=OUT_C_per_group, K=IN_C_per_group
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    GROUPS: tl.constexpr,
    do_relu: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Global channel indices
    global_n = pid_g * N + offs_n  # global output channel

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + offs_k
        mask_k = k_offs < K
        mask_m = offs_m < M
        mask_n = offs_n < N

        # x: [M, GROUPS*K] -> group g uses columns [g*K : (g+1)*K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + (pid_g * K + k_offs)[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # w: [GROUPS*N, K] -> group g uses rows [g*N : (g+1)*N]
        w_ptrs = w_ptr + (pid_g * N + offs_n)[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w_mask = mask_n[:, None] & mask_k[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=False)

    # BN + optional ReLU
    mask_m = offs_m < M
    mask_n = offs_n < N

    bn_mean = tl.load(bn_mean_ptr + global_n, mask=mask_n, other=0.0)
    bn_var  = tl.load(bn_var_ptr  + global_n, mask=mask_n, other=1.0)
    bn_w    = tl.load(bn_weight_ptr + global_n, mask=mask_n, other=1.0)
    bn_b    = tl.load(bn_bias_ptr   + global_n, mask=mask_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    acc = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]

    if do_relu:
        acc = tl.maximum(acc, 0.0)

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + global_n[None, :] * stride_on
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ─── Kernel 2: Depthwise Conv 3x3 + BN (no relu) ────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_C': 64}, num_warps=4, num_stages=3),
    ],
    key=['M_total', 'C'],
)
@triton.jit
def depthwise_conv3x3_bn_kernel(
    x_ptr, w_ptr, out_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    N, C, H, W,
    eps: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)

    mask_m = offs_m < M_total
    mask_c = offs_c < C

    # Decode m -> (n, h, w)
    w_idx = offs_m % W
    h_idx = (offs_m // W) % H
    n_idx = offs_m // (W * H)

    # Load BN params
    bn_mean = tl.load(bn_mean_ptr + offs_c, mask=mask_c, other=0.0)
    bn_var  = tl.load(bn_var_ptr  + offs_c, mask=mask_c, other=1.0)
    bn_w    = tl.load(bn_weight_ptr + offs_c, mask=mask_c, other=1.0)
    bn_b    = tl.load(bn_bias_ptr   + offs_c, mask=mask_c, other=0.0)
    inv_std = 1.0 / tl.sqrt(bn_var + eps)

    # For each (m, c) pair, compute depthwise conv
    # x layout: [N, C, H, W] -> stride_n=C*H*W, stride_c=H*W, stride_h=W, stride_w=1
    # w layout: [C, 1, 3, 3] -> stride_c=9, stride_kh=3, stride_kw=1

    acc = tl.zeros((BLOCK_M, BLOCK_C), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = h_idx[:, None] + kh - 1  # padding=1
            iw = w_idx[:, None] + kw - 1

            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            valid = valid & mask_m[:, None] & mask_c[None, :]

            x_idx = n_idx[:, None] * (C * H * W) + offs_c[None, :] * (H * W) + ih * W + iw
            x_val = tl.load(x_ptr + x_idx, mask=valid, other=0.0)

            w_idx_flat = offs_c * 9 + kh * 3 + kw
            w_val = tl.load(w_ptr + w_idx_flat, mask=mask_c, other=0.0)

            acc += x_val * w_val[None, :]

    # Apply BN (no relu)
    acc = (acc - bn_mean[None, :]) * inv_std[None, :] * bn_w[None, :] + bn_b[None, :]

    # Output layout: [N, C, H, W]
    out_idx = n_idx[:, None] * (C * H * W) + offs_c[None, :] * (H * W) + h_idx[:, None] * W + w_idx[:, None]
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=mask_m[:, None] & mask_c[None, :])


# ─── Kernel 3: Channel Shuffle ───────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=3),
    ],
    key=['total'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    groups: tl.constexpr,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # offs = n*(C*H*W) + c*(H*W) + hw
    hw = offs % (H * W)
    tmp = offs // (H * W)
    c = tmp % C
    n = tmp // C

    # channel shuffle: c -> (c % groups_per_group) * groups + (c // groups_per_group)
    # groups=3, channels_per_group = C // groups
    cpg = C // groups
    # original c = group_idx * cpg + within_group
    group_idx = c // cpg
    within_group = c % cpg
    # shuffled c = within_group * groups + group_idx
    new_c = within_group * groups + group_idx

    src_idx = n * (C * H * W) + c * (H * W) + hw
    dst_idx = n * (C * H * W) + new_c * (H * W) + hw

    val = tl.load(x_ptr + src_idx, mask=mask, other=0.0)
    tl.store(out_ptr + dst_idx, val, mask=mask)


# ─── Kernel 4: Group Conv 1x1 (conv3: 120->480, groups=3) + BN + ReLU ────────
# Reuses group_conv1x1_bn_relu_kernel

# ─── Kernel 5: Shortcut Conv 1x1 (240->480, groups=1) + BN + Add ─────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bn_add_kernel(
    x_ptr, w_ptr, residual_ptr, out_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, K, BLOCK_K):
        k_offs = k + tl.arange(0, BLOCK_K)
        mask_k = k_offs < K
        mask_m = offs_m < M
        mask_n = offs_n < N

        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = mask_m[:, None] & mask_k[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = w_ptr + offs_n[:, None] * stride_wn + k_offs[None, :] * stride_wk
        w_mask = mask_n[:, None] & mask_k[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=False)

    mask_m = offs_m < M
    mask_n = offs_n < N
    out_mask = mask_m[:, None] & mask_n[None, :]

    bn_mean = tl.load(bn_mean_ptr + offs_n, mask=mask_n, other=0.0)
    bn_var  = tl.load(bn_var_ptr  + offs_n, mask=mask_n, other=1.0)
    bn_w    = tl.load(bn_weight_ptr + offs_n, mask=mask_n, other=1.0)
    bn_b    = tl.load(bn_bias_ptr   + offs_n, mask=mask_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + eps)
    acc = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]

    # Load residual and add
    res_ptrs = residual_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    res = tl.load(res_ptrs, mask=out_mask, other=0.0)
    acc = acc + res

    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_conv2_weight, p_bn2_weight, p_bn2_bias,
    p_conv3_weight, p_bn3_weight, p_bn3_bias,
    p_shortcut_0_weight, p_shortcut_1_weight, p_shortcut_1_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_bn2_running_mean, b_bn2_running_var,
    b_bn3_running_mean, b_bn3_running_var,
    b_shortcut_1_running_mean, b_shortcut_1_running_var,
    x,
):
    device = x.device
    dtype = x.dtype
    N_batch, IN_C, H, W = x.shape  # 10, 240, 224, 224
    groups = 3
    MID_C = 120  # out_channels // 4 = 480 // 4
    OUT_C = 480

    # ── Step 1: conv1 (group 1x1: 240->120, groups=3) + BN1 + ReLU ──────────
    # x: [10, 240, 224, 224] in NCHW
    # Reshape to [M, IN_C] where M = N*H*W
    M = N_batch * H * W  # 10 * 224 * 224 = 501760
    IN_C_per_group = IN_C // groups   # 80
    OUT_C1_per_group = MID_C // groups  # 40

    # x_flat: [M, IN_C] - need contiguous in (N, H, W, C) order
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [10, 224, 224, 240]
    x_flat = x_nhwc.reshape(M, IN_C)  # [M, 240]

    # w1: [120, 80, 1, 1] -> [120, 80]
    w1_flat = p_conv1_weight.reshape(MID_C, IN_C_per_group)  # [120, 80]

    out1 = torch.empty(M, MID_C, device=device, dtype=dtype)

    grid1 = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C1_per_group, meta['BLOCK_N']),
        groups,
    )
    group_conv1x1_bn_relu_kernel[grid1](
        x_flat, w1_flat, out1,
        b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias,
        M, OUT_C1_per_group, IN_C_per_group,
        x_flat.stride(0), x_flat.stride(1),
        w1_flat.stride(0), w1_flat.stride(1),
        out1.stride(0), out1.stride(1),
        GROUPS=groups,
        do_relu=True,
        eps=1e-5,
    )
    # out1: [M, 120] -> reshape to [10, 224, 224, 120] -> [10, 120, 224, 224]
    out1_nchw = out1.reshape(N_batch, H, W, MID_C).permute(0, 3, 1, 2).contiguous()

    # ── Step 2: conv2 (depthwise 3x3: 120->120) + BN2 ───────────────────────
    # w2: [120, 1, 3, 3]
    out2 = torch.empty(N_batch, MID_C, H, W, device=device, dtype=dtype)
    M_total = N_batch * H * W

    grid2 = lambda meta: (
        triton.cdiv(M_total, meta['BLOCK_M']),
        triton.cdiv(MID_C, meta['BLOCK_C']),
    )
    depthwise_conv3x3_bn_kernel[grid2](
        out1_nchw, p_conv2_weight, out2,
        b_bn2_running_mean, b_bn2_running_var, p_bn2_weight, p_bn2_bias,
        N_batch, MID_C, H, W,
        M_total,
        eps=1e-5,
    )

    # ── Step 3: Channel Shuffle (groups=3) ──────────────────────────────────
    out3 = torch.empty_like(out2)
    total = N_batch * MID_C * H * W

    grid3 = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid3](
        out2, out3,
        N_batch, MID_C, H, W,
        groups=groups,
        total=total,
    )

    # ── Step 4: conv3 (group 1x1: 120->480, groups=3) + BN3 + ReLU ──────────
    OUT_C3_per_group = OUT_C // groups  # 160
    IN_C3_per_group = MID_C // groups   # 40

    # w3: [480, 40, 1, 1] -> [480, 40]
    w3_flat = p_conv3_weight.reshape(OUT_C, IN_C3_per_group)  # [480, 40]

    # out3 in NCHW -> NHWC -> flat
    out3_nhwc = out3.permute(0, 2, 3, 1).contiguous()
    out3_flat = out3_nhwc.reshape(M, MID_C)

    out4 = torch.empty(M, OUT_C, device=device, dtype=dtype)

    grid4 = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C3_per_group, meta['BLOCK_N']),
        groups,
    )
    group_conv1x1_bn_relu_kernel[grid4](
        out3_flat, w3_flat, out4,
        b_bn3_running_mean, b_bn3_running_var, p_bn3_weight, p_bn3_bias,
        M, OUT_C3_per_group, IN_C3_per_group,
        out3_flat.stride(0), out3_flat.stride(1),
        w3_flat.stride(0), w3_flat.stride(1),
        out4.stride(0), out4.stride(1),
        GROUPS=groups,
        do_relu=True,
        eps=1e-5,
    )
    # out4: [M, 480] -> [10, 224, 224, 480] -> [10, 480, 224, 224]
    out4_nchw = out4.reshape(N_batch, H, W, OUT_C).permute(0, 3, 1, 2).contiguous()

    # ── Step 5: Shortcut (conv 1x1: 240->480) + BN + Add ────────────────────
    # w_sc: [480, 240, 1, 1] -> [480, 240]
    w_sc_flat = p_shortcut_0_weight.reshape(OUT_C, IN_C)  # [480, 240]

    # x_flat already computed above: [M, 240]
    out5 = torch.empty(M, OUT_C, device=device, dtype=dtype)

    # out4 as residual in [M, 480] layout
    out4_flat = out4_nchw.permute(0, 2, 3, 1).contiguous().reshape(M, OUT_C)

    grid5 = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    conv1x1_bn_add_kernel[grid5](
        x_flat, w_sc_flat, out4_flat, out5,
        b_shortcut_1_running_mean, b_shortcut_1_running_var,
        p_shortcut_1_weight, p_shortcut_1_bias,
        M, OUT_C, IN_C,
        x_flat.stride(0), x_flat.stride(1),
        w_sc_flat.stride(0), w_sc_flat.stride(1),
        out5.stride(0), out5.stride(1),
        eps=1e-5,
    )

    # Reshape to [10, 480, 224, 224]
    result = out5.reshape(N_batch, H, W, OUT_C).permute(0, 3, 1, 2).contiguous()
    return result


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'
    p_conv1_weight = torch.randn(120, 80, 1, 1, device=device)
    p_bn1_weight = torch.ones(120, device=device)
    p_bn1_bias = torch.zeros(120, device=device)
    p_conv2_weight = torch.randn(120, 1, 3, 3, device=device)
    p_bn2_weight = torch.ones(120, device=device)
    p_bn2_bias = torch.zeros(120, device=device)
    p_conv3_weight = torch.randn(480, 40, 1, 1, device=device)
    p_bn3_weight = torch.ones(480, device=device)
    p_bn3_bias = torch.zeros(480, device=device)
    p_shortcut_0_weight = torch.randn(480, 240, 1, 1, device=device)
    p_shortcut_1_weight = torch.ones(480, device=device)
    p_shortcut_1_bias = torch.zeros(480, device=device)
    b_bn1_running_mean = torch.zeros(120, device=device)
    b_bn1_running_var = torch.ones(120, device=device)
    b_bn2_running_mean = torch.zeros(120, device=device)
    b_bn2_running_var = torch.ones(120, device=device)
    b_bn3_running_mean = torch.zeros(480, device=device)
    b_bn3_running_var = torch.ones(480, device=device)
    b_shortcut_1_running_mean = torch.zeros(480, device=device)
    b_shortcut_1_running_var = torch.ones(480, device=device)
    x = torch.randn(10, 240, 224, 224, device=device)
    return [
        p_conv1_weight, p_bn1_weight, p_bn1_bias,
        p_conv2_weight, p_bn2_weight, p_bn2_bias,
        p_conv3_weight, p_bn3_weight, p_bn3_bias,
        p_shortcut_0_weight, p_shortcut_1_weight, p_shortcut_1_bias,
        b_bn1_running_mean, b_bn1_running_var,
        b_bn2_running_mean, b_bn2_running_var,
        b_bn3_running_mean, b_bn3_running_var,
        b_shortcut_1_running_mean, b_shortcut_1_running_var,
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
        (
            p_conv1_weight, p_bn1_weight, p_bn1_bias,
            p_conv2_weight, p_bn2_weight, p_bn2_bias,
            p_conv3_weight, p_bn3_weight, p_bn3_bias,
            p_shortcut_0_weight, p_shortcut_1_weight, p_shortcut_1_bias,
            b_bn1_running_mean, b_bn1_running_var,
            b_bn2_running_mean, b_bn2_running_var,
            b_bn3_running_mean, b_bn3_running_var,
            b_shortcut_1_running_mean, b_shortcut_1_running_var,
            x,
        ) = inputs

        import torch.nn.functional as F

        def bn_inference(x, mean, var, weight, bias, eps=1e-5):
            return (x - mean[None, :, None, None]) / torch.sqrt(var[None, :, None, None] + eps) * weight[None, :, None, None] + bias[None, :, None, None]

        # conv1: group conv 1x1
        ref1 = F.conv2d(x, p_conv1_weight, None, stride=1, padding=0, groups=3)
        ref1 = bn_inference(ref1, b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias)
        ref1 = F.relu(ref1)

        # conv2: dep