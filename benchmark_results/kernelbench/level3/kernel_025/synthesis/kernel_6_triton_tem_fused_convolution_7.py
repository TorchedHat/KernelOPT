import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: Group Conv 1x1 (240->120, groups=3) + BN + ReLU ───────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'GROUP_OUT_C', 'GROUP_IN_C'],
)
@triton.jit
def group_conv1x1_bn_relu_kernel(
    X_ptr, W_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    Y_ptr,
    # dims
    BATCH, IN_C, OUT_C, H, W,
    GROUPS: tl.constexpr,
    # strides for X (NHWC layout)
    sx_n, sx_h, sx_w, sx_c,
    # strides for W (out_c, in_c_per_group)
    sw_o, sw_i,
    # strides for Y (NCHW layout)
    sy_n, sy_c, sy_h, sy_w,
    GROUP_IN_C: tl.constexpr,
    GROUP_OUT_C: tl.constexpr,
    EPS,
    DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # decode m -> (batch, h, w)
    w_idx = offs_m % W
    h_idx = (offs_m // W) % H
    n_idx = offs_m // (W * H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, GROUP_IN_C, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)
        # X in NHWC: ptr = X + n*sx_n + h*sx_h + w*sx_w + (g*GROUP_IN_C + k_local)*sx_c
        x_ptrs = (X_ptr
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < GROUP_IN_C)
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # W shape: (OUT_C, GROUP_IN_C) stored as (GROUP_OUT_C per group)
        # W_ptr offset: group * GROUP_OUT_C * GROUP_IN_C
        w_ptrs = (W_ptr
        mask_w = (offs_k[:, None] < GROUP_IN_C) & (offs_n[None, :] < GROUP_OUT_C)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # BN + optional ReLU epilogue
    out_c_global = pid_g * GROUP_OUT_C + offs_n  # [BLOCK_N]
    mean_vals = tl.load(mean_ptr + out_c_global, mask=offs_n < GROUP_OUT_C, other=0.0)
    var_vals = tl.load(var_ptr + out_c_global, mask=offs_n < GROUP_OUT_C, other=1.0)
    gamma_vals = tl.load(gamma_ptr + out_c_global, mask=offs_n < GROUP_OUT_C, other=1.0)
    beta_vals = tl.load(beta_ptr + out_c_global, mask=offs_n < GROUP_OUT_C, other=0.0)

    inv_std = 1.0 / tl.sqrt(var_vals + EPS)
    # broadcast: acc [BLOCK_M, BLOCK_N], mean/var/gamma/beta [BLOCK_N]
    result = (acc - mean_vals[None, :]) * inv_std[None, :] * gamma_vals[None, :] + beta_vals[None, :]

    if DO_RELU:
        result = tl.maximum(result, 0.0)

    # Store to Y in NCHW layout
    # Y[n, c, h, w] = Y_ptr + n*sy_n + c*sy_c + h*sy_h + w*sy_w
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < GROUP_OUT_C)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, result.to(tl.float32), mask=mask_out)


# ─── Kernel 2: Depthwise Conv 3x3 (120->120) + BN ────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 1}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 1}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 1}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 1024, 'BLOCK_N': 1}, num_warps=8, num_stages=2),
    ],
    key=['M', 'C'],
)
@triton.jit
def depthwise_conv3x3_bn_kernel(
    X_ptr, W_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    Y_ptr,
    BATCH, C, H, W,
    # X strides (NCHW)
    sx_n, sx_c, sx_h, sx_w,
    # W strides: W[c, 0, kh, kw] -> W_ptr + c*9 + kh*3 + kw
    # Y strides (NCHW)
    sy_n, sy_c, sy_h, sy_w,
    EPS,
    BLOCK_M: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M

    # decode: offs -> (n, c, h, w)
    w_idx = offs % W
    h_idx = (offs // W) % H
    c_idx = (offs // (W * H)) % C
    n_idx = offs // (W * H * C)

    # accumulate over 3x3 kernel
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for kh in tl.static_range(3):
        for kw in tl.static_range(3):
            ih = h_idx + kh - 1
            iw = w_idx + kw - 1
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            x_ptrs = X_ptr + n_idx * sx_n + c_idx * sx_c + ih * sx_h + iw * sx_w
            x_val = tl.load(x_ptrs, mask=valid, other=0.0)
            w_val = tl.load(W_ptr + c_idx * 9 + kh * 3 + kw, mask=mask, other=0.0)
            acc += x_val * w_val

    # BN (no relu)
    mean_val = tl.load(mean_ptr + c_idx, mask=mask, other=0.0)
    var_val = tl.load(var_ptr + c_idx, mask=mask, other=1.0)
    gamma_val = tl.load(gamma_ptr + c_idx, mask=mask, other=1.0)
    beta_val = tl.load(beta_ptr + c_idx, mask=mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var_val + EPS)
    result = (acc - mean_val) * inv_std * gamma_val + beta_val

    y_ptrs = Y_ptr + n_idx * sy_n + c_idx * sy_c + h_idx * sy_h + w_idx * sy_w
    tl.store(y_ptrs, result, mask=mask)


# ─── Kernel 3: Channel Shuffle (in-place view+transpose+view) ────────────────
# Input: NCHW [B, C, H, W], groups=3, C=120
# Output: NCHW [B, C, H, W] with channels shuffled
# shuffle: view(B, groups, C//groups, H, W) -> transpose(1,2) -> view(B, C, H, W)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
    ],
    key=['N_ELEM'],
)
@triton.jit
def channel_shuffle_kernel(
    X_ptr, Y_ptr,
    B, C, H, W,
    GROUPS: tl.constexpr,
    N_ELEM,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEM

    # decode output index: (n, c_out, h, w)
    w_idx = offs % W
    h_idx = (offs // W) % H
    c_out = (offs // (W * H)) % C
    n_idx = offs // (W * H * C)

    # c_out in shuffled layout corresponds to:
    # original: view(B, groups, C//groups, H, W)[n, g, c_per_g, h, w]
    # after transpose(1,2): [n, c_per_g, g, h, w]
    # c_out = c_per_g * groups + g
    # so: g = c_out % groups, c_per_g = c_out // groups
    # original channel = g * (C // groups) + c_per_g
    channels_per_group = C // GROUPS
    g = c_out % GROUPS
    c_per_g = c_out // GROUPS
    c_in = g * channels_per_group + c_per_g

    x_ptrs = X_ptr + n_idx * (C * H * W) + c_in * (H * W) + h_idx * W + w_idx
    x_val = tl.load(x_ptrs, mask=mask, other=0.0)

    y_ptrs = Y_ptr + n_idx * (C * H * W) + c_out * (H * W) + h_idx * W + w_idx
    tl.store(y_ptrs, x_val, mask=mask)


# ─── Kernel 4: Group Conv 1x1 (120->480, groups=3) + BN + ReLU ───────────────
# Reuses group_conv1x1_bn_relu_kernel

# ─── Kernel 5: Shortcut Conv 1x1 (240->480, groups=1) + BN ───────────────────
# Also reuses group_conv1x1_bn_relu_kernel with GROUPS=1

# ─── Kernel 6: Add shortcut (in-place) ───────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4, num_stages=3),
    ],
    key=['N_ELEM'],
)
@triton.jit
def add_inplace_kernel(
    A_ptr, B_ptr,
    N_ELEM,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_ELEM
    a = tl.load(A_ptr + offs, mask=mask, other=0.0)
    b = tl.load(B_ptr + offs, mask=mask, other=0.0)
    tl.store(A_ptr + offs, a + b, mask=mask)


def kernel_function(
    p_conv1_weight, p_bn1_weight, p_bn1_bias,
    p_conv2_weight, p_bn2_weight, p_bn2_bias,
    p_conv3_weight, p_bn3_weight, p_bn3_bias,
    p_shortcut_0_weight, p_shortcut_1_weight, p_shortcut_1_bias,
    b_bn1_running_mean, b_bn1_running_var,
    b_bn2_running_mean, b_bn2_running_var,
    b_bn3_running_mean, b_bn3_running_var,
    b_shortcut_1_running_mean, b_shortcut_1_running_var,
):
    BATCH, IN_C, H, W = x.shape  # 10, 240, 224, 224
    GROUPS = 3
    MID_C = 120   # out of conv1
    OUT_C = 480   # out of conv3
    EPS = 1e-5

    device = x.device
    dtype = x.dtype

    # ── Step 1: Convert x to NHWC for group conv1x1 ──────────────────────────
    # x is NCHW [10, 240, 224, 224]
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [10, 224, 224, 240]

    # ── Step 2: Group Conv1 (240->120, groups=3) + BN1 + ReLU ─────────────────
    out1 = torch.empty((BATCH, MID_C, H, W), device=device, dtype=dtype)

    M1 = BATCH * H * W
    GROUP_IN_C1 = IN_C // GROUPS   # 80
    GROUP_OUT_C1 = MID_C // GROUPS  # 40

    # Reshape weight for kernel: [OUT_C, GROUP_IN_C] = [120, 80]
    w1 = p_conv1_weight.reshape(MID_C, GROUP_IN_C1)  # [120, 80]

    grid1 = lambda meta: (
        triton.cdiv(M1, meta['BLOCK_M']),
        triton.cdiv(GROUP_OUT_C1, meta['BLOCK_N']),
        GROUPS,
    )
    group_conv1x1_bn_relu_kernel[grid1](
        x_nhwc, w1,
        b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias,
        out1,
        BATCH, IN_C, MID_C, H, W,
        GROUPS,
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w1.stride(0), w1.stride(1),
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        M1,
        GROUP_IN_C1,
        GROUP_OUT_C1,
        EPS,
    )

    # ── Step 3: Depthwise Conv2 (120->120, 3x3) + BN2 ─────────────────────────
    out2 = torch.empty((BATCH, MID_C, H, W), device=device, dtype=dtype)

    # Weight shape: [120, 1, 3, 3] -> reshape to [120, 9]
    w2 = p_conv2_weight.reshape(MID_C, 9).contiguous()

    M2 = BATCH * MID_C * H * W
    grid2 = lambda meta: (triton.cdiv(M2, meta['BLOCK_M']),)
    depthwise_conv3x3_bn_kernel[grid2](
        out1, w2,
        b_bn2_running_mean, b_bn2_running_var, p_bn2_weight, p_bn2_bias,
        out2,
        BATCH, MID_C, H, W,
        out1.stride(0), out1.stride(1), out1.stride(2), out1.stride(3),
        out2.stride(0), out2.stride(1), out2.stride(2), out2.stride(3),
        M2,
        EPS,
    )

    # ── Step 4: Channel Shuffle ───────────────────────────────────────────────
    out_shuffle = torch.empty_like(out2)
    N_ELEM_S = BATCH * MID_C * H * W
    grid_s = lambda meta: (triton.cdiv(N_ELEM_S, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid_s](
        out2, out_shuffle,
        BATCH, MID_C, H, W,
        GROUPS,
        N_ELEM_S,
    )

    # ── Step 5: Group Conv3 (120->480, groups=3) + BN3 + ReLU ─────────────────
    out3 = torch.empty((BATCH, OUT_C, H, W), device=device, dtype=dtype)

    # Convert shuffled output to NHWC
    out_shuffle_nhwc = out_shuffle.permute(0, 2, 3, 1).contiguous()

    GROUP_IN_C3 = MID_C // GROUPS   # 40
    GROUP_OUT_C3 = OUT_C // GROUPS  # 160

    # Weight: [480, 40, 1, 1] -> [480, 40]
    w3 = p_conv3_weight.reshape(OUT_C, GROUP_IN_C3)

    grid3 = lambda meta: (
        triton.cdiv(M1, meta['BLOCK_M']),
        triton.cdiv(GROUP_OUT_C3, meta['BLOCK_N']),
        GROUPS,
    )
    group_conv1x1_bn_relu_kernel[grid3](
        out_shuffle_nhwc, w3,
        b_bn3_running_mean, b_bn3_running_var, p_bn3_weight, p_bn3_bias,
        out3,
        BATCH, MID_C, OUT_C, H, W,
        GROUPS,
        out_shuffle_nhwc.stride(0), out_shuffle_nhwc.stride(1),
        out_shuffle_nhwc.stride(2), out_shuffle_nhwc.stride(3),
        w3.stride(0), w3.stride(1),
        out3.stride(0), out3.stride(1), out3.stride(2), out3.stride(3),
        M1,
        GROUP_IN_C3,
        GROUP_OUT_C3,
        EPS,
    )

    # ── Step 6: Shortcut Conv (240->480, groups=1) + BN ───────────────────────
    shortcut_out = torch.empty((BATCH, OUT_C, H, W), device=device, dtype=dtype)

    GROUP_IN_C_SC = IN_C   # 240
    GROUP_OUT_C_SC = OUT_C  # 480

    # Weight: [480, 240, 1, 1] -> [480, 240]
    w_sc = p_shortcut_0_weight.reshape(OUT_C, IN_C)

    grid_sc = lambda meta: (
        triton.cdiv(M1, meta['BLOCK_M']),
        triton.cdiv(GROUP_OUT_C_SC, meta['BLOCK_N']),
        1,
    )
    group_conv1x1_bn_relu_kernel[grid_sc](
        x_nhwc, w_sc,
        b_shortcut_1_running_mean, b_shortcut_1_running_var,
        p_shortcut_1_weight, p_shortcut_1_bias,
        shortcut_out,
        BATCH, IN_C, OUT_C, H, W,
        1,  # GROUPS=1
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        w_sc.stride(0), w_sc.stride(1),
        shortcut_out.stride(0), shortcut_out.stride(1),
        shortcut_out.stride(2), shortcut_out.stride(3),
        M1,
        GROUP_IN_C_SC,
        GROUP_OUT_C_SC,
        EPS,
        False,  # DO_RELU=False
    )

    # ── Step 7: Add shortcut ──────────────────────────────────────────────────
    N_ELEM_ADD = BATCH * OUT_C * H * W
    grid_add = lambda meta: (triton.cdiv(N_ELEM_ADD, meta['BLOCK_SIZE']),)
    add_inplace_kernel[grid_add](
        out3, shortcut_out,
        N_ELEM_ADD,
    )

    return out3


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

        import torch.nn as nn
        import torch.nn.functional as F

        # Reference computation
        def ref_forward(x):
            # Conv1 + BN1 + ReLU
            # group conv: 240->120, groups=3
            # weight: [120, 80, 1, 1]
            out = F.conv2d(x, p_conv1_weight, bias=None, stride=1, padding=0, groups=3)
            out = F.batch_norm(out, b_bn1_running_mean, b_bn1_running_var,
                               p_bn1_weight, p_bn1_bias, training=False, eps=1e-5)
            out = F.relu(out)

            # Conv2 + BN2 (depthwise)
            out = F.conv2d(out, p_conv2_weight, bias=None, stride=1, padding=1, groups=120)
            out = F.batch_norm(out, b_bn2_running_mean, b_bn2_running_var,
                               p_bn2_weight, p_bn2_bias, training=False, eps=1e-5)

            # Channel shuffle
            B, C, H, W = out.shape
            out = out.view(B, 3, C // 3, H, W)
            out = out.transpose(1, 2).contiguous()
            out = out.view(B, C, H, W)

            # Conv3 + BN3 + ReLU
            out = F.conv2d(out, p_conv3_weight, bias=None, stride=1, padding=0, groups=3)
            out = F.batch_norm(out, b_bn3_running_mean, b_bn3_running_var,
                               p_bn3_weight, p_bn3_bias, training=False, eps=1e-5)
            out = F.relu(out)

            # Shortcut
            sc = F.conv2d(x, p_shortcut_0_weight, bias=None, stride=1, padding=0, groups=1)
            sc = F.batch_norm(sc, b_shortcut_1_running_mean, b_shortcut_1_running_var,
                              p_shortcut_1_weight, p_shortcut_1_bias, training=False, eps=1e-5)

            out = out + sc
            return out

        ref = ref_forward(x)
        torch.cuda.synchronize()

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1