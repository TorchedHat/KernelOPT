import torch
import triton
import triton.language as tl
import math

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Fused BN + ReLU (pointwise, applied after conv1)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # channel index: layout is (N, C, H, W) stored as (N, C, H, W) contiguous
    # offsets // (H*W) % C  -- but we pass C and spatial = numel // (N*C)
    # We encode channel as: c = (offset // spatial) % C
    # But we don't have spatial here directly; we compute from numel and C
    # Actually numel = N*C*H*W, spatial = H*W = numel // (N*C)
    # c = (offset // spatial) % C  = (offset * C // numel) ... tricky
    # Simpler: c = (offset // (numel // (N*C))) % C
    # We'll pass spatial = H*W as a separate arg... let's use c_stride
    # Actually layout after conv is (N, C, H, W) contiguous, so:
    # linear index = n*C*H*W + c*H*W + h*W + w
    # c = (offset // HW) % C  where HW = numel / (N*C) ... need HW
    # Let's just pass HW
    # We'll restructure: pass spatial (H*W)
    # For now use the trick: c = (offset // spatial) % C
    # We pass spatial as a constexpr... no, it varies. Pass as arg.
    # Actually let's just pass it. See wrapper.
    pass


# Cleaner approach: pass spatial size
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_relu_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, spatial, C,
    eps: tl.constexpr,
    do_relu: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)

    # Compute channel index
    c = (offsets // spatial) % C

    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var  = tl.load(var_ptr  + c, mask=mask, other=0.0)
    w    = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b    = tl.load(bias_ptr   + c, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    if do_relu:
        out = tl.maximum(out, 0.0)

    tl.store(out_ptr + offsets, out, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Channel Shuffle (groups=3, in-place reshape+transpose)
# Input layout: (N, C, H, W) contiguous, C=120, groups=3
# Output: same shape but channels shuffled
# shuffle: view(N, groups, C//groups, H, W) -> transpose(1,2) -> view(N, C, H, W)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def channel_shuffle_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    groups,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Decode output index (N, C, H, W) contiguous
    HW = H * W
    CHW = C * HW
    n   = offsets // CHW
    rem = offsets % CHW
    c   = rem // HW
    rem2 = rem % HW
    h   = rem2 // W
    w   = rem2 % W

    # In the shuffled output, channel c came from:
    # output[n, c, h, w] = input[n, c_in, h, w]
    # where c = (c_in // cpg) + (c_in % cpg) * groups  ... no
    # shuffle: view(N, G, cpg, H, W).transpose(1,2) -> (N, cpg, G, H, W) -> view(N, C, H, W)
    # So output channel c corresponds to:
    #   cpg_idx = c // G, g_idx = c % G  (in transposed view)
    #   original: g_idx * cpg + cpg_idx
    cpg = C // groups  # channels per group
    g_idx   = c % groups
    cpg_idx = c // groups
    c_in = g_idx * cpg + cpg_idx

    src_offset = n * CHW + c_in * HW + h * W + w
    src_mask = mask & (n < N) & (c_in < C) & (h < H) & (w < W)

    val = tl.load(x_ptr + src_offset, mask=src_mask, other=0.0)
    tl.store(out_ptr + offsets, val, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Fused BN + add (for final add with shortcut, no relu)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
    ],
    key=['numel'],
)
@triton.jit
def fused_bn_add_kernel(
    x_ptr, shortcut_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, spatial, C,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sc = tl.load(shortcut_ptr + offsets, mask=mask, other=0.0)

    c = (offsets // spatial) % C

    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var  = tl.load(var_ptr  + c, mask=mask, other=0.0)
    w    = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b    = tl.load(bias_ptr   + c, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b + sc

    tl.store(out_ptr + offsets, out, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Grouped Conv2d 1x1 kernel: (N, Cin, H, W) -> (N, Cout, H, W)
# groups=3, kernel=1x1
# We implement as GEMM: for each spatial position and group,
# multiply [Cin/G] input channels by [Cout/G x Cin/G] weight
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def grouped_conv1x1_kernel(
    X_ptr, W_ptr, Out_ptr,
    # X shape: (N, Cin, H, W) contiguous -> stride: (Cin*H*W, H*W, W, 1)
    # W shape: (Cout, Cin//G, 1, 1) -> (Cout, Cin//G)
    # Out shape: (N, Cout, H, W)
    N, Cin, H, W, Cout, G,
    M,   # = N * H * W (spatial positions)
    N_out,  # = Cout
    K,   # = Cin // G (per-group input channels)
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Each program handles one group, one tile of (M, Cout//G)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_g = tl.program_id(2)

    G_out = N_out // G  # Cout per group
    G_in  = K           # Cin per group

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)  # spatial positions
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)  # output channels within group

    # Decode spatial position
    HW = H * W
    n_idx = offs_m // HW
    hw    = offs_m % HW
    h_idx = hw // W
    w_idx = hw % W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, G_in, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)  # input channels within group

        # Load X tile: shape (BLOCK_M, BLOCK_K)
        # X[n, pid_g*G_in + k_off, h, w]
        # X stride: (Cin*HW, HW, W, 1)
        c_in = pid_g * G_in + offs_k  # absolute input channel
        x_ptrs = (
        )
        x_mask = (
            (offs_m[:, None] < M)
        )
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile: shape (BLOCK_K, BLOCK_N)
        # W[pid_g*G_out + offs_n, offs_k, 0, 0] -> W[(pid_g*G_out + offs_n)*G_in + offs_k]
        c_out_abs = pid_g * G_out + offs_n  # absolute output channel
        w_ptrs = (
        )
        w_mask = (offs_k[:, None] < G_in) & (offs_n[None, :] < G_out)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output: Out[n, pid_g*G_out + offs_n, h, w]
    c_out_abs = pid_g * G_out + offs_n
    out_ptrs = (
    )
    out_mask = (
        (offs_m[:, None] < M)
    )
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Depthwise Conv2d 3x3 kernel: (N, C, H, W) -> (N, C, H, W), padding=1
# Each output channel is independent
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_C': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_C': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_C': 1}, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_C': 1}, num_warps=2),
    ],
    key=['M_total', 'C'],
)
@triton.jit
def depthwise_conv3x3_kernel(
    X_ptr, W_ptr, Out_ptr,
    N, C, H, W,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    mask = offs < M_total

    HW = H * W
    CHW = C * HW

    n_idx = offs // CHW
    rem   = offs % CHW
    c_idx = rem // HW
    rem2  = rem % HW
    h_idx = rem2 // W
    w_idx = rem2 % W

    # Accumulate over 3x3 kernel
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = h_idx + kh - 1  # padding=1
            iw = w_idx + kw - 1

            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)

            x_ptrs = X_ptr + n_idx * CHW + c_idx * HW + ih * W + iw
            x_val = tl.load(x_ptrs, mask=valid, other=0.0)

            # W shape: (C, 1, 3, 3) -> W[c, 0, kh, kw] = W[c*9 + kh*3 + kw]
            w_ptrs = W_ptr + c_idx * 9 + kh * 3 + kw
            w_val = tl.load(w_ptrs, mask=mask, other=0.0)

            acc = acc + x_val * w_val

    tl.store(Out_ptr + offs, acc, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel function
# ─────────────────────────────────────────────────────────────────────────────
def kernel_function(
):
    device = x.device
    dtype  = x.dtype
    N, Cin, H, W = x.shape  # 10, 240, 224, 224
    eps = 1e-5

    # ── Step 1: Conv1 (group=3, 240->120, 1x1) ──────────────────────────────
    # p_conv1_weight: [120, 80, 1, 1] -> [120, 80] (Cout=120, Cin/G=80, G=3)
    G1 = 3
    Cout1 = 120
    Cin1  = 240
    K1    = Cin1 // G1  # 80
    M1    = N * H * W   # 10 * 224 * 224 = 501760

    conv1_out = torch.empty((N, Cout1, H, W), device=device, dtype=dtype)
    w1 = p_conv1_weight.reshape(Cout1, K1)  # [120, 80]

    grid1 = lambda meta: (
        triton.cdiv(M1, meta['BLOCK_M']),
        triton.cdiv(Cout1 // G1, meta['BLOCK_N']),
        G1,
    )
    grouped_conv1x1_kernel[grid1](
        x, w1, conv1_out,
        N, Cin1, H, W, Cout1, G1,
        M1, Cout1, K1,
    )

    # ── Step 2: BN1 + ReLU ─────────────────────────────────────────────────
    numel1 = N * Cout1 * H * W  # 60211200
    spatial1 = H * W  # 50176
    bn1_out = torch.empty_like(conv1_out)
    grid_bn1 = lambda meta: (triton.cdiv(numel1, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid_bn1](
        conv1_out, bn1_out,
        b_bn1_running_mean, b_bn1_running_var, p_bn1_weight, p_bn1_bias,
        numel1, spatial1, Cout1,
        eps, True,
    )

    # ── Step 3: Conv2 (depthwise, 120->120, 3x3, padding=1) ─────────────────
    C2 = 120
    conv2_out = torch.empty((N, C2, H, W), device=device, dtype=dtype)
    M2_total = N * C2 * H * W

    grid2 = lambda meta: (triton.cdiv(M2_total, meta['BLOCK_M']),)
    depthwise_conv3x3_kernel[grid2](
        bn1_out, p_conv2_weight, conv2_out,
        N, C2, H, W, M2_total,
    )

    # ── Step 4: BN2 (no relu) ──────────────────────────────────────────────
    numel2 = N * C2 * H * W
    spatial2 = H * W
    bn2_out = torch.empty_like(conv2_out)
    grid_bn2 = lambda meta: (triton.cdiv(numel2, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid_bn2](
        conv2_out, bn2_out,
        b_bn2_running_mean, b_bn2_running_var, p_bn2_weight, p_bn2_bias,
        numel2, spatial2, C2,
        eps, False,
    )

    # ── Step 5: Channel Shuffle (groups=3) ─────────────────────────────────
    shuffle_out = torch.empty_like(bn2_out)
    numel_sh = N * C2 * H * W
    grid_sh = lambda meta: (triton.cdiv(numel_sh, meta['BLOCK_SIZE']),)
    channel_shuffle_kernel[grid_sh](
        bn2_out, shuffle_out,
        N, C2, H, W, G1, numel_sh,
    )

    # ── Step 6: Conv3 (group=3, 120->480, 1x1) ──────────────────────────────
    G3    = 3
    Cout3 = 480
    Cin3  = 120
    K3    = Cin3 // G3  # 40
    M3    = N * H * W

    conv3_out = torch.empty((N, Cout3, H, W), device=device, dtype=dtype)
    w3 = p_conv3_weight.reshape(Cout3, K3)  # [480, 40]

    grid3 = lambda meta: (
        triton.cdiv(M3, meta['BLOCK_M']),
        triton.cdiv(Cout3 // G3, meta['BLOCK_N']),
        G3,
    )
    grouped_conv1x1_kernel[grid3](
        shuffle_out, w3, conv3_out,
        N, Cin3, H, W, Cout3, G3,
        M3, Cout3, K3,
    )

    # ── Step 7: BN3 + ReLU ─────────────────────────────────────────────────
    numel3 = N * Cout3 * H * W
    spatial3 = H * W
    bn3_out = torch.empty_like(conv3_out)
    grid_bn3 = lambda meta: (triton.cdiv(numel3, meta['BLOCK_SIZE']),)
    fused_bn_relu_kernel[grid_bn3](
        conv3_out, bn3_out,
        b_bn3_running_mean, b_bn3_running_var, p_bn3_weight, p_bn3_bias,
        numel3, spatial3, Cout3,
        eps, True,
    )

    # ── Step 8: Shortcut Conv (240->480, 1x1, no groups) ────────────────────
    G_sc  = 1
    Cout_sc = 480
    Cin_sc  = 240
    K_sc    = Cin_sc  # 240
    M_sc    = N * H * W

    shortcut_conv_out = torch.empty((N, Cout_sc, H, W), device=device, dtype=dtype)
    w_sc = p_shortcut_0_weight.reshape(Cout_sc, K_sc)  # [480, 240]

    grid_sc = lambda meta: (
        triton.cdiv(M_sc, meta['BLOCK_M']),
        triton.cdiv(Cout_sc, meta['BLOCK_N']),
        G_sc,
    )
    grouped_conv1x1_kernel[grid_sc](
        x, w_sc, shortcut_conv_out,
        N, Cin_sc, H, W, Cout_sc, G_sc,
        M_sc, Cout_sc, K_sc,
    )

    # ── Step 9: Shortcut BN + Add ──────────────────────────────────────────
    numel_sc = N * Cout_sc * H * W
    spatial_sc = H * W
    out = torch.empty_like(bn3_out)
    grid_add = lambda meta: (triton.cdiv(numel_sc, meta['BLOCK_SIZE']),)
    fused_bn_add_kernel[grid_add](
        shortcut_conv_out, bn3_out, out,
        b_shortcut_1_running_mean, b_shortcut_1_running_var,
        p_shortcut_1_weight, p_shortcut_1_bias,
        numel_sc, spatial_sc, Cout_sc,
        eps,
    )

    return out


def get_inputs():
    torch.manual_seed(42)
    device = 'cuda'
    return [
        torch.randn(120, 80, 1, 1, device=device),   # p_conv1_weight
        torch.randn(120, device=device),               # p_bn1_weight
        torch.randn(120, device=device),               # p_bn1_bias
        torch.randn(120, 1, 3, 3, device=device),     # p_conv2_weight
        torch.randn(120, device=device),               # p_bn2_weight
        torch.randn(120, device=device),               # p_bn2_bias
        torch.randn(480, 40, 1, 1, device=device),    # p_conv3_weight
        torch.randn(480, device=device),               # p_bn3_weight
        torch.randn(480, device=device),               # p_bn3_bias
        torch.randn(480, 240, 1, 1, device=device),   # p_shortcut_0_weight
        torch.randn(480, device=device),               # p_shortcut_1_weight
        torch.randn(480, device=device),               # p_shortcut_1_bias
        torch.randn(120, device=device),               # b_bn1_running_mean
        torch.abs(torch.randn(120, device=device)),    # b_bn1_running_var
        torch.randn(120, device=device),               # b_bn2_running_mean
        torch.abs(torch.randn(120, device=device)),    # b_bn2_running_var
        torch.randn(480, device=device),               # b_bn3_running_mean
        torch.abs(torch.randn(480, device=device)),    # b_bn3_running_var
        torch.randn(480, device=device),               # b_shortcut_1_running_mean
        torch.abs(torch.randn(480, device=device)),    # b_shortcut_1_running_var
        torch.randn(10, 240, 224, 224, device=device), # x
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()