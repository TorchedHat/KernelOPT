import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─── Kernel 1: Conv3d implicit GEMM ───────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1024, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'IN_C'],
)
@triton.jit
def conv3d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_D, IN_H, IN_W,
    OUT_C, OUT_D, OUT_H, OUT_W,
    KD, KH, KW,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wd, stride_wh, stride_ww,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    M_TOTAL,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: batch * OUT_D * OUT_H * OUT_W + d * OUT_H * OUT_W + h * OUT_W + w
    tmp = offs_m
    out_w = tmp % OUT_W
    tmp = tmp // OUT_W
    out_h = tmp % OUT_H
    tmp = tmp // OUT_H
    out_d = tmp % OUT_D
    batch = tmp // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KD * KH * KW
    K_BLOCKS = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_block in range(K_BLOCKS):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)
        # Decode K: in_c * KD * KH * KW + kd * KH * KW + kh * KW + kw
        tmp_k = offs_k
        kw = tmp_k % KW
        tmp_k = tmp_k // KW
        kh = tmp_k % KH
        tmp_k = tmp_k // KH
        kd = tmp_k % KD
        in_c = tmp_k // KD

        # Input coordinates
        id_ = out_d[:, None] * 1 + kd[None, :]  # stride=1, pad=0
        ih = out_h[:, None] * 1 + kh[None, :]
        iw = out_w[:, None] * 1 + kw[None, :]

        valid_m = (offs_m < M_TOTAL)
        valid_k = (offs_k < TOTAL_K)
        valid_n = (offs_n < OUT_C)

        x_valid = (valid_m[:, None] & valid_k[None, :] &
                   (id_ >= 0) & (id_ < IN_D) &
                   (ih >= 0) & (ih < IN_H) &
                   (iw >= 0) & (iw < IN_W) &
                   (batch[:, None] < BATCH))

        x_ptrs = (X_ptr +
                  batch[:, None] * stride_xn +
                  in_c[None, :] * stride_xc +
        x_tile = tl.load(x_ptrs, mask=x_valid, other=0.0)

        w_ptrs = (W_ptr +
                  offs_n[None, :] * stride_wco +
                  in_c[:, None] * stride_wci +
                  kd[:, None] * stride_wd +
                  kh[:, None] * stride_wh +
                  kw[:, None] * stride_ww)
        w_valid = valid_k[:, None] & valid_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_valid, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=(offs_n < OUT_C), other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr +
              batch[:, None] * stride_yn +
              offs_n[None, :] * stride_yc +
              out_d[:, None] * stride_yd +
              out_h[:, None] * stride_yh +
              out_w[:, None] * stride_yw)
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


# ─── Kernel 2: InstanceNorm stats (mean + var) per (batch, channel) ──────────
# Each program handles one (batch, channel) pair, reducing over D*H*W = 12600
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_N': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_N': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_N': 2048}, num_warps=8, num_stages=4),
    ],
    key=['DHW'],
)
@triton.jit
def instance_norm_stats_kernel(
    BATCH, C, DHW,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // C
    c = pid % C

    base = batch * C * DHW + c * DHW

    mean_acc = tl.zeros((), dtype=tl.float32)
    var_acc = tl.zeros((), dtype=tl.float32)

    for off in range(0, DHW, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < DHW
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        mean_acc += tl.sum(x, axis=0)

    mean_val = mean_acc / DHW
    tl.store(Mean_ptr + pid, mean_val)

    for off in range(0, DHW, BLOCK_N):
        offs = off + tl.arange(0, BLOCK_N)
        mask = offs < DHW
        x = tl.load(X_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
        diff = x - mean_val
        var_acc += tl.sum(diff * diff, axis=0)

    var_val = var_acc / DHW
    tl.store(Var_ptr + pid, var_val)


# ─── Kernel 3: Normalize + clamp + mul + max over channels ───────────────────
# Each program handles one (batch, d, h, w) position, reducing over C=16
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_X': 64, 'BLOCK_C': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_X': 128, 'BLOCK_C': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_X': 256, 'BLOCK_C': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_X': 512, 'BLOCK_C': 16}, num_warps=8, num_stages=2),
    ],
    key=['TOTAL_SPATIAL', 'C'],
)
@triton.jit
def norm_clamp_max_kernel(
    BATCH, C, DHW,
    CLAMP_MIN: tl.constexpr,
    CLAMP_MAX: tl.constexpr,
    BLOCK_X: tl.constexpr,
    BLOCK_C: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_x = pid * BLOCK_X + tl.arange(0, BLOCK_X)
    # offs_x indexes into (batch, dhw) flattened
    batch = offs_x // DHW
    dhw = offs_x % DHW

    valid_x = offs_x < TOTAL_SPATIAL

    # We reduce over C channels
    max_val = tl.full((BLOCK_X,), float('-inf'), dtype=tl.float32)
    max_idx = tl.zeros((BLOCK_X,), dtype=tl.int64)

    for c in range(C):
        # Load multiplier
        mult = tl.load(Mult_ptr + c)
        # Load mean and var for this (batch, c)
        mean_idx = batch * C + c
        mean_val = tl.load(Mean_ptr + mean_idx, mask=valid_x, other=0.0)
        var_val = tl.load(Var_ptr + mean_idx, mask=valid_x, other=0.0)

        # Load conv output: X[batch, c, dhw]
        x_idx = batch * C * DHW + c * DHW + dhw
        x = tl.load(X_ptr + x_idx, mask=valid_x, other=0.0).to(tl.float32)

        # Apply multiplier (first mul)
        x = x * mult

        # Instance norm: (x - mean) / sqrt(var + eps)
        inv_std = libdevice.rsqrt(var_val + 1e-5)
        x = (x - mean_val) * inv_std

        # Clamp
        x = tl.maximum(x, CLAMP_MIN)
        x = tl.minimum(x, CLAMP_MAX)

        # Second mul by multiplier
        x = x * mult

        # Update max
        is_greater = x > max_val
        max_val = tl.where(is_greater, x, max_val)
        max_idx = tl.where(is_greater, tl.full((BLOCK_X,), c, dtype=tl.int64), max_idx)

    # Store results
    tl.store(ValOut_ptr + offs_x, max_val, mask=valid_x)
    tl.store(IdxOut_ptr + offs_x, max_idx, mask=valid_x)


def kernel_function(p_conv_weight, p_conv_bias, p_multiplier, x):
    BATCH, IN_C, IN_D, IN_H, IN_W = x.shape
    OUT_C, _, KD, KH, KW = p_conv_weight.shape
    OUT_D = IN_D - KD + 1
    OUT_H = IN_H - KH + 1
    OUT_W = IN_W - KW + 1
    DHW = OUT_D * OUT_H * OUT_W
    M_TOTAL = BATCH * DHW

    # Allocate conv output
    conv_out = torch.empty((BATCH, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=torch.float32)

    # Launch conv kernel
    grid_conv = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )
    conv3d_kernel[grid_conv](
        x, p_conv_weight, p_conv_bias, conv_out,
        BATCH, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        p_conv_weight.stride(0), p_conv_weight.stride(1), p_conv_weight.stride(2),
        p_conv_weight.stride(3), p_conv_weight.stride(4),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        conv_out.stride(3), conv_out.stride(4),
        M_TOTAL,
    )

    # Flatten multiplier to (C,)
    mult_flat = p_multiplier.view(OUT_C)

    # Compute per-(batch,channel) stats on conv_out (before multiplier for instance norm)
    # But wait: instance norm is applied AFTER the first mul by multiplier.
    # So we need stats of (conv_out * multiplier).
    # We can compute stats on conv_out * multiplier directly.
    # Let's create a temp buffer for conv_out * multiplier
    # Actually, we can fold the multiplier into the stats computation.
    # For efficiency, let's compute stats on the fly in the stats kernel.
    # We'll pass conv_out and mult separately and compute x*mult inside.

    # Allocate stats buffers
    mean_buf = torch.empty((BATCH, OUT_C), device=x.device, dtype=torch.float32)
    var_buf = torch.empty((BATCH, OUT_C), device=x.device, dtype=torch.float32)

    # We need a modified stats kernel that applies multiplier first
    # Let's use a separate approach: create scaled buffer
    # conv_out_scaled = conv_out * mult_flat[None, :, None, None, None]
    # Then compute stats on conv_out_scaled
    # This avoids an extra kernel by reusing instance_norm_stats_kernel with scaled data

    # For simplicity, create scaled buffer
    conv_scaled = conv_out * mult_flat.view(1, OUT_C, 1, 1, 1)

    # Stats kernel: one program per (batch, channel)
    grid_stats = (BATCH * OUT_C,)
    instance_norm_stats_kernel[grid_stats](
        conv_scaled,
        mean_buf, var_buf,
        BATCH, OUT_C, DHW,
    )

    # Allocate outputs
    val_out = torch.empty((BATCH, DHW), device=x.device, dtype=torch.float32)
    idx_out = torch.empty((BATCH, DHW), device=x.device, dtype=torch.int64)

    TOTAL_SPATIAL = BATCH * DHW
    grid_max = lambda meta: (triton.cdiv(TOTAL_SPATIAL, meta['BLOCK_X']),)

    norm_clamp_max_kernel[grid_max](
        conv_scaled,
        mult_flat,
        mean_buf, var_buf,
        val_out, idx_out,
        BATCH, OUT_C, DHW,
        -1.0, 1.0,
        TOTAL_SPATIAL,
    )

    # Reshape outputs to (BATCH, OUT_D, OUT_H, OUT_W)
    val_out = val_out.view(BATCH, OUT_D, OUT_H, OUT_W)
    idx_out = idx_out.view(BATCH, OUT_D, OUT_H, OUT_W)

    return val_out, idx_out


def get_inputs():
    batch_size = 128
    in_channels = 3
    out_channels = 16
    depth, height, width = 16, 32, 32
    kernel_size = 3

    p_conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size, device='cuda')
    p_conv_bias = torch.randn(out_channels, device='cuda')
    p_multiplier = torch.randn(out_channels, 1, 1, 1, device='cuda')
    x = torch.rand(batch_size, in_channels, depth, height, width, device='cuda')
    return [p_conv_weight, p_conv_bias, p_multiplier, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_conv_weight, p_conv_bias, p_multiplier, x = inputs

    val_out, idx_out = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        # Reference computation
        conv = nn.Conv3d(3, 16, 3, bias=True).cuda()
        with torch.no_grad():
            conv.weight.copy_(p_conv_weight)
            conv.bias.copy_(p_conv_bias)

        inst_norm = nn.InstanceNorm3d(16).cuda()

        with torch.no_grad():
            ref_x = conv(x)
            ref_x = ref_x * p_multiplier
            ref_x = inst_norm(ref_x)
            ref_x = torch.clamp(ref_x, -1.0, 1.0)
            ref_x = ref_x * p_multiplier
            ref_val, ref_idx = torch.max(ref_x, dim=1)

        print(f"val_out shape: {val_out.shape}, ref shape: {ref_val.shape}")
        print(f"Max diff val: {(val_out - ref_val).abs().max().item():.4e}")
        print(f"Max diff idx: {(idx_out - ref_idx).abs().max().item():.4e}")

        assert torch.allclose(val_out, ref_val, rtol=1e-3, atol=1e-5), \
            f"Values don't match! max_diff={(val_out - ref_val).abs().max().item():.4e}"
        assert torch.all(idx_out == ref_idx), \
            f"Indices don't match! max_diff={(idx_out - ref_idx).abs().max().item():.4e}"
        print("Kernel executed successfully.")