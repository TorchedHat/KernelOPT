import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


# ─── Kernel 1: Conv3d as implicit GEMM ───────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv3d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_D, IN_H, IN_W,
    OUT_C, OUT_D, OUT_H, OUT_W,
    KD, KH, KW,
    stride_xn, stride_xc, stride_xd, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkd, stride_wkh, stride_wkw,
    stride_yn, stride_yc, stride_yd, stride_yh, stride_yw,
    M_TOTAL, TOTAL_K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: batch, out_d, out_h, out_w
    tmp = offs_m
    out_w_idx = tmp % OUT_W
    tmp = tmp // OUT_W
    out_h_idx = tmp % OUT_H
    tmp = tmp // OUT_H
    out_d_idx = tmp % OUT_D
    batch_idx = tmp // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K: in_c, kd, kh, kw
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KW
        tmp_k = tmp_k // KW
        kh_idx = tmp_k % KH
        kd_idx = tmp_k // KH

        # Input spatial coords
        in_d = out_d_idx[:, None] + kd_idx[None, :]
        in_h = out_h_idx[:, None] + kh_idx[None, :]
        in_w = out_w_idx[:, None] + kw_idx[None, :]

        valid_m = (offs_m < M_TOTAL)[:, None]
        valid_k = (offs_k < TOTAL_K)[None, :]
        valid_n = (offs_n < OUT_C)[None, :]

        valid_x = (
            (batch_idx[:, None] < BATCH) &
            (in_d >= 0) & (in_d < IN_D) &
            (in_h >= 0) & (in_h < IN_H) &
            (in_w >= 0) & (in_w < IN_W)
        )

        x_ptrs = (
            batch_idx[:, None] * stride_xn +
            in_c_idx[None, :] * stride_xc +
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (
            offs_n[None, :] * stride_wco +
            in_c_idx[:, None] * stride_wci +
            kd_idx[:, None] * stride_wkd +
            kh_idx[:, None] * stride_wkh +
            kw_idx[:, None] * stride_wkw
        )
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=(offs_n < OUT_C), other=0.0)
    acc = acc + bias[None, :]

    # Store
    valid_store = (
        (offs_m < M_TOTAL)[:, None] &
        (offs_n < OUT_C)[None, :] &
        (batch_idx[:, None] < BATCH) &
        (out_d_idx[:, None] < OUT_D) &
        (out_h_idx[:, None] < OUT_H) &
        (out_w_idx[:, None] < OUT_W)
    )
    y_ptrs = (
        batch_idx[:, None] * stride_yn +
        offs_n[None, :] * stride_yc +
        out_d_idx[:, None] * stride_yd +
        out_h_idx[:, None] * stride_yh +
        out_w_idx[:, None] * stride_yw
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_store)


# ─── Kernel 2: Fused GroupNorm + Mean ────────────────────────────────────────
# Each program handles one (batch, group) pair.
# Reduction over (channels_per_group * D * H * W) = 3 * 22 * 30 * 30 = 59400
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_R': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_R': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_R': 256},  num_warps=4,  num_stages=2),
    ],
    key=['R_SIZE', 'NUM_GROUPS'],
)
@triton.jit
def groupnorm_mean_kernel(
    BATCH,
    NUM_GROUPS,
    CHANNELS_PER_GROUP,
    OUT_D, OUT_H, OUT_W,
    R_SIZE,      # = CHANNELS_PER_GROUP * OUT_D * OUT_H * OUT_W
    TOTAL_ELEMENTS,  # = NUM_GROUPS * R_SIZE (per batch)
    EPS: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # program_id(0) = batch * NUM_GROUPS + group
    pid = tl.program_id(0)
    batch = pid // NUM_GROUPS
    group = pid % NUM_GROUPS

    # Compute mean and variance over R_SIZE elements
    mean_acc = tl.zeros((), dtype=tl.float32)
    var_acc  = tl.zeros((), dtype=tl.float32)

    # First pass: compute mean
    for r_start in range(0, R_SIZE, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R_SIZE

        # Decode r into (c_local, d, h, w)
        dhw = offs_r % (OUT_D * OUT_H * OUT_W)
        c_local = offs_r // (OUT_D * OUT_H * OUT_W)
        w_idx = dhw % OUT_W
        dh = dhw // OUT_W
        h_idx = dh % OUT_H
        d_idx = dh // OUT_H

        c_global = group * CHANNELS_PER_GROUP + c_local

        # Y layout: (BATCH, 24, OUT_D, OUT_H, OUT_W)
        # stride: (24*OUT_D*OUT_H*OUT_W, OUT_D*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1)
        y_idx = (
        )
        val = tl.load(Y_ptr + y_idx, mask=mask_r, other=0.0)
        mean_acc += tl.sum(val, axis=0)

    mean_val = mean_acc / R_SIZE

    # Second pass: compute variance
    for r_start in range(0, R_SIZE, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R_SIZE

        dhw = offs_r % (OUT_D * OUT_H * OUT_W)
        c_local = offs_r // (OUT_D * OUT_H * OUT_W)
        w_idx = dhw % OUT_W
        dh = dhw // OUT_W
        h_idx = dh % OUT_H
        d_idx = dh // OUT_H

        c_global = group * CHANNELS_PER_GROUP + c_local

        y_idx = (
        )
        val = tl.load(Y_ptr + y_idx, mask=mask_r, other=0.0)
        diff = val - mean_val
        var_acc += tl.sum(diff * diff, axis=0)

    var_val = var_acc / R_SIZE
    inv_std = libdevice.rsqrt(var_val + EPS)

    # Third pass: normalize, apply scale/bias, accumulate mean for output
    sum_acc = tl.zeros((), dtype=tl.float32)

    for r_start in range(0, R_SIZE, BLOCK_R):
        offs_r = r_start + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R_SIZE

        dhw = offs_r % (OUT_D * OUT_H * OUT_W)
        c_local = offs_r // (OUT_D * OUT_H * OUT_W)
        w_idx = dhw % OUT_W
        dh = dhw // OUT_W
        h_idx = dh % OUT_H
        d_idx = dh // OUT_H

        c_global = group * CHANNELS_PER_GROUP + c_local

        y_idx = (
        )
        val = tl.load(Y_ptr + y_idx, mask=mask_r, other=0.0)

        gn_w = tl.load(GN_W_ptr + c_global, mask=(c_global < NUM_GROUPS * CHANNELS_PER_GROUP), other=1.0)
        gn_b = tl.load(GN_B_ptr + c_global, mask=(c_global < NUM_GROUPS * CHANNELS_PER_GROUP), other=0.0)

        normalized = (val - mean_val) * inv_std
        out_val = normalized * gn_w + gn_b
        sum_acc += tl.sum(tl.where(mask_r, out_val, 0.0), axis=0)

    # Atomic add to output (multiple groups contribute to same batch output)
    tl.atomic_add(OUT_ptr + batch, sum_acc / TOTAL_ELEMENTS)


def kernel_function(p_conv_weight, p_conv_bias, p_group_norm_weight, p_group_norm_bias, x):
    BATCH, IN_C, IN_D, IN_H, IN_W = x.shape
    OUT_C, _, KD, KH, KW = p_conv_weight.shape
    OUT_D = IN_D - KD + 1
    OUT_H = IN_H - KH + 1
    OUT_W = IN_W - KW + 1
    M_TOTAL = BATCH * OUT_D * OUT_H * OUT_W
    TOTAL_K = IN_C * KD * KH * KW

    # Allocate conv output
    conv_out = torch.empty((BATCH, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    # Launch conv kernel
    def cdiv(a, b): return (a + b - 1) // b
    grid_conv = (cdiv(M_TOTAL, 128), cdiv(OUT_C, 32))

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
        M_TOTAL, TOTAL_K,
    )

    # GroupNorm + Mean
    NUM_GROUPS = 8
    CHANNELS_PER_GROUP = OUT_C // NUM_GROUPS  # 3
    R_SIZE = CHANNELS_PER_GROUP * OUT_D * OUT_H * OUT_W  # 3 * 22 * 30 * 30 = 59400
    TOTAL_ELEMENTS = OUT_C * OUT_D * OUT_H * OUT_W  # 24 * 22 * 30 * 30 = 475200

    output = torch.zeros(BATCH, device=x.device, dtype=x.dtype)

    grid_gn = (BATCH * NUM_GROUPS,)
    groupnorm_mean_kernel[grid_gn](
        conv_out,
        p_group_norm_weight, p_group_norm_bias,
        output,
        BATCH, NUM_GROUPS, CHANNELS_PER_GROUP,
        OUT_D, OUT_H, OUT_W,
        R_SIZE, TOTAL_ELEMENTS,
        1e-5,
    )

    return output


def get_inputs():
    batch_size = 128
    in_channels = 3
    out_channels = 24
    D, H, W = 24, 32, 32
    kernel_size = 3
    num_groups = 8

    p_conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
    p_conv_bias = torch.randn(out_channels)
    p_group_norm_weight = torch.randn(out_channels)
    p_group_norm_bias = torch.randn(out_channels)
    x = torch.rand(batch_size, in_channels, D, H, W)

    return [p_conv_weight, p_conv_bias, p_group_norm_weight, p_group_norm_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]

    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        p_conv_weight, p_conv_bias, p_group_norm_weight, p_group_norm_bias, x = inputs_cuda

        # Reference computation
        conv = nn.Conv3d(3, 24, 3, bias=True).cuda()
        conv.weight.data = p_conv_weight
        conv.bias.data = p_conv_bias

        gn = nn.GroupNorm(8, 24).cuda()
        gn.weight.data = p_group_norm_weight
        gn.bias.data = p_group_norm_bias

        with torch.no_grad():
            ref_conv = conv(x)
            ref_gn = gn(ref_conv)
            ref_out = ref_gn.mean(dim=[1, 2, 3, 4])

        print(f"Output shape: {output.shape}")
        print(f"Output[:5]: {output[:5]}")
        print(f"Ref[:5]:    {ref_out[:5]}")
        max_diff = (output - ref_out).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")

        assert torch.allclose(output, ref_out, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")