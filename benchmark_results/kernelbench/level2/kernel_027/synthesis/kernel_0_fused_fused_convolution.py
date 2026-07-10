import torch
import triton
import triton.language as tl
import math

# ─── Kernel 1: Conv3D (implicit GEMM) ───────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1024, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=1),
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

    # Decode linear M-offset -> (batch, out_d, out_h, out_w)
    tmp = offs_m
    out_w = tmp % OUT_W
    tmp = tmp // OUT_W
    out_h = tmp % OUT_H
    tmp = tmp // OUT_H
    out_d = tmp % OUT_D
    batch = tmp // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KD * KH * KW
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_block in range(BLOCK_K_COUNT):
        k_off = k_block * BLOCK_K
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K-offset -> (in_c, kd, kh, kw)
        tmp_k = offs_k
        kw_idx = tmp_k % KW
        tmp_k = tmp_k // KW
        kh_idx = tmp_k % KH
        tmp_k = tmp_k // KH
        kd_idx = tmp_k % KD
        in_c = tmp_k // KD

        # Input coordinates
        id_ = out_d[:, None] + kd_idx[None, :]
        ih = out_h[:, None] + kh_idx[None, :]
        iw = out_w[:, None] + kw_idx[None, :]

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_spatial = (id_ >= 0) & (id_ < IN_D) & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_spatial

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Store
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ─── Kernel 2: GroupNorm stats (mean + var via Welford) ─────────────────────
# One program per (batch, group): computes mean and variance over
# (channels_per_group * D * H * W) elements after applying hardswish.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_R': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_R': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_R': 2048}, num_warps=8, num_stages=1),
    ],
    key=['R'],
)
@triton.jit
def groupnorm_stats_kernel(
    B, C, D, H, W,
    R,           # reduction size = (C//G) * D * H * W
    BLOCK_R: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (batch, group)
    batch = pid // G
    group = pid % G

    C_per_G = C // G
    spatial = D * H * W

    mean_acc = tl.zeros((), dtype=tl.float32)
    m2_acc = tl.zeros((), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)

    for r_off in range(0, R, BLOCK_R):
        offs_r = r_off + tl.arange(0, BLOCK_R)
        mask_r = offs_r < R

        # Decode r -> (c_local, spatial_idx)
        c_local = offs_r // spatial
        sp_idx = offs_r % spatial
        d_idx = sp_idx // (H * W)
        hw = sp_idx % (H * W)
        h_idx = hw // W
        w_idx = hw % W

        c_global = group * C_per_G + c_local

        x_ptrs = (X_ptr
        x = tl.load(x_ptrs, mask=mask_r & (c_local < C_per_G), other=0.0)

        # HardSwish: x * clamp(x+3, 0, 6) / 6
        x_hs = x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) * (1.0 / 6.0)

        # Welford online update
        x_hs = tl.where(mask_r & (c_local < C_per_G), x_hs, 0.0)
        valid_count = tl.sum(tl.where(mask_r & (c_local < C_per_G), tl.full(x_hs.shape, 1.0, tl.float32), 0.0).to(tl.float32))

        # Simple accumulation for mean/var
        x_sum = tl.sum(tl.where(mask_r & (c_local < C_per_G), x_hs, 0.0))
        x2_sum = tl.sum(tl.where(mask_r & (c_local < C_per_G), x_hs * x_hs, 0.0))

        mean_acc += x_sum
        m2_acc += x2_sum
        count += valid_count

    mean_val = mean_acc / count
    var_val = m2_acc / count - mean_val * mean_val

    tl.store(Mean_ptr + pid, mean_val)
    tl.store(Var_ptr + pid, var_val)


# ─── Kernel 3: GroupNorm normalize + mean pooling ───────────────────────────
# One program per (batch, channel): normalize then accumulate spatial mean.

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_S': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_S': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_S': 2048}, num_warps=8, num_stages=1),
    ],
    key=['SPATIAL'],
)
@triton.jit
def groupnorm_norm_mean_kernel(
    B, C, D, H, W,
    G,
    EPS,
    BLOCK_S: tl.constexpr,
):
    pid = tl.program_id(0)  # one program per (batch, channel)
    batch = pid // C
    channel = pid % C

    group = channel // (C // G)

    mean_val = tl.load(Mean_ptr + batch * G + group)
    var_val = tl.load(Var_ptr + batch * G + group)
    inv_std = 1.0 / tl.sqrt(var_val + EPS)
    gamma = tl.load(GN_W_ptr + channel)
    beta = tl.load(GN_B_ptr + channel)

    sum_acc = tl.zeros((), dtype=tl.float32)

    for s_off in range(0, SPATIAL, BLOCK_S):
        offs_s = s_off + tl.arange(0, BLOCK_S)
        mask_s = offs_s < SPATIAL

        d_idx = offs_s // (H * W)
        hw = offs_s % (H * W)
        h_idx = hw // W
        w_idx = hw % W

        x_ptrs = (X_ptr
        x = tl.load(x_ptrs, mask=mask_s, other=0.0)

        # HardSwish
        x_hs = x * tl.minimum(tl.maximum(x + 3.0, 0.0), 6.0) * (1.0 / 6.0)

        # Normalize
        x_norm = (x_hs - mean_val) * inv_std * gamma + beta

        sum_acc += tl.sum(tl.where(mask_s, x_norm, 0.0))

    mean_out = sum_acc / SPATIAL
    tl.store(Out_ptr + batch * C + channel, mean_out)


def kernel_function(conv_weight, conv_bias, gn_weight, gn_bias, x):
    B, IN_C, IN_D, IN_H, IN_W = x.shape
    OUT_C, _, KD, KH, KW = conv_weight.shape

    OUT_D = IN_D - KD + 1
    OUT_H = IN_H - KH + 1
    OUT_W = IN_W - KW + 1

    # Allocate conv output
    conv_out = torch.empty(B, OUT_C, OUT_D, OUT_H, OUT_W, device=x.device, dtype=x.dtype)

    M_TOTAL = B * OUT_D * OUT_H * OUT_W

    grid_conv = (
        triton.cdiv(M_TOTAL, 128),  # will be overridden by autotune
        triton.cdiv(OUT_C, 16),
    )

    # Use lambda to get autotune-selected BLOCK_M
    def get_grid(meta):
        return (triton.cdiv(M_TOTAL, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))

    conv3d_kernel[get_grid](
        x, conv_weight, conv_bias, conv_out,
        B, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        conv_weight.stride(0), conv_weight.stride(1), conv_weight.stride(2), conv_weight.stride(3), conv_weight.stride(4),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3), conv_out.stride(4),
        M_TOTAL,
    )

    # GroupNorm stats
    G = 4  # num_groups
    SPATIAL = OUT_D * OUT_H * OUT_W
    R = (OUT_C // G) * SPATIAL

    gn_mean = torch.empty(B, G, device=x.device, dtype=x.dtype)
    gn_var = torch.empty(B, G, device=x.device, dtype=x.dtype)

    groupnorm_stats_kernel[(B * G,)](
        conv_out, gn_mean, gn_var,
        B, OUT_C, OUT_D, OUT_H, OUT_W,
        G, R,
    )

    # GroupNorm normalize + mean pooling
    out = torch.empty(B, OUT_C, device=x.device, dtype=x.dtype)

    groupnorm_norm_mean_kernel[(B * OUT_C,)](
        conv_out, gn_mean, gn_var,
        gn_weight, gn_bias, out,
        B, OUT_C, OUT_D, OUT_H, OUT_W,
        G, SPATIAL, 1e-5,
    )

    return out


def get_inputs():
    torch.manual_seed(42)
    batch_size = 1024
    in_channels = 3
    out_channels = 16
    depth, height, width = 16, 32, 32
    kernel_size = 4

    conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, kernel_size)
    conv_bias = torch.randn(out_channels)
    gn_weight = torch.randn(out_channels)
    gn_bias = torch.randn(out_channels)
    x = torch.rand(batch_size, in_channels, depth, height, width)

    return [conv_weight, conv_bias, gn_weight, gn_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs_cpu = get_inputs()
    inputs = [t.cuda() for t in inputs_cpu]

    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F

        conv_weight, conv_bias, gn_weight, gn_bias, x = inputs

        # Reference
        conv_out_ref = torch.nn.functional.conv3d(x, conv_weight, conv_bias)
        hs_ref = torch.nn.functional.hardswish(conv_out_ref)
        gn_ref = torch.nn.functional.group_norm(hs_ref, 4, gn_weight, gn_bias, eps=1e-5)
        ref = torch.mean(gn_ref, dim=[2, 3, 4])

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")