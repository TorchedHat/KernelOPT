import torch
import triton
import triton.language as tl
import math

# ConvTranspose2d + BatchNorm + Tanh fused kernel
# ConvTranspose2d with stride=1, padding=1, kernel=5:
# out_h = in_h - 1 + kernel - 2*padding = 32 - 1 + 5 - 2 = 34
# For each output position (oh, ow), we sum over kernel positions (kh, kw) and input channels (ic)
# where ih = oh - kh + padding, iw = ow - kw + padding (for stride=1)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv_transpose_bn_tanh_kernel(
    X_ptr, W_ptr, bias_ptr,
    bn_mean_ptr, bn_var_ptr, bn_weight_ptr, bn_bias_ptr,
    Y_ptr,
    # Input dims
    BATCH, IN_C, IN_H, IN_W,
    # Output dims
    OUT_C, OUT_H, OUT_W,
    # Kernel dims
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # Strides for X (NCHW)
    s_xn, s_xc, s_xh, s_xw,
    # Strides for W (IC, OC, KH, KW) - transposed conv weight layout
    s_wic, s_woc, s_wkh, s_wkw,
    # Strides for Y (NCHW)
    s_yn, s_yc, s_yh, s_yw,
    # BN epsilon
    bn_eps,
    # Total sizes
    M_TOTAL, TOTAL_K,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M offset: M = BATCH * OUT_H * OUT_W
    batch = offs_m // (OUT_H * OUT_W)
    rem = offs_m % (OUT_H * OUT_W)
    out_h = rem // OUT_W
    out_w = rem % OUT_W

    mask_m = offs_m < M_TOTAL
    mask_n = offs_n < OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOTAL_K

        # Decode K: K = IC * KH * KW
        ic = offs_k // (KERNEL_H * KERNEL_W)
        rem_k = offs_k % (KERNEL_H * KERNEL_W)
        kh = rem_k // KERNEL_W
        kw = rem_k % KERNEL_W

        # For ConvTranspose2d (stride=1):
        # output[n, oc, oh, ow] = sum_{ic, kh, kw} input[n, ic, ih, iw] * weight[ic, oc, kh, kw]
        # where ih = oh - kh + PAD_H, iw = ow - kw + PAD_W
        ih = out_h[:, None] - kh[None, :] + PAD_H
        iw = out_w[:, None] - kw[None, :] + PAD_W

        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        valid = valid_spatial & mask_m[:, None] & mask_k[None, :]

        # Load input [BLOCK_M, BLOCK_K]
        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        # Load weight [BLOCK_K, BLOCK_N]: weight shape is (IC, OC, KH, KW)
        w_ptrs = (W_ptr
        w_mask = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Apply BatchNorm: (acc - mean) / sqrt(var + eps) * weight + bias
    bn_mean = tl.load(bn_mean_ptr + offs_n, mask=mask_n, other=0.0)
    bn_var = tl.load(bn_var_ptr + offs_n, mask=mask_n, other=1.0)
    bn_w = tl.load(bn_weight_ptr + offs_n, mask=mask_n, other=1.0)
    bn_b = tl.load(bn_bias_ptr + offs_n, mask=mask_n, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var + bn_eps)
    acc = (acc - bn_mean[None, :]) * inv_std[None, :] * bn_w[None, :] + bn_b[None, :]

    # Apply Tanh
    acc = tl.extra.libdevice.tanh(acc)

    # Store output in NCHW layout
    # Y[n, oc, oh, ow] = acc
    out_ptrs = (Y_ptr
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# MaxPool2d + GroupNorm fused kernel
# MaxPool2d: kernel=2, stride=2 -> (N, C, H//2, W//2)
# GroupNorm: groups=8, channels=128 -> group_size=16

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_HW': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 128, }, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_HW': 256}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_HW': 512}, num_warps=8, num_stages=2),
    ],
    key=['OUT_H', 'OUT_W', 'C_PER_GROUP'],
)
@triton.jit
def maxpool_groupnorm_kernel(
    N, C, IN_H, IN_W,
    OUT_H, OUT_W,
    POOL_KH, POOL_KW,
    POOL_SH, POOL_SW,
    NUM_GROUPS, C_PER_GROUP,
    gn_eps,
    # Strides for X (NCHW)
    s_xn, s_xc, s_xh, s_xw,
    # Strides for Y (NCHW)
    s_yn, s_yc, s_yh, s_yw,
    BLOCK_HW: tl.constexpr,
):
    # Each program handles one (batch, group) pair
    # We compute maxpool for all channels in the group, then apply groupnorm
    pid = tl.program_id(0)
    num_groups_total = N * NUM_GROUPS
    if pid >= num_groups_total:
        return

    batch = pid // NUM_GROUPS
    group = pid % NUM_GROUPS

    # Channels in this group
    c_start = group * C_PER_GROUP
    # c_end = c_start + C_PER_GROUP

    total_out = OUT_H * OUT_W * C_PER_GROUP

    # First pass: compute maxpool and accumulate mean/var for groupnorm
    # We need to store intermediate maxpool results
    # Strategy: compute maxpool values, then compute mean/var, then normalize

    # Accumulate mean and variance across all (c, oh, ow) in this group
    mean_acc = tl.zeros((), dtype=tl.float32)
    var_acc = tl.zeros((), dtype=tl.float32)
    count = tl.zeros((), dtype=tl.float32)

    # We'll do two passes: first compute maxpool + accumulate stats, then normalize
    # But we can't store intermediate results easily without shared memory
    # Instead, use a two-pass approach with reloading

    # Pass 1: compute maxpool values and accumulate stats
    for hw_off in range(0, total_out, BLOCK_HW):
        hw_idx = hw_off + tl.arange(0, BLOCK_HW)
        mask = hw_idx < total_out

        # Decode: hw_idx = c_local * OUT_H * OUT_W + oh * OUT_W + ow
        c_local = hw_idx // (OUT_H * OUT_W)
        rem = hw_idx % (OUT_H * OUT_W)
        oh = rem // OUT_W
        ow = rem % OUT_W

        c_global = c_start + c_local

        # MaxPool: find max over pool window
        pool_max = tl.full((BLOCK_HW,), float('-inf'), dtype=tl.float32)

        for pkh in range(POOL_KH):
            for pkw in range(POOL_KW):
                ih = oh * POOL_SH + pkh
                iw = ow * POOL_SW + pkw
                valid_pool = (ih < IN_H) & (iw < IN_W) & mask
                x_ptrs = (X_ptr
                x_val = tl.load(x_ptrs, mask=valid_pool, other=float('-inf'))
                pool_max = tl.maximum(pool_max, x_val)

        # Accumulate stats
        pool_max_masked = tl.where(mask, pool_max, 0.0)
        mean_acc += tl.sum(pool_max_masked, axis=0)
        var_acc += tl.sum(pool_max_masked * pool_max_masked, axis=0)
        count += tl.sum(mask.to(tl.float32), axis=0)

    # Compute mean and variance
    mean = mean_acc / count
    var = var_acc / count - mean * mean

    inv_std = 1.0 / tl.sqrt(var + gn_eps)

    # Pass 2: compute maxpool again, normalize, and store
    for hw_off in range(0, total_out, BLOCK_HW):
        hw_idx = hw_off + tl.arange(0, BLOCK_HW)
        mask = hw_idx < total_out

        c_local = hw_idx // (OUT_H * OUT_W)
        rem = hw_idx % (OUT_H * OUT_W)
        oh = rem // OUT_W
        ow = rem % OUT_W

        c_global = c_start + c_local

        # MaxPool again
        pool_max = tl.full((BLOCK_HW,), float('-inf'), dtype=tl.float32)

        for pkh in range(POOL_KH):
            for pkw in range(POOL_KW):
                ih = oh * POOL_SH + pkh
                iw = ow * POOL_SW + pkw
                valid_pool = (ih < IN_H) & (iw < IN_W) & mask
                x_ptrs = (X_ptr
                x_val = tl.load(x_ptrs, mask=valid_pool, other=float('-inf'))
                pool_max = tl.maximum(pool_max, x_val)

        # GroupNorm normalize
        normalized = (pool_max - mean) * inv_std

        # Load GN weight and bias
        gn_w = tl.load(GN_W_ptr + c_global, mask=mask & (c_global < C), other=1.0)
        gn_b = tl.load(GN_B_ptr + c_global, mask=mask & (c_global < C), other=0.0)

        out_val = normalized * gn_w + gn_b

        # Store
        out_ptrs = (Y_ptr
        tl.store(out_ptrs, out_val, mask=mask & (c_global < C))


def kernel_function(
    p_conv_transpose_weight,
    p_conv_transpose_bias,
    p_batch_norm_weight,
    p_batch_norm_bias,
    p_group_norm_weight,
    p_group_norm_bias,
    b_batch_norm_running_mean,
    b_batch_norm_running_var,
    x,
):
    # ConvTranspose2d params
    BATCH, IN_C, IN_H, IN_W = x.shape
    # Weight shape for ConvTranspose2d: (in_channels, out_channels, kH, kW)
    OUT_C = p_conv_transpose_weight.shape[1]
    KERNEL_H = p_conv_transpose_weight.shape[2]
    KERNEL_W = p_conv_transpose_weight.shape[3]
    STRIDE_H = 1
    STRIDE_W = 1
    PAD_H = 1
    PAD_W = 1

    # Output size for ConvTranspose2d (stride=1):
    # out_h = (in_h - 1) * stride - 2*padding + kernel_size
    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + KERNEL_H
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + KERNEL_W

    M_TOTAL = BATCH * OUT_H * OUT_W
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    # Intermediate buffer after conv+bn+tanh
    conv_out = torch.empty(BATCH, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)

    # Launch conv+bn+tanh kernel
    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    # Weight strides: (IC, OC, KH, KW)
    w = p_conv_transpose_weight  # shape: (IC, OC, KH, KW)

    conv_transpose_bn_tanh_kernel[grid](
        x, w, p_conv_transpose_bias,
        b_batch_norm_running_mean, b_batch_norm_running_var,
        p_batch_norm_weight, p_batch_norm_bias,
        conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        w.stride(0), w.stride(1), w.stride(2), w.stride(3),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        1e-5,
        M_TOTAL, TOTAL_K,
    )

    # MaxPool2d + GroupNorm
    POOL_KH = 2
    POOL_KW = 2
    POOL_SH = 2
    POOL_SW = 2
    POOL_OUT_H = OUT_H // 2
    POOL_OUT_W = OUT_W // 2
    NUM_GROUPS = 8
    C_PER_GROUP = OUT_C // NUM_GROUPS

    final_out = torch.empty(BATCH, OUT_C, POOL_OUT_H, POOL_OUT_W, device=x.device, dtype=x.dtype)

    grid2 = (BATCH * NUM_GROUPS,)

    maxpool_groupnorm_kernel[grid2](
        conv_out,
        p_group_norm_weight, p_group_norm_bias,
        final_out,
        BATCH, OUT_C, OUT_H, OUT_W,
        POOL_OUT_H, POOL_OUT_W,
        POOL_KH, POOL_KW,
        POOL_SH, POOL_SW,
        NUM_GROUPS, C_PER_GROUP,
        1e-5,
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2), conv_out.stride(3),
        final_out.stride(0), final_out.stride(1), final_out.stride(2), final_out.stride(3),
    )

    return final_out


def get_inputs():
    batch_size = 512
    in_channels = 64
    out_channels = 128
    height, width = 32, 32
    kernel_size = 5
    stride = 1
    padding = 1
    groups = 8
    num_groups = 8

    torch.manual_seed(42)
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')

    import torch.nn as nn
    model = nn.Sequential()
    conv_t = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
    bn = nn.BatchNorm2d(out_channels)
    gn = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)

    conv_t = conv_t.cuda()
    bn = bn.cuda()
    gn = gn.cuda()

    # Use fixed weights for reproducibility
    torch.manual_seed(0)
    p_conv_transpose_weight = torch.randn(in_channels, out_channels, kernel_size, kernel_size, device='cuda') * 0.01
    p_conv_transpose_bias = torch.zeros(out_channels, device='cuda')
    p_batch_norm_weight = torch.ones(out_channels, device='cuda')
    p_batch_norm_bias = torch.zeros(out_channels, device='cuda')
    p_group_norm_weight = torch.ones(out_channels, device='cuda')
    p_group_norm_bias = torch.zeros(out_channels, device='cuda')
    b_batch_norm_running_mean = torch.zeros(out_channels, device='cuda')
    b_batch_norm_running_var = torch.ones(out_channels, device='cuda')

    return [
        p_conv_transpose_weight,
        p_conv_transpose_bias,
        p_batch_norm_weight,
        p_batch_norm_bias,
        p_group_norm_weight,
        p_group_norm_bias,
        b_batch_norm_running_mean,
        b_batch_norm_running_var,
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
        import torch.nn.functional as F

        (p_conv_transpose_weight, p_conv_transpose_bias,
         p_batch_norm_weight, p_batch_norm_bias,
         p_group_norm_weight, p_group_norm_bias,
         b_batch_norm_running_mean, b_batch_norm_running_var,
         x) = inputs

        # Reference computation
        # ConvTranspose2d
        ref_conv = F.conv_transpose2d(x, p_conv_transpose_weight, p_conv_transpose_bias,
                                       stride=1, padding=1)
        # BatchNorm (inference mode with running stats)
        ref_bn = F.batch_norm(ref_conv, b_batch_norm_running_mean, b_batch_norm_running_var,
                               p_batch_norm_weight, p_batch_norm_bias, training=False, eps=1e-5)
        # Tanh
        ref_tanh = torch.tanh(ref_bn)
        # MaxPool2d
        ref_pool = F.max_pool2d(ref_tanh, kernel_size=2, stride=2)
        # GroupNorm
        ref_gn = F.group_norm(ref_pool, num_groups=8, weight=p_group_norm_weight,
                               bias=p_group_norm_bias, eps=1e-5)

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref_gn.shape}")
        print(f"Max diff: {(output - ref_gn).abs().max().item():.6e}")
        print(f"Output range: [{output.min().item():.4f}, {output.max().item():.4f}]")
        print(f"Ref range: [{ref_gn.min().item():.4f}, {ref_gn.max().item():.4f}]")

        assert torch.allclose(output, ref_gn, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref_gn).abs().max().item()):.4e}"
        print("Kernel executed successfully.")