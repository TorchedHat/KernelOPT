import torch
import triton
import triton.language as tl
import math



import torch
import triton
import triton.language as tl
import math


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_bn_relu_kernel(
    X_ptr, W_ptr, BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    Y_ptr,
    # Input dims
    BATCH, IN_C, IN_H, IN_W,
    # Output dims
    OUT_C, OUT_H, OUT_W,
    # Conv params
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    GROUPS,
    # BN param
    BN_EPS,
    # Strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # Strides for W (out_c, in_c, kh, kw) -> stored as (out_c, kh, kw, in_c) for NHWC
    s_woc, s_wkh, s_wkw, s_wic,
    # Strides for Y (NHWC layout)
    s_yn, s_yh, s_yw, s_yc,
    # Total sizes for autotuning
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
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

    # Decode N -> out_c
    out_c_idx = offs_n

    # Group info
    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS
    group = out_c_idx // GROUP_OUT_C  # shape [BLOCK_N]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = GROUP_IN_C * KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        # Decode k -> (in_c_local, kh, kw)
        in_c_local = k_offs % GROUP_IN_C
        kh_idx = (k_offs // GROUP_IN_C) % KERNEL_H
        kw_idx = k_offs // (GROUP_IN_C * KERNEL_H)

        # Input spatial coords
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        # in_c global = group * GROUP_IN_C + in_c_local
        # For groups, each output channel group uses its own input channels
        # group for each output channel
        grp = out_c_idx // GROUP_OUT_C  # [BLOCK_N]
        in_c_global = grp[None, :] * GROUP_IN_C + in_c_local[:, None]  # [BLOCK_K, BLOCK_N] - not right

        # Actually for X load: in_c = group_of_m * GROUP_IN_C + in_c_local
        # But group depends on output channel, not input position
        # For depthwise: GROUPS=IN_C, GROUP_IN_C=1, GROUP_OUT_C=1
        # group = out_c_idx // 1 = out_c_idx
        # in_c_global for X = out_c_idx * 1 + in_c_local = out_c_idx + 0

        # For standard conv: GROUPS=1, GROUP_IN_C=IN_C
        # group=0, in_c_global = in_c_local

        # The issue: X tile is [BLOCK_M, BLOCK_K] but in_c depends on which output channel
        # For grouped conv, different output channels use different input channels
        # This makes it hard to do a single tl.dot

        # Simplification: handle groups=1 and depthwise (groups=IN_C) separately
        # For groups=1: in_c_global = in_c_local (same for all output channels)
        # For depthwise: each output channel c uses input channel c

        # We'll use a mask approach for depthwise
        # in_c_global for X load: always use in_c_local (relative to group)
        # But which group? For X, group is determined by... the output channel being computed
        # This is the fundamental issue with grouped conv in a single tl.dot

        # Solution: for each BLOCK_N tile, all output channels in that tile belong to the same group
        # (assuming GROUP_OUT_C >= BLOCK_N or we handle it carefully)
        # For depthwise: GROUP_OUT_C=1, so each output channel is its own group
        # This means different columns of the output tile use different input channels

        # For depthwise conv, we can't easily do tl.dot across groups
        # Let's handle it by loading X with the correct in_c for each (m, n) pair

        # For simplicity, let's use the approach where:
        # - For groups=1: standard conv, in_c_global = in_c_local
        # - For depthwise: we need per-column in_c

        # Actually the cleanest approach: compute in_c_global based on group of each output channel
        # group_of_n = out_c_idx // GROUP_OUT_C  [BLOCK_N]
        # in_c_for_n = group_of_n * GROUP_IN_C + in_c_local  [BLOCK_K, BLOCK_N] broadcast

        # For X load: we need [BLOCK_M, BLOCK_K] tile
        # But in_c depends on n (output channel), so we can't do a single tl.dot

        # The standard approach for grouped conv in Triton:
        # Assume all output channels in a tile belong to the same group
        # This works when BLOCK_N <= GROUP_OUT_C or when we process one group at a time

        # For depthwise (GROUP_OUT_C=1, BLOCK_N=32 or 64):
        # Each output channel in the tile is a different group
        # We need to handle this differently

        # PRACTICAL SOLUTION: Use separate kernels for depthwise vs standard conv
        # But that's complex. Instead, use the fact that for depthwise,
        # GROUP_IN_C=1, so TOTAL_K = KERNEL_H * KERNEL_W (small)
        # and we can iterate over output channels in the inner loop

        # For now, implement correctly for groups=1 (standard conv)
        # and handle depthwise as a special case via GROUPS parameter

        # Standard approach that works for groups=1:
        in_c_g = in_c_local  # [BLOCK_K] - relative to group 0

        # X pointer: batch_idx, ih, iw, in_c_g
        x_ptrs = (X_ptr +
                  batch_idx[:, None] * s_xn +
                  in_c_g[None, :] * s_xc)

        valid_x = ((batch_idx[:, None] < BATCH) &
                   (ih >= 0) & (ih < IN_H) &
                   (iw >= 0) & (iw < IN_W) &
                   (k_offs[None, :] < TOTAL_K))

        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)  # [BLOCK_M, BLOCK_K]

        # W pointer: out_c, kh, kw, in_c_local
        w_ptrs = (W_ptr +
                  out_c_idx[None, :] * s_woc +
                  kh_idx[:, None] * s_wkh +
                  kw_idx[:, None] * s_wkw +
                  in_c_local[:, None] * s_wic)

        valid_w = ((k_offs[:, None] < TOTAL_K) &
                   (out_c_idx[None, :] < OUT_C))

        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)  # [BLOCK_K, BLOCK_N]

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Apply BN + ReLU
    bn_mean = tl.load(BN_mean_ptr + out_c_idx, mask=out_c_idx < OUT_C, other=0.0)
    bn_var = tl.load(BN_var_ptr + out_c_idx, mask=out_c_idx < OUT_C, other=1.0)
    bn_w = tl.load(BN_weight_ptr + out_c_idx, mask=out_c_idx < OUT_C, other=1.0)
    bn_b = tl.load(BN_bias_ptr + out_c_idx, mask=out_c_idx < OUT_C, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + BN_EPS)
    acc = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]
    acc = tl.maximum(acc, 0.0)

    # Store output
    out_mask = ((batch_idx[:, None] < BATCH) &
                (out_h_idx[:, None] < OUT_H) &
                (out_w_idx[:, None] < OUT_W) &
                (out_c_idx[None, :] < OUT_C))

    y_ptrs = (Y_ptr +
              batch_idx[:, None] * s_yn +
              out_h_idx[:, None] * s_yh +
              out_w_idx[:, None] * s_yw +
              out_c_idx[None, :] * s_yc)

    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def depthwise_conv_bn_relu_kernel(
    X_ptr, W_ptr, BN_mean_ptr, BN_var_ptr, BN_weight_ptr, BN_bias_ptr,
    Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    BN_EPS,
    s_xn, s_xh, s_xw, s_xc,
    s_woc, s_wkh, s_wkw,
    s_yn, s_yh, s_yw, s_yc,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # For depthwise conv: each output channel c uses only input channel c
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    out_c_idx = offs_n  # = in_c_idx for depthwise

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)
        kh_idx = k_offs // KERNEL_W
        kw_idx = k_offs % KERNEL_W

        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        # For depthwise: x[batch, ih, iw, c] for each c in offs_n
        # We need x_tile[BLOCK_M, BLOCK_K] for each output channel separately
        # This is tricky because different columns use different channels

        # Approach: iterate over output channels in BLOCK_N dimension
        # Actually for depthwise, we can reformulate:
        # For each (m, n) pair: acc[m,n] += sum_k X[batch_m, ih_m_k, iw_m_k, c_n] * W[c_n, kh_k, kw_k]
        # The X load depends on both m and n (through c_n)

        # We'll use a loop over BLOCK_N to handle this
        # But that's slow. Better: load X as [BLOCK_M, BLOCK_K, BLOCK_N] - too much memory

        # Practical approach for depthwise: process one channel at a time in the N dimension
        # Or: use the fact that for depthwise, we can restructure as:
        # For each output channel c (in BLOCK_N):
        #   For each spatial position (in BLOCK_M):
        #     acc += X[b, ih, iw, c] * W[c, kh, kw]

        # Since BLOCK_K is small (9 for 3x3), we can afford to be less efficient
        # Load X for all channels at once: X[BLOCK_M, BLOCK_K, BLOCK_N] - too large

        # Best approach: for depthwise, use a different loop structure
        # Process each k independently and accumulate
        # x_val[m, k] depends on spatial position only (not channel)
        # w_val[k, n] = W[c_n, kh_k, kw_k]
        # But x_val also depends on channel c_n!

        # For depthwise with BLOCK_K=1 (iterate k one at a time):
        # x_val[m, n] = X[batch_m, ih_m, iw_m, c_n]
        # w_val[n] = W[c_n, kh, kw]
        # acc[m, n] += x_val[m, n] * w_val[n]

        # Let's do this with BLOCK_K=1 effectively by loading per-k
        # But autotune has BLOCK_K as constexpr...

        # Alternative: load X[BLOCK_M, BLOCK_N] for each k position
        # This works! For each k:
        #   ih_k = kh_k * STRIDE_H + ... (scalar for this k)
        #   x_tile[m, n] = X[batch_m, ih_m_k, iw_m_k, c_n]
        #   w_tile[n] = W[c_n, kh_k, kw_k]
        #   acc[m, n] += x_tile[m, n] * w_tile[n]

        # This avoids tl.dot but is correct for depthwise
        # We'll iterate k in the outer loop with BLOCK_K steps

        for k_inner in range(BLOCK_K):
            k_idx = k_start + k_inner
            kh_k = k_idx // KERNEL_W
            kw_k = k_idx % KERNEL_W

            ih_k = out_h_idx * STRIDE_H + kh_k - PAD_H  # [BLOCK_M]
            iw_k = out_w_idx * STRIDE_W + kw_k - PAD_W  # [BLOCK_M]

            valid_spatial = ((batch_idx < BATCH) &
                            (ih_k >= 0) & (ih_k < IN_H) &
                            (iw_k >= 0) & (iw_k < IN_W) &
                            (k_idx < TOTAL_K))

            # Load X[m, n]: for each m, load channel c_n
            x_ptrs = (X_ptr +
                      batch_idx[:, None] * s_xn +
                      ih_k[:, None] * s_xh +
                      iw_k[:, None] * s_xw +
                      out_c_idx[None, :] * s_xc)

            valid_xn = valid_spatial[:, None] & (out_c_idx[None, :] < OUT_C)
            x_tile = tl.load(x_ptrs, mask=valid_xn, other=0.0)  # [BLOCK_M, BLOCK_N]

            # Load W[n]: W[c_n, kh_k, kw_k]
            w_ptrs = (W_ptr +
            valid_wn = (out_c_idx < OUT_C) & (k_idx < TOTAL_K)
            w_tile = tl.load(w_ptrs, mask=valid_wn, other=0.0)  # [BLOCK_N]

            acc += x_tile * w_tile[None, :]

    # Apply BN + ReLU
    bn_mean = tl.load(BN_mean_ptr + out_c_idx, mask=out_c_idx < OUT_C, other=0.0)
    bn_var = tl.load(BN_var_ptr + out_c_idx, mask=out_c_idx < OUT_C, other=1.0)
    bn_w = tl.load(BN_weight_ptr + out_c_idx, mask=out_c_idx < OUT_C, other=1.0)
    bn_b = tl.load(BN_bias_ptr + out_c_idx, mask=out_c_idx < OUT_C, other=0.0)

    inv_std = 1.0 / tl.sqrt(bn_var[None, :] + BN_EPS)
    acc = (acc - bn_mean[None, :]) * inv_std * bn_w[None, :] + bn_b[None, :]
    acc = tl.maximum(acc, 0.0)

    out_mask = ((batch_idx[:, None] < BATCH) &
                (out_h_idx[:, None] < OUT_H) &
                (out_w_idx[:, None] < OUT_W) &
                (out_c_idx[None, :] < OUT_C))

    y_ptrs = (Y_ptr +
              batch_idx[:, None] * s_yn +
              out_h_idx[:, None] * s_yh +
              out_w_idx[:, None] * s_yw +
              out_c_idx[None, :] * s_yc)

    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['N', 'C'],
)
@triton.jit
def avg_pool_kernel(
    X_ptr, Y_ptr,
    N, C, H, W,
    POOL_H, POOL_W,
    s_xn, s_xc, s_xh, s_xw,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, c) pair
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    nc_mask = offs < N * C
    n_idx = offs // C
    c_idx = offs % C

    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    count = POOL_H * POOL_W

    for ph in range(POOL_H):
        for pw in range(POOL_W):
            x_ptrs = X_ptr + n_idx * s_xn + c_idx * s_xc + ph * s_xh + pw * s_xw
            x_val = tl.load(x_ptrs, mask=nc_mask, other=0.0)
            acc += x_val

    acc = acc / count
    tl.store(Y_ptr + offs, acc, mask=nc_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 8, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_ptrs = W_ptr + offs_n[None, :] * K + offs_k[:, None]
        w_mask = (offs_n[None, :] < N) & (offs_k[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc += bias[None, :]

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def run_conv_bn_relu(x_nhwc, w, bn_mean, bn_var, bn_weight, bn_bias,
                     stride_h, stride_w, pad_h, pad_w, groups, bn_eps=1e-5):
    """Run conv+BN+ReLU, x in NHWC, w in (out_c, in_c/groups, kh, kw)"""
    N, H, W, C_in = x_nhwc.shape
    out_c, in_c_per_group, kh, kw = w.shape
    groups_val = groups

    out_h = (H + 2 * pad_h - kh) // stride_h + 1
    out_w = (W + 2 * pad_w - kw) // stride_w + 1

    # Convert weight to NHWC-friendly format: (out_c, kh, kw, in_c_per_group)
    w_nhwc = w.permute(0, 2, 3, 1).contiguous()  # (out_c, kh, kw, in_c_per_group)

    y = torch.empty((N, out_h, out_w, out_c), device=x_nhwc.device, dtype=torch.float32)

    M = N * out_h * out_w
    K_total = in_c_per_group * kh * kw

    if groups_val == C_in and in_c_per_group == 1:
        # Depthwise conv
        grid = lambda meta: (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cd