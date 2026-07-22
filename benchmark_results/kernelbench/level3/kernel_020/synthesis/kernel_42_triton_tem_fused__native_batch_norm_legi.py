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
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptrs = x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
    w_ptrs = w_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_x = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        mask_w = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x, w, acc)

        x_ptrs += BLOCK_K * stride_xk
        w_ptrs += BLOCK_K * stride_wk

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_out = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    if b_ptr is not None:
        b = tl.load(b_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
        acc = acc + b[None, :]

    out_ptrs = out_ptr + offs_m_out[:, None] * stride_om + offs_n_out[None, :] * stride_on
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def bn_relu6_kernel(
    x_ptr, out_ptr,
    mean_ptr, var_ptr, weight_ptr, bias_ptr,
    numel, C, HW,
    eps: tl.constexpr,
    apply_relu6: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel

    x = tl.load(x_ptr + offs, mask=mask, other=0.0)

    # channel index
    c = (offs // HW) % C

    mean = tl.load(mean_ptr + c, mask=mask, other=0.0)
    var = tl.load(var_ptr + c, mask=mask, other=0.0)
    w = tl.load(weight_ptr + c, mask=mask, other=1.0)
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)

    x_norm = (x - mean) / tl.sqrt(var + eps)
    out = x_norm * w + b

    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)

    tl.store(out_ptr + offs, out, mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M_total', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_bn_relu6_kernel(
    X_ptr, W_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    GROUPS: tl.constexpr,
    M_total, TOTAL_K,
    eps: tl.constexpr,
    apply_relu6: tl.constexpr,
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

    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS

    group = offs_n // GROUP_OUT_C
    local_oc = offs_n % GROUP_OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, TOTAL_K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        in_c = offs_k % GROUP_IN_C
        kw = (offs_k // GROUP_IN_C) % KERNEL_W
        kh = offs_k // (GROUP_IN_C * KERNEL_W)

        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_total
        valid_k = offs_k[None, :] < TOTAL_K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < BATCH

        x_c = in_c[None, :] + group[:, None] * GROUP_IN_C  # shape issues - need to handle groups carefully

        x_ptrs = (X_ptr
                  + (in_c[None, :] + tl.zeros((BLOCK_M, 1), dtype=tl.int32) * 0) * (IN_H * IN_W)

        mask_x = valid_m & valid_k & valid_h & valid_w & valid_batch

        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: [OUT_C, GROUP_IN_C, KH, KW] -> [GROUP_IN_C * KH * KW, OUT_C]
        w_ptrs = (W_ptr

        mask_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Apply BN + optional ReLU6
    mean = tl.load(mean_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    gamma = tl.load(gamma_ptr + offs_n, mask=offs_n < OUT_C, other=1.0)
    beta = tl.load(beta_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)

    inv_std = 1.0 / tl.sqrt(var[None, :] + eps)
    out = (acc - mean[None, :]) * inv_std * gamma[None, :] + beta[None, :]

    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)

    mask_out = (offs_m[:, None] < M_total) & (offs_n[None, :] < OUT_C)

    # Store in NCHW format: out[n, c, h, w] = out[n * OUT_C * OUT_H * OUT_W + c * OUT_H * OUT_W + h * OUT_W + w]
    out_ptrs = (out_ptr

    tl.store(out_ptrs, out.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M_total', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def depthwise_conv2d_bn_relu6_kernel(
    X_ptr, W_ptr,
    mean_ptr, var_ptr, gamma_ptr, beta_ptr,
    out_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_H, OUT_W,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    M_total, TOTAL_K,
    eps: tl.constexpr,
    apply_relu6: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # For depthwise conv: groups=IN_C, GROUP_IN_C=1, GROUP_OUT_C=1
    # Each output channel corresponds to one input channel
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # For depthwise: TOTAL_K = KH * KW, each output channel c uses input channel c
    for k in range(0, TOTAL_K, BLOCK_K):
        offs_k = k + tl.arange(0, BLOCK_K)

        kw = offs_k % KERNEL_W
        kh = offs_k // KERNEL_W

        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_total
        valid_k = offs_k[None, :] < TOTAL_K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx[:, None] < BATCH

        # For depthwise, each output channel c reads from input channel c
        # x[batch, c, ih, iw] for each output channel c
        # We need to handle this per output channel
        # x_ptrs shape: [BLOCK_M, BLOCK_K] but channel varies with offs_n
        # This is tricky - depthwise means channel is fixed per output channel
        # We'll handle it differently: for each (m, k), the channel is offs_n
        # But offs_n varies across columns, not rows...
        # Actually for depthwise, we need x[batch, oc, ih, iw] * w[oc, 0, kh, kw]
        # The reduction is only over kh, kw (TOTAL_K = KH*KW)
        # So for each output (m, n), we compute sum over k of x[batch_m, oc_n, ih_mk, iw_mk] * w[oc_n, kh_k, kw_k]
        # This means x depends on both m and n dimensions

        # Load x: shape [BLOCK_M, BLOCK_K] but we need channel from offs_n
        # We need to do this as outer product style
        # x_tile[m, k] = x[batch_m, oc_n, ih_mk, iw_mk] -- but oc_n varies!
        # This doesn't fit the standard GEMM pattern well.
        # Instead, let's compute it as: for each n, accumulate over k
        # But that would be sequential over n...

        # Alternative: treat as GEMM where:
        # For depthwise, the "matrix" is diagonal in channel dimension
        # We can still use tl.dot if we're careful

        # Actually the standard approach: iterate over channels in the outer loop
        # and spatial kernel in inner. But here we have BLOCK_N output channels.
        # For each output channel oc in offs_n, we need x[batch, oc, ih, iw]
        # This means x_tile needs to be [BLOCK_M, BLOCK_K] for each oc separately.

        # Let's use a different approach: load x as [BLOCK_M, BLOCK_K] for a fixed oc
        # and w as [BLOCK_K, 1], then accumulate. But this doesn't vectorize over BLOCK_N.

        # For simplicity and correctness, we'll use a loop over BLOCK_N channels
        # This is less efficient but correct. For better perf, we'd need a different tiling.

        # Actually, let's just do it correctly with masked loads
        # x[batch_m, oc_n, ih_mk, iw_mk]: depends on both m and n
        # We can't directly use tl.dot for this pattern

        # Use element-wise computation instead
        # For each element (m, n), compute sum_k x[batch_m, oc_n, ih_mk, iw_mk] * w[oc_n, kh_k, kw_k]

        # Load x for all (m, k) combinations, but channel varies with n
        # This requires a 3D tensor which triton doesn't support directly

        # Workaround: process one channel at a time (loop over BLOCK_N)
        # This is O(BLOCK_N * BLOCK_K) loads but correct

        # For now, use a simpler approach: load x[m, k] for each n separately
        # This is done implicitly by the masked load pattern

        # Actually the cleanest solution for depthwise is:
        # Reshape to treat it as: M = BATCH * OUT_H * OUT_W, N = IN_C, K = KH * KW
        # x_reshaped[m, n, k] = x[batch_m, n, ih_mk, iw_mk]
        # w_reshaped[n, k] = w[n, 0, kh_k, kw_k]
        # out[m, n] = sum_k x_reshaped[m, n, k] * w_reshaped[n, k]
        # This is a batched dot product, not a standard GEMM

        # For Triton, we can handle this by loading x with channel from offs_n
        # x_ptrs[m, k] for channel offs_n[n] -- but this creates a 3D dependency

        # Simplest correct implementation: use element-wise ops
        # For each k, load x[m, n] = x[batch_m, oc_n, ih_mk, iw_mk]
        # and w[n, k] = w[oc_n, kh_k, kw_k]
        # acc[m, n] += x[m, n] * w[n, k]  -- but this is element-wise, not dot

        # Let's implement this correctly:
        # For each k offset, compute x_mk_n and w_nk and do element-wise multiply-add

        # x_mk_n[m, n] = x[batch_m, oc_n, ih_mk, iw_mk]
        x_ptrs_dw = (X_ptr

        # This approach doesn't work cleanly in a vectorized way
        # Let's just break out of this and use a simpler loop
        break

    # Fall back to element-wise computation for depthwise
    # Reset acc
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(TOTAL_K):
        kw_val = k % KERNEL_W
        kh_val = k // KERNEL_W

        ih_k = out_h_idx * STRIDE_H + kh_val - PAD_H
        iw_k = out_w_idx * STRIDE_W + kw_val - PAD_W

        valid_h_k = (ih_k >= 0) & (ih_k < IN_H)
        valid_w_k = (iw_k >= 0) & (iw_k < IN_W)
        valid_m_k = (offs_m < M_total) & (batch_idx < BATCH)

        # x[m, n] = x[batch_m, oc_n, ih_k_m, iw_k_m]
        x_ptrs_k = (X_ptr

        mask_x_k = (valid_m_k[:, None] & valid_h_k[:, None] & valid_w_k[:, None]

        x_val = tl.load(x_ptrs_k, mask=mask_x_k, other=0.0)

        # w[n] = w[oc_n, 0, kh_val, kw_val]
        w_ptrs_k = W_ptr + offs_n * (KERNEL_H * KERNEL_W) + kh_val * KERNEL_W + kw_val
        w_val = tl.load(w_ptrs_k, mask=offs_n < IN_C, other=0.0)

        acc = acc + x_val * w_val[None, :]

    # Apply BN + optional ReLU6
    mean = tl.load(mean_ptr + offs_n, mask=offs_n < IN_C, other=0.0)
    var = tl.load(var_ptr + offs_n, mask=offs_n < IN_C, other=0.0)
    gamma = tl.load(gamma_ptr + offs_n, mask=offs_n < IN_C, other=1.0)
    beta = tl.load(beta_ptr + offs_n, mask=offs_n < IN_C, other=0.0)

    inv_std = 1.0 / tl.sqrt(var[None, :] + eps)
    out = (acc - mean[None, :]) * inv_std * gamma[None, :] + beta[None, :]

    if apply_relu6:
        out = tl.minimum(tl.maximum(out, 0.0), 6.0)

    mask_out = (offs_m[:, None] < M_total) & (offs_n[None, :] < IN_C)

    out_ptrs = (out_ptr

    tl.store(out_ptrs, out.to(tl.float32), mask=mask_out)


def run_conv2d_bn_relu6(x, weight, bn_mean, bn_var, bn_weight, bn_bias,
                         stride, padding, groups, apply_relu6, eps=1e-5):
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    STRIDE_H, STRIDE_W = stride
    PAD_H, PAD_W = padding
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1

    out = torch.empty(BATCH, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)

    M_total = BATCH * OUT_H * OUT_W
    GROUP_IN_C = IN_C // groups
    TOTAL_K = GROUP_IN_C * KH * KW

    if groups == IN_C and groups == OUT_C:
        # Depthwise convolution
        grid = lambda meta: (
            triton.cdiv(M_total, meta['BLOCK_M']),
            triton.cdiv(IN_C, meta['BLOCK_N']),
        )
        depthwise_conv2d_bn_relu6_kernel[grid](
            x, weight,
            bn_mean, bn_var, bn_weight, bn_bias,
            out,
            BATCH, IN_C, IN_H, IN_W,
            OUT_H, OUT_W,
            KH, KW,
            STRIDE_H, STRIDE_W,
            PAD_H, PAD_W,
            M_total, KH * KW,
            eps,
            apply_relu6,
        )
    else:
        grid = lambda meta: (
            triton.cdiv(M_total, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )
        conv2d_bn_relu6_kernel[grid](
            x, weight,
            bn_mean, bn_var, bn_weight, bn_bias,
            out,
            BATCH, IN_C, IN_H, IN_W,
            OUT_C, OUT_H, OUT_W,
            KH, KW,
            STRIDE_H, STRIDE_W,
            PAD_H, PAD_W,
            groups,
            M_total, TOTAL_K,
            eps,
            apply_relu6,
        )

    return out


def run_linear(x, weight, bias):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    out = torch.empty