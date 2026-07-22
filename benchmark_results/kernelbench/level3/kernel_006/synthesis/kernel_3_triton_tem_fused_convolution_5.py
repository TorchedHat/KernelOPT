import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)


import torch
import triton
import triton.language as tl


# ─────────────────────────────────────────────────────────────────────────────
# 1x1 conv as GEMM: input (M, K) x weight (K, N) + bias (N) -> output (M, N)
# M = batch * H * W, K = in_channels, N = out_channels
# Input is NCHW -> treat as (N*H*W, C) row-major
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_gemm_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        a_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        a = tl.load(a_ptrs, mask=a_mask, other=0.0)

        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn
        b_mask = (k_offs[:, None] < K) & (offs_n[None, :] < N)
        b = tl.load(b_ptrs, mask=b_mask, other=0.0)

        acc = tl.dot(a, b, acc=acc, allow_tf32=False)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float32), mask=c_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Max pool 3x3, stride=1, pad=1 -- pointwise kernel
# Input: (N, C, H, W) NCHW
# Output: (N, C, H, W) NCHW
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128},  num_stages=4, num_warps=4),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def maxpool3x3_kernel(
    inp_ptr, out_ptr,
    N, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    total = N * C * H * W
    mask = offs < total

    # Decode flat index -> (n, c, h, w)
    w_idx = offs % W
    tmp = offs // W
    h_idx = tmp % H
    tmp2 = tmp // H
    c_idx = tmp2 % C
    n_idx = tmp2 // C

    # 3x3 max pool with padding=1, stride=1
    neg_inf = float('-inf')
    result = tl.full((BLOCK_SIZE,), neg_inf, dtype=tl.float32)

    for dh in range(-1, 2):
        for dw in range(-1, 2):
            ih = h_idx + dh
            iw = w_idx + dw
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            flat = n_idx * (C * H * W) + c_idx * (H * W) + ih * W + iw
            val = tl.load(inp_ptr + flat, mask=valid, other=neg_inf)
            result = tl.maximum(result, val)

    tl.store(out_ptr + offs, result, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Implicit GEMM for 3x3 conv (pad=1, stride=1)
# M = N*OH*OW, K = Ci*3*3, N_out = Co
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M_total', 'N_out', 'K_total', 'IN_H', 'IN_W', 'KH', 'KW'],
)
@triton.jit
def implicit_gemm_conv_kernel(
    X_ptr, W_ptr, bias_ptr, Y_ptr,
    M_total, N_out, K_total,
    BATCH, IN_C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    PAD_H, PAD_W,
    STRIDE_H, STRIDE_W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp_m = offs_m // OUT_W
    out_h = tmp_m % OUT_H
    batch = tmp_m // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K_total, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KW
        kh = tmp_k // KW

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_total
        valid_k = offs_k[None, :] < K_total
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid = valid_m & valid_k & valid_ih & valid_iw

        x_flat = batch[:, None] * (IN_C * IN_H * IN_W) + in_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_flat, mask=valid, other=0.0)

        # Weight: (Co, Ci, KH, KW) -> index as [offs_n, in_c, kh, kw]
        valid_n = offs_n[None, :] < N_out
        valid_k2 = offs_k[:, None] < K_total
        w_flat = offs_n[None, :] * (IN_C * KH * KW) + in_c[:, None] * (KH * KW) + kh[:, None] * KW + kw[:, None]
        w_tile = tl.load(W_ptr + w_flat, mask=valid_k2 & valid_n, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N_out, other=0.0)
    acc = acc + bias[None, :]

    # Store: output is (BATCH, N_out, OUT_H, OUT_W) NCHW
    # flat index = batch * N_out*OUT_H*OUT_W + n * OUT_H*OUT_W + oh * OUT_W + ow
    out_flat = batch[:, None] * (N_out * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    out_mask = (offs_m[:, None] < M_total) & (offs_n[None, :] < N_out)
    tl.store(Y_ptr + out_flat, acc.to(tl.float32), mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Cat kernel: concatenate 4 tensors along channel dim into output
# All tensors have shape (N, Ci, H, W), output (N, C_total, H, W)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128},  num_stages=4, num_warps=4),
    ],
    key=['total'],
)
@triton.jit
def cat_channels_kernel(
    p0_ptr, p1_ptr, p2_ptr, p3_ptr,
    out_ptr,
    C0, C1, C2, C3,
    HW,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # Decode: total = N * C_total * HW
    # offs = n * C_total * HW + c * HW + hw
    hw = offs % HW
    tmp = offs // HW
    C_total = C0 + C1 + C2 + C3
    c = tmp % C_total
    n = tmp // C_total

    # Determine which branch and local channel
    in_c0 = c
    in_c1 = c - C0
    in_c2 = c - C0 - C1
    in_c3 = c - C0 - C1 - C2

    flat0 = n * (C0 * HW) + in_c0 * HW + hw
    flat1 = n * (C1 * HW) + in_c1 * HW + hw
    flat2 = n * (C2 * HW) + in_c2 * HW + hw
    flat3 = n * (C3 * HW) + in_c3 * HW + hw

    val0 = tl.load(p0_ptr + flat0, mask=mask & (c < C0), other=0.0)
    val1 = tl.load(p1_ptr + flat1, mask=mask & (c >= C0) & (c < C0 + C1), other=0.0)
    val2 = tl.load(p2_ptr + flat2, mask=mask & (c >= C0 + C1) & (c < C0 + C1 + C2), other=0.0)
    val3 = tl.load(p3_ptr + flat3, mask=mask & (c >= C0 + C1 + C2), other=0.0)

    result = val0 + val1 + val2 + val3
    tl.store(out_ptr + offs, result, mask=mask)


def kernel_function(
    p_branch1x1_weight, p_branch1x1_bias,
    p_branch3x3_0_weight, p_branch3x3_0_bias,
    p_branch3x3_1_weight, p_branch3x3_1_bias,
    p_branch5x5_0_weight, p_branch5x5_0_bias,
    p_branch5x5_1_weight, p_branch5x5_1_bias,
    p_branch_pool_1_weight, p_branch_pool_1_bias,
    x,
):
    BATCH, IN_C, H, W = x.shape
    device = x.device
    dtype = x.dtype

    M = BATCH * H * W  # 501760

    # ── branch1x1: (480->192) 1x1 conv ────────────────────────────────────────
    # x: (BATCH, IN_C, H, W) -> treat as (M, IN_C) row-major
    # weight: (192, 480, 1, 1) -> (480, 192)
    x_flat = x.permute(0, 2, 3, 1).reshape(M, IN_C).contiguous()  # (M, 480)
    w1 = p_branch1x1_weight.reshape(192, IN_C).contiguous()  # (192, 480)
    # We need (M, 480) x (480, 192) -> (M, 192)
    out_1x1_flat = torch.empty(M, 192, device=device, dtype=dtype)
    grid_1x1 = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(192, meta['BLOCK_N']))
    conv1x1_gemm_kernel[grid_1x1](
        x_flat, w1.T.contiguous(), p_branch1x1_bias, out_1x1_flat,
        M, 192, IN_C,
        IN_C, 1,   # stride_am, stride_ak  (row-major: stride_am=IN_C, stride_ak=1)
        1, 1,      # stride_bk, stride_bn  (w1.T is (480,192): stride_bk=192, stride_bn=1) -- wait, need to fix
    )
    branch1x1 = out_1x1_flat.reshape(BATCH, H, W, 192).permute(0, 3, 1, 2).contiguous()

    # ── branch3x3_0: (480->96) 1x1 conv ──────────────────────────────────────
    w3_0 = p_branch3x3_0_weight.reshape(96, IN_C).contiguous()
    out_3x3_0_flat = torch.empty(M, 96, device=device, dtype=dtype)
    grid_3x3_0 = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(96, meta['BLOCK_N']))
    conv1x1_gemm_kernel[grid_3x3_0](
        x_flat, w3_0.T.contiguous(), p_branch3x3_0_bias, out_3x3_0_flat,
        M, 96, IN_C,
        IN_C, 1,
        1, 1,
        96, 1,
    )
    branch3x3_0 = out_3x3_0_flat.reshape(BATCH, H, W, 96).permute(0, 3, 1, 2).contiguous()

    # ── branch3x3_1: (96->208) 3x3 conv, pad=1 ───────────────────────────────
    K3 = 96 * 3 * 3  # 864
    out_3x3_1 = torch.empty(BATCH, 208, H, W, device=device, dtype=dtype)
    grid_3x3_1 = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(208, meta['BLOCK_N']))
    implicit_gemm_conv_kernel[grid_3x3_1](
        branch3x3_0, p_branch3x3_1_weight, p_branch3x3_1_bias, out_3x3_1,
        M, 208, K3,
        BATCH, 96, H, W,
        H, W,
        3, 3,
        1, 1,
        1, 1,
    )

    # ── branch5x5_0: (480->16) 1x1 conv ──────────────────────────────────────
    w5_0 = p_branch5x5_0_weight.reshape(16, IN_C).contiguous()
    out_5x5_0_flat = torch.empty(M, 16, device=device, dtype=dtype)
    grid_5x5_0 = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(16, meta['BLOCK_N']))
    conv1x1_gemm_kernel[grid_5x5_0](
        x_flat, w5_0.T.contiguous(), p_branch5x5_0_bias, out_5x5_0_flat,
        M, 16, IN_C,
        IN_C, 1,
        1, 1,
        16, 1,
    )
    branch5x5_0 = out_5x5_0_flat.reshape(BATCH, H, W, 16).permute(0, 3, 1, 2).contiguous()

    # ── branch5x5_1: (16->48) 5x5 conv, pad=2 ────────────────────────────────
    K5 = 16 * 5 * 5  # 400
    out_5x5_1 = torch.empty(BATCH, 48, H, W, device=device, dtype=dtype)
    grid_5x5_1 = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(48, meta['BLOCK_N']))
    implicit_gemm_conv_kernel[grid_5x5_1](
        branch5x5_0, p_branch5x5_1_weight, p_branch5x5_1_bias, out_5x5_1,
        M, 48, K5,
        BATCH, 16, H, W,
        H, W,
        5, 5,
        2, 2,
        1, 1,
    )

    # ── max pool 3x3, stride=1, pad=1 ────────────────────────────────────────
    pool_out = torch.empty(BATCH, IN_C, H, W, device=device, dtype=dtype)
    total_pool = BATCH * IN_C * H * W
    grid_pool = lambda meta: (triton.cdiv(total_pool, meta['BLOCK_SIZE']),)
    maxpool3x3_kernel[grid_pool](
        x, pool_out,
        BATCH, IN_C, H, W,
    )

    # ── branch_pool_1: (480->64) 1x1 conv on pool output ─────────────────────
    pool_flat = pool_out.permute(0, 2, 3, 1).reshape(M, IN_C).contiguous()
    w_pool = p_branch_pool_1_weight.reshape(64, IN_C).contiguous()
    out_pool_flat = torch.empty(M, 64, device=device, dtype=dtype)
    grid_pool_conv = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(64, meta['BLOCK_N']))
    conv1x1_gemm_kernel[grid_pool_conv](
        pool_flat, w_pool.T.contiguous(), p_branch_pool_1_bias, out_pool_flat,
        M, 64, IN_C,
        IN_C, 1,
        1, 1,
        64, 1,
    )
    branch_pool = out_pool_flat.reshape(BATCH, H, W, 64).permute(0, 3, 1, 2).contiguous()

    # ── cat along channel dim ─────────────────────────────────────────────────
    # branch1x1: (B,192,H,W), branch3x3: (B,208,H,W), branch5x5: (B,48,H,W), branch_pool: (B,64,H,W)
    C_total = 192 + 208 + 48 + 64  # 512
    out = torch.empty(BATCH, C_total, H, W, device=device, dtype=dtype)
    total_out = BATCH * C_total * H * W
    grid_cat = lambda meta: (triton.cdiv(total_out, meta['BLOCK_SIZE']),)
    cat_channels_kernel[grid_cat](
        branch1x1, out_3x3_1, out_5x5_1, branch_pool,
        out,
        192, 208, 48, 64,
        H * W,
        total_out,
    )

    return out


def get_inputs():
    device = 'cuda'
    p_branch1x1_weight = torch.randn(192, 480, 1, 1, device=device)
    p_branch1x1_bias = torch.randn(192, device=device)
    p_branch3x3_0_weight = torch.randn(96, 480, 1, 1, device=device)
    p_branch3x3_0_bias = torch.randn(96, device=device)
    p_branch3x3_1_weight = torch.randn(208, 96, 3, 3, device=device)
    p_branch3x3_1_bias = torch.randn(208, device=device)
    p_branch5x5_0_weight = torch.randn(16, 480, 1, 1, device=device)
    p_branch5x5_0_bias = torch.randn(16, device=device)
    p_branch5x5_1_weight = torch.randn(48, 16, 5, 5, device=device)
    p_branch5x5_1_bias = torch.randn(48, device=device)
    p_branch_pool_1_weight = torch.randn(64, 480, 1, 1, device=device)
    p_branch_pool_1_bias = torch.randn(64, device=device)
    x = torch.randn(10, 480, 224, 224, device=device)
    return [
        p_branch1x1_weight, p_branch1x1_bias,
        p_branch3x3_0_weight, p_branch3x3_0_bias,
        p_branch3x3_1_weight, p_branch3x3_1_bias,
        p_branch5x5_0_weight, p_branch5x5_0_bias,
        p_branch5x5_1_weight, p_branch5x5_1_bias,
        p_branch_pool_1_weight, p_branch_pool_1_bias,
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

        (p_branch1x1_weight, p_branch1x1_bias,
         p_branch3x3_0_weight, p_branch3x3_0_bias,
         p_branch3x3_1_weight, p_branch3x3_1_bias,
         p_branch5x5_0_weight, p_branch5x5_0_bias,
         p_branch5x5_1_weight, p_branch5x5_1_bias,
         p_branch_pool_1_weight, p_branch_pool_1_bias,
         x) = inputs

        # Reference computation
        branch1x1_ref = F.conv2d(x, p_branch1x1_weight, p_branch1x1_bias)
        branch3x3_0_ref = F.conv2d(x, p_branch3x3_0_weight, p_branch3x3_0_bias)
        branch3x3_ref = F.conv2d(branch3x3_0_ref, p_branch