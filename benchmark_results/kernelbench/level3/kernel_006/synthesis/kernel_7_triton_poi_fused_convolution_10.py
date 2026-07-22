import torch
import triton
import triton.language as tl
import argparse

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Max-pool 3x3, stride=1, pad=1  (pointwise over output elements)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BS': 256}, num_warps=4),
        triton.Config({'BS': 512}, num_warps=8),
        triton.Config({'BS': 1024}, num_warps=8),
        triton.Config({'BS': 128}, num_warps=4),
    ],
    key=['N_ELEM'],
)
@triton.jit
def maxpool3x3_s1p1_kernel(
    x_ptr, out_ptr,
    B, C, H, W,
    N_ELEM,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BS + tl.arange(0, BS)
    mask = offs < N_ELEM

    # decode flat index -> (b, c, oh, ow)
    tmp = offs
    ow = tmp % W;  tmp = tmp // W
    oh = tmp % H;  tmp = tmp // H
    c  = tmp % C;  b  = tmp // C

    best = tl.full([BS], float('-inf'), dtype=tl.float32)

    for dh in range(3):
        for dw in range(3):
            ih = oh + dh - 1
            iw = ow + dw - 1
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            idx = b * (C * H * W) + c * (H * W) + ih * W + iw
            val = tl.load(x_ptr + idx, mask=valid, other=float('-inf'))
            best = tl.maximum(best, val)

    tl.store(out_ptr + offs, best, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: 1x1 conv as GEMM  (M = B*H*W spatial positions, N = out_ch, K = in_ch)
# Input layout: (B, C_in, H, W) -> treat as (M, K) where M=B*H*W, K=C_in
# Weight layout: (C_out, C_in, 1, 1) -> treat as (K, N) = (C_in, C_out)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'TM': 64,  'TN': 64,  'TK': 32}, num_warps=4, num_stages=3),
        triton.Config({'TM': 128, 'TN': 64,  'TK': 32}, num_warps=4, num_stages=3),
        triton.Config({'TM': 64,  'TN': 128, 'TK': 32}, num_warps=4, num_stages=3),
        triton.Config({'TM': 128, 'TN': 128, 'TK': 32}, num_warps=8, num_stages=3),
        triton.Config({'TM': 64,  'TN': 64,  'TK': 64}, num_warps=4, num_stages=4),
        triton.Config({'TM': 128, 'TN': 64,  'TK': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * TM + tl.arange(0, TM)
    rn = pid_n * TN + tl.arange(0, TN)

    acc = tl.zeros((TM, TN), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, TK)):
        rk = k * TK + tl.arange(0, TK)
        # x tile: (TM, TK)  -- x is (M, K) row-major
        x_mask = (rm[:, None] < M) & (rk[None, :] < K)
        x_tile = tl.load(x_ptr + rm[:, None] * K + rk[None, :], mask=x_mask, other=0.0)
        # w tile: (TK, TN)  -- w is (N, K) -> we want (K, N), so index as w[rn, rk]^T
        w_mask = (rn[None, :] < N) & (rk[:, None] < K)
        w_tile = tl.load(w_ptr + rn[None, :] * K + rk[:, None], mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # add bias
    bias = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: 3x3 conv (implicit GEMM)
# M = B*OH*OW, N = C_out, K = C_in * 3 * 3
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'TM': 64,  'TN': 64,  'TK': 32}, num_warps=4, num_stages=3),
        triton.Config({'TM': 128, 'TN': 64,  'TK': 32}, num_warps=4, num_stages=3),
        triton.Config({'TM': 64,  'TN': 128, 'TK': 32}, num_warps=4, num_stages=3),
        triton.Config({'TM': 128, 'TN': 128, 'TK': 32}, num_warps=8, num_stages=3),
        triton.Config({'TM': 64,  'TN': 64,  'TK': 64}, num_warps=4, num_stages=4),
        triton.Config({'TM': 128, 'TN': 64,  'TK': 64}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K', 'C_IN', 'KH', 'KW', 'IN_H', 'IN_W', 'OUT_H', 'OUT_W', 'PAD_H', 'PAD_W'],
)
@triton.jit
def conv_nxn_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    B, C_IN, IN_H, IN_W,
    C_OUT, OUT_H, OUT_W,
    KH, KW, PAD_H, PAD_W,
    M, N, K,
    TM: tl.constexpr, TN: tl.constexpr, TK: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    rm = pid_m * TM + tl.arange(0, TM)  # output spatial positions
    rn = pid_n * TN + tl.arange(0, TN)  # output channels

    # decode rm -> (batch, out_h, out_w)
    ow = rm % OUT_W
    oh = (rm // OUT_W) % OUT_H
    ob = rm // (OUT_W * OUT_H)

    acc = tl.zeros((TM, TN), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, TK)):
        rk = k * TK + tl.arange(0, TK)
        # decode rk -> (c_in, kh, kw)
        kw_idx = rk % KW
        kh_idx = (rk // KW) % KH
        ci_idx = rk // (KW * KH)

        # input coordinates
        ih = oh[:, None] + kh_idx[None, :] - PAD_H
        iw = ow[:, None] + kw_idx[None, :] - PAD_W

        valid_m = rm[:, None] < M
        valid_k = rk[None, :] < K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        x_mask = valid_m & valid_k & valid_spatial

        x_idx = ob[:, None] * (C_IN * IN_H * IN_W) + ci_idx[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(x_ptr + x_idx, mask=x_mask, other=0.0)

        # weight: (C_OUT, C_IN, KH, KW) -> index as [rn, ci_idx, kh_idx, kw_idx]
        w_mask = (rn[None, :] < N) & (rk[:, None] < K)
        w_idx = rn[None, :] * K + rk[:, None]
        w_tile = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(b_ptr + rn, mask=rn < N, other=0.0)
    acc = acc + bias[None, :]

    out_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.store(out_ptr + rm[:, None] * N + rn[None, :], acc, mask=out_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 4: Cat + reformat
# Each branch output is (B, C_i, H, W) in row-major.
# We need to write into output (B, C_total, H, W).
# This is a pointwise copy with offset into the channel dimension.
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BS': 256},  num_warps=4),
        triton.Config({'BS': 512},  num_warps=8),
        triton.Config({'BS': 1024}, num_warps=8),
        triton.Config({'BS': 128},  num_warps=4),
    ],
    key=['N_ELEM'],
)
@triton.jit
def cat_copy_kernel(
    src_ptr, dst_ptr,
    B, C_SRC, H, W,
    C_DST, C_OFF,
    N_ELEM,
    BS: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BS + tl.arange(0, BS)
    mask = offs < N_ELEM

    # src is (B, C_SRC, H, W) row-major, flat index = offs
    # decode to (b, c, h, w)
    tmp = offs
    w_idx = tmp % W;   tmp = tmp // W
    h_idx = tmp % H;   tmp = tmp // H
    c_idx = tmp % C_SRC; b_idx = tmp // C_SRC

    val = tl.load(src_ptr + offs, mask=mask, other=0.0)

    # dst is (B, C_DST, H, W) row-major
    dst_idx = b_idx * (C_DST * H * W) + (c_idx + C_OFF) * (H * W) + h_idx * W + w_idx
    tl.store(dst_ptr + dst_idx, val, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: launch 1x1 conv
# x: (B, C_in, H, W), w: (C_out, C_in, 1, 1), b: (C_out,)
# out: (B, C_out, H, W) stored as (M, N) where M=B*H*W, N=C_out
# ─────────────────────────────────────────────────────────────────────────────
def launch_conv1x1(x, w, bias, B, C_in, H, W, C_out):
    M = B * H * W
    N = C_out
    K = C_in
    # x needs to be (M, K) = (B*H*W, C_in)
    # x is (B, C_in, H, W) -> permute to (B, H, W, C_in) -> reshape (M, K)
    x_t = x.permute(0, 2, 3, 1).contiguous().reshape(M, K)
    # w is (C_out, C_in, 1, 1) -> reshape (C_out, C_in) = (N, K)
    w_t = w.reshape(N, K)
    out_flat = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['TM']), triton.cdiv(N, meta['TN']))
    conv1x1_kernel[grid](x_t, w_t, bias, out_flat, M, N, K)
    # out_flat is (M, N) = (B*H*W, C_out) -> reshape (B, H, W, C_out) -> permute (B, C_out, H, W)
    return out_flat.reshape(B, H, W, C_out).permute(0, 3, 1, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: launch NxN conv
# ─────────────────────────────────────────────────────────────────────────────
def launch_conv_nxn(x, w, bias, B, C_in, H, W, C_out, KH, KW, pad):
    OH = H  # stride=1, padding keeps size
    OW = W
    M = B * OH * OW
    N = C_out
    K = C_in * KH * KW
    # w needs to be (C_out, K) = (N, K) for the kernel
    w_t = w.reshape(C_out, K)
    out_flat = torch.empty((M, N), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['TM']), triton.cdiv(N, meta['TN']))
    conv_nxn_kernel[grid](
        x, w_t, bias, out_flat,
        B, C_in, H, W,
        C_out, OH, OW,
        KH, KW, pad, pad,
        M, N, K,
    )
    return out_flat.reshape(B, OH, OW, C_out).permute(0, 3, 1, 2).contiguous()


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel function
# ─────────────────────────────────────────────────────────────────────────────
def kernel_function(
):
    B, C_in, H, W = x.shape
    device = x.device

    # ── Branch 1: 1x1 conv ──────────────────────────────────────────────────
    branch1 = launch_conv1x1(x, p_branch1x1_weight, p_branch1x1_bias,

    # ── Branch 2: 1x1 -> 3x3 conv ───────────────────────────────────────────
    b2_reduce = launch_conv1x1(x, p_branch3x3_0_weight, p_branch3x3_0_bias,
    branch2 = launch_conv_nxn(b2_reduce, p_branch3x3_1_weight, p_branch3x3_1_bias,

    # ── Branch 3: 1x1 -> 5x5 conv ───────────────────────────────────────────
    b3_reduce = launch_conv1x1(x, p_branch5x5_0_weight, p_branch5x5_0_bias,
    branch3 = launch_conv_nxn(b3_reduce, p_branch5x5_1_weight, p_branch5x5_1_bias,

    # ── Branch 4: maxpool 3x3 s1p1 -> 1x1 conv ──────────────────────────────
    pool_out = torch.empty_like(x)
    N_ELEM = B * C_in * H * W
    grid_pool = lambda meta: (triton.cdiv(N_ELEM, meta['BS']),)
    maxpool3x3_s1p1_kernel[grid_pool](x, pool_out, B, C_in, H, W, N_ELEM)

    branch4 = launch_conv1x1(pool_out, p_branch_pool_1_weight, p_branch_pool_1_bias,

    # ── Cat along channel dim: [192, 208, 48, 64] = 512 ─────────────────────
    C_total = 192 + 208 + 48 + 64  # = 512
    output = torch.empty((B, C_total, H, W), device=device, dtype=torch.float32)

    branches = [
        (branch1, 192, 0),
        (branch2, 208, 192),
        (branch3, 48,  192 + 208),
        (branch4, 64,  192 + 208 + 48),
    ]
    for src, C_src, C_off in branches:
        n_elem = B * C_src * H * W
        grid_cat = lambda meta, n=n_elem: (triton.cdiv(n, meta['BS']),)
        cat_copy_kernel[grid_cat](
            src, output,
            B, C_src, H, W,
            C_total, C_off,
            n_elem,
        )

    return output


def get_inputs():
    B, C_in, H, W = 10, 480, 224, 224
    return [
        torch.randn(192, 480, 1, 1),   # p_branch1x1_weight
        torch.randn(192),               # p_branch1x1_bias
        torch.randn(96, 480, 1, 1),    # p_branch3x3_0_weight
        torch.randn(96),                # p_branch3x3_0_bias
        torch.randn(208, 96, 3, 3),    # p_branch3x3_1_weight
        torch.randn(208),               # p_branch3x3_1_bias
        torch.randn(16, 480, 1, 1),    # p_branch5x5_0_weight
        torch.randn(16),                # p_branch5x5_0_bias
        torch.randn(48, 16, 5, 5),     # p_branch5x5_1_weight
        torch.randn(48),                # p_branch5x5_1_bias
        torch.randn(64, 480, 1, 1),    # p_branch_pool_1_weight
        torch.randn(64),                # p_branch_pool_1_bias
        torch.randn(B, C_in, H, W),    # x
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = [t.cuda() for t in get_inputs()]
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
        branch1x1 = F.conv2d(x, p_branch1x1_weight, p_branch1x1_bias)
        b2_r = F.conv2d(x, p_branch3x3_0_weight, p_branch3x3_0_bias)
        branch3x3 = F.conv2d(b2_r, p_branch3x3_1_weight, p_branch3x3_1_bias, padding=1)
        b3_r = F.conv2d(x, p_branch5x5_0_weight, p_branch5x5_0_bias)
        branch5x5 = F.conv2d(b3_r, p_branch5x5_1_weight, p_branch5x5_1_bias, padding=2)
        pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = F.conv2d(pool, p_branch_pool_1_weight, p_branch_pool_1_bias)
        ref = torch.cat([branch1x1, branch3x3, branch5x5, branch_pool], dim=1)

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")