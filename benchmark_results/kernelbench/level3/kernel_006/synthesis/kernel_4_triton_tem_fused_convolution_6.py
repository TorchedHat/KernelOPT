import torch
import triton
import triton.language as tl
import math

# ─── 1x1 conv as GEMM ────────────────────────────────────────────────────────
# Input x: (N, C, H, W) -> reshape to (N*H*W, C)
# Weight: (Co, C, 1, 1) -> reshape to (Co, C)
# Output: (N*H*W, Co) -> reshape to (N, Co, H, W)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
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
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc, allow_tf32=False)

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_out = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N)

    bias = tl.load(b_ptr + offs_n_out, mask=offs_n_out < N, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m_out[:, None] * N + offs_n_out[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# ─── Max Pool 2D (3x3, stride=1, pad=1) ──────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def maxpool2d_3x3_s1_p1_kernel(
    x_ptr, out_ptr,
    N_TOTAL,
    H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # offs indexes into (N*C, H*W) flattened
    HW = H * W
    nc = offs // HW
    hw = offs % HW
    oh = hw // W
    ow = hw % W

    base = nc * HW

    # 3x3 window with padding=1
    val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for dh in range(3):
        for dw in range(3):
            ih = oh + dh - 1
            iw = ow + dw - 1
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            idx = base + ih * W + iw
            v = tl.load(x_ptr + idx, mask=valid, other=float('-inf'))
            val = tl.maximum(val, v)

    tl.store(out_ptr + offs, val, mask=mask)


# ─── Implicit GEMM Conv2d (general) ──────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def implicit_gemm_conv2d_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    N, C, H, W,
    Co, KH, KW,
    OH, OW,
    PAD_H, PAD_W,
    STRIDE_H, STRIDE_W,
    M, N_out, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, oh, ow)
    OHW = OH * OW
    batch = offs_m // OHW
    ohw = offs_m % OHW
    oh = ohw // OW
    ow = ohw % OW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Decode K -> (c, kh, kw)
        c = offs_k % C
        kh = (offs_k // C) % KH
        kw = offs_k // (C * KH)

        # Input coordinates
        ih = oh[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = ow[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_x = (offs_m[:, None] < M) & (offs_k[None, :] < K) & \
                  (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & \
                  (batch[:, None] < N)

        x_idx = batch[:, None] * (C * H * W) + c[None, :] * (H * W) + ih * W + iw
        x = tl.load(x_ptr + x_idx, mask=valid_x, other=0.0)

        # Weight: (Co, C, KH, KW) -> index [offs_n, c, kh, kw]
        valid_w = (offs_n[None, :] < N_out) & (offs_k[:, None] < K)
        w_idx = offs_n[None, :] * K + offs_k[:, None]
        w = tl.load(w_ptr + w_idx, mask=valid_w, other=0.0)

        acc = tl.dot(x, w, acc, allow_tf32=False)

    offs_m_out = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_out = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_out = (offs_m_out[:, None] < M) & (offs_n_out[None, :] < N_out)

    bias = tl.load(b_ptr + offs_n_out, mask=offs_n_out < N_out, other=0.0)
    acc = acc + bias[None, :]

    out_ptrs = out_ptr + offs_m_out[:, None] * N_out + offs_n_out[None, :]
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# ─── Cat kernel: write 4 tensors into output along channel dim ───────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['TOTAL'],
)
@triton.jit
def cat4_kernel(
    p0, p1, p2, p3,
    out_ptr,
    C0, C1, C2, C3,
    HW,
    TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL

    # offs indexes (N * C_total * HW) flattened
    C_total = C0 + C1 + C2 + C3
    n = offs // (C_total * HW)
    rem = offs % (C_total * HW)
    c = rem // HW
    hw = rem % HW

    # Determine which branch
    in_b0 = c < C0
    in_b1 = (c >= C0) & (c < C0 + C1)
    in_b2 = (c >= C0 + C1) & (c < C0 + C1 + C2)
    in_b3 = c >= C0 + C1 + C2

    c0 = c
    c1 = c - C0
    c2 = c - C0 - C1
    c3 = c - C0 - C1 - C2

    idx0 = n * C0 * HW + c0 * HW + hw
    idx1 = n * C1 * HW + c1 * HW + hw
    idx2 = n * C2 * HW + c2 * HW + hw
    idx3 = n * C3 * HW + c3 * HW + hw

    v0 = tl.load(p0 + idx0, mask=mask & in_b0, other=0.0)
    v1 = tl.load(p1 + idx1, mask=mask & in_b1, other=0.0)
    v2 = tl.load(p2 + idx2, mask=mask & in_b2, other=0.0)
    v3 = tl.load(p3 + idx3, mask=mask & in_b3, other=0.0)

    val = tl.where(in_b0, v0, tl.where(in_b1, v1, tl.where(in_b2, v2, v3)))
    tl.store(out_ptr + offs, val, mask=mask)


def run_conv1x1(x_nhwc, w, b, Co):
    """x_nhwc: (M, C), w: (Co, C), b: (Co,) -> out: (M, Co)"""
    M, C = x_nhwc.shape
    N = Co
    K = C
    out = torch.empty((M, N), device=x_nhwc.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    conv1x1_bias_kernel[grid](
        x_nhwc, w, b, out,
        M, N, K,
    )
    return out


def run_implicit_gemm(x, w, b, pad_h, pad_w, stride_h=1, stride_w=1):
    """x: (N,C,H,W), w: (Co,C,KH,KW), b: (Co,) -> out: (N,Co,OH,OW)"""
    N, C, H, W = x.shape
    Co, _, KH, KW = w.shape
    OH = (H + 2 * pad_h - KH) // stride_h + 1
    OW = (W + 2 * pad_w - KW) // stride_w + 1
    M = N * OH * OW
    N_out = Co
    K = C * KH * KW

    # Flatten weight to (Co, K)
    w_flat = w.reshape(Co, K).contiguous()
    x_cont = x.contiguous()

    out_flat = torch.empty((M, N_out), device=x.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N_out, meta['BLOCK_N']))
    implicit_gemm_conv2d_kernel[grid](
        x_cont, w_flat, b, out_flat,
        N, C, H, W,
        Co, KH, KW,
        OH, OW,
        pad_h, pad_w,
        stride_h, stride_w,
        M, N_out, K,
    )
    return out_flat.reshape(N, OH, OW, Co).permute(0, 3, 1, 2).contiguous()


def run_maxpool(x):
    """x: (N,C,H,W) -> out: (N,C,H,W) with 3x3, stride=1, pad=1"""
    N, C, H, W = x.shape
    x_cont = x.contiguous()
    out = torch.empty_like(x_cont)
    N_TOTAL = N * C * H * W
    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    maxpool2d_3x3_s1_p1_kernel[grid](
        x_cont, out,
        N_TOTAL, H, W,
    )
    return out


def kernel_function(
    p_branch1x1_weight, p_branch1x1_bias,
    p_branch3x3_0_weight, p_branch3x3_0_bias,
    p_branch3x3_1_weight, p_branch3x3_1_bias,
    p_branch5x5_0_weight, p_branch5x5_0_bias,
    p_branch5x5_1_weight, p_branch5x5_1_bias,
    p_branch_pool_1_weight, p_branch_pool_1_bias,
    x,
):
    N, C, H, W = x.shape

    # Prepare x as (M, C) for 1x1 convolutions
    M = N * H * W
    x_cont = x.contiguous()
    # For 1x1 conv: reshape x to (N, C, H*W) then permute to (N, H*W, C) -> (M, C)
    x_flat = x_cont.permute(0, 2, 3, 1).reshape(M, C)  # (M, C)

    # Branch 1x1: (M, 192)
    w1 = p_branch1x1_weight.reshape(192, C).contiguous()
    b1 = p_branch1x1_bias.contiguous()
    branch1x1_flat = run_conv1x1(x_flat, w1, b1, 192)
    branch1x1 = branch1x1_flat.reshape(N, H, W, 192).permute(0, 3, 1, 2).contiguous()

    # Branch 3x3 step 1: 1x1 conv -> (M, 96)
    w3_0 = p_branch3x3_0_weight.reshape(96, C).contiguous()
    b3_0 = p_branch3x3_0_bias.contiguous()
    branch3x3_0_flat = run_conv1x1(x_flat, w3_0, b3_0, 96)
    branch3x3_0 = branch3x3_0_flat.reshape(N, H, W, 96).permute(0, 3, 1, 2).contiguous()

    # Branch 3x3 step 2: 3x3 conv -> (N, 208, H, W)
    branch3x3 = run_implicit_gemm(branch3x3_0, p_branch3x3_1_weight, p_branch3x3_1_bias, 1, 1)

    # Branch 5x5 step 1: 1x1 conv -> (M, 16)
    w5_0 = p_branch5x5_0_weight.reshape(16, C).contiguous()
    b5_0 = p_branch5x5_0_bias.contiguous()
    branch5x5_0_flat = run_conv1x1(x_flat, w5_0, b5_0, 16)
    branch5x5_0 = branch5x5_0_flat.reshape(N, H, W, 16).permute(0, 3, 1, 2).contiguous()

    # Branch 5x5 step 2: 5x5 conv -> (N, 48, H, W)
    branch5x5 = run_implicit_gemm(branch5x5_0, p_branch5x5_1_weight, p_branch5x5_1_bias, 2, 2)

    # Branch pool: MaxPool2d(3x3, stride=1, pad=1) -> (N, 480, H, W)
    pooled = run_maxpool(x_cont)

    # Branch pool 1x1 conv: (M, 64)
    pooled_flat = pooled.permute(0, 2, 3, 1).reshape(M, 480)
    w_pool = p_branch_pool_1_weight.reshape(64, 480).contiguous()
    b_pool = p_branch_pool_1_bias.contiguous()
    branch_pool_flat = run_conv1x1(pooled_flat, w_pool, b_pool, 64)
    branch_pool = branch_pool_flat.reshape(N, H, W, 64).permute(0, 3, 1, 2).contiguous()

    # Cat: [branch1x1(192), branch3x3(208), branch5x5(48), branch_pool(64)] -> 512
    C0, C1, C2, C3 = 192, 208, 48, 64
    C_total = C0 + C1 + C2 + C3  # 512
    HW = H * W
    TOTAL = N * C_total * HW
    out = torch.empty((N, C_total, H, W), device=x.device, dtype=torch.float32)

    grid = lambda meta: (triton.cdiv(TOTAL, meta['BLOCK_SIZE']),)
    cat4_kernel[grid](
        branch1x1, branch3x3, branch5x5, branch_pool,
        out,
        C0, C1, C2, C3,
        HW,
        TOTAL,
    )
    return out


def get_inputs():
    torch.manual_seed(0)
    in_channels = 480
    out_1x1 = 192
    reduce_3x3 = 96
    out_3x3 = 208
    reduce_5x5 = 16
    out_5x5 = 48
    pool_proj = 64
    batch_size = 10
    height = 224
    width = 224

    p_branch1x1_weight = torch.randn(out_1x1, in_channels, 1, 1)
    p_branch1x1_bias = torch.randn(out_1x1)
    p_branch3x3_0_weight = torch.randn(reduce_3x3, in_channels, 1, 1)
    p_branch3x3_0_bias = torch.randn(reduce_3x3)
    p_branch3x3_1_weight = torch.randn(out_3x3, reduce_3x3, 3, 3)
    p_branch3x3_1_bias = torch.randn(out_3x3)
    p_branch5x5_0_weight = torch.randn(reduce_5x5, in_channels, 1, 1)
    p_branch5x5_0_bias = torch.randn(reduce_5x5)
    p_branch5x5_1_weight = torch.randn(out_5x5, reduce_5x5, 5, 5)
    p_branch5x5_1_bias = torch.randn(out_5x5)
    p_branch_pool_1_weight = torch.randn(pool_proj, in_channels, 1, 1)
    p_branch_pool_1_bias = torch.randn(pool_proj)
    x = torch.randn(batch_size, in_channels, height, width)

    return [
        p_branch1x1_weight.cuda(), p_branch1x1_bias.cuda(),
        p_branch3x3_0_weight.cuda(), p_branch3x3_0_bias.cuda(),
        p_branch3x3_1_weight.cuda(), p_branch3x3_1_bias.cuda(),
        p_branch5x5_0_weight.cuda(), p_branch5x5_0_bias.cuda(),
        p_branch5x5_1_weight.cuda(), p_branch5x5_1_bias.cuda(),
        p_branch_pool_1_weight.cuda(), p_branch_pool_1_bias.cuda(),
        x.cuda(),
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

        branch1x1_ref = F.conv2d(x, p_branch1x1_weight, p_branch1x1_bias)
        branch3x3_0_ref = F.conv2d(x, p_branch3x3_0_weight, p_branch3x3_0_bias)
        branch3x3_ref = F.conv2d(branch3x3_0_ref, p_branch3x3_1_weight, p_branch3x3_1_bias, padding=1)
        branch5x5_0_ref = F.conv2d(x, p_branch5x5_0_weight, p_branch5x5_0_bias)
        branch5x5_ref = F.conv2d(branch5x5_0_ref, p_branch5x5_1_weight, p_branch5x5_1_bias, padding=2)
        pooled_ref = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool_ref = F.conv2d(pooled_ref, p_branch_pool_1_weight, p_branch_pool_1_bias)
        ref = torch.cat([branch1x1_ref, branch3x3_ref, branch5x5_ref, branch_pool_ref], dim=1)

        print(f"Output shape: {output.shape}, Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")