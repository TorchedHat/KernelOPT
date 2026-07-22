import torch
import triton
import triton.language as tl
import math

# ─── Max Pool 3x3 stride=1 pad=1 kernel ───────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 4}, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 4}, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 8}, num_warps=8),
    ],
    key=['N', 'C', 'H', 'W'],
)
@triton.jit
def maxpool3x3_s1p1_kernel(
    x_ptr, out_ptr,
    N, C, H, W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Each program handles BLOCK_M spatial positions for BLOCK_N channels
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_b = tl.program_id(2)

    HW = H * W
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_c = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < HW
    mask_c = offs_c < C

    oh = offs_m // W
    ow = offs_m % W

    result = tl.full((BLOCK_M, BLOCK_N), float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh - 1 + kh
            iw = ow - 1 + kw
            valid_h = (ih >= 0) & (ih < H)
            valid_w = (iw >= 0) & (iw < W)
            valid = valid_h[:, None] & valid_w[:, None] & mask_m[:, None] & mask_c[None, :]
            idx = pid_b * C * HW + offs_c[None, :] * HW + ih[:, None] * W + iw[:, None]
            val = tl.load(x_ptr + idx, mask=valid, other=float('-inf'))
            result = tl.maximum(result, val)

    out_mask = mask_m[:, None] & mask_c[None, :]
    out_idx = pid_b * C * HW + offs_c[None, :] * HW + oh[:, None] * W + ow[:, None]
    tl.store(out_ptr + out_idx, result, mask=out_mask)


# ─── 1x1 Conv as GEMM (bias add fused) ────────────────────────────────────────
# Input: [N, Ci, H, W] -> reshape to [N*H*W, Ci]
# Weight: [Co, Ci] -> transpose to [Ci, Co]
# Output: [N*H*W, Co] -> reshape to [N, Co, H, W]
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bias_kernel(
    A_ptr, B_ptr, bias_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        mask_k = k_offs < K

        a_ptrs = A_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = B_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn

        a = tl.load(a_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        b = tl.load(b_ptrs, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(a, b, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    c_ptrs = C_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc, mask=mask_m[:, None] & mask_n[None, :])


# ─── Implicit GEMM for 3x3 conv (pad=1, stride=1) ─────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv3x3_p1_s1_kernel(
    X_ptr, W_ptr, bias_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    PAD_H, PAD_W,
    M_TOTAL, TOTAL_K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M_TOTAL
    mask_n = offs_n < OUT_C

    # Decode M -> (batch, out_h, out_w)
    HW = OUT_H * OUT_W
    batch = offs_m // HW
    rem = offs_m % HW
    out_h = rem // OUT_W
    out_w = rem % OUT_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOTAL_K

        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        ih = out_h[:, None] + kh[None, :] - PAD_H
        iw = out_w[:, None] + kw[None, :] - PAD_W

        valid = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & mask_m[:, None] & mask_k[None, :]

        x_idx = batch[:, None] * IN_C * IN_H * IN_W + in_c[None, :] * IN_H * IN_W + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=valid, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> index [k, n] = w[n, in_c, kh, kw]
        w_idx = offs_n[None, :] * IN_C * KERNEL_H * KERNEL_W + in_c[:, None] * KERNEL_H * KERNEL_W + kh[:, None] * KERNEL_W + kw[:, None]
        w_tile = tl.load(W_ptr + w_idx, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Store: output is [BATCH, OUT_C, OUT_H, OUT_W]
    y_idx = batch[:, None] * OUT_C * OUT_H * OUT_W + offs_n[None, :] * OUT_H * OUT_W + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(Y_ptr + y_idx, acc, mask=mask_m[:, None] & mask_n[None, :])


# ─── Implicit GEMM for 5x5 conv (pad=2, stride=1) ─────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv5x5_p2_s1_kernel(
    X_ptr, W_ptr, bias_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    PAD_H, PAD_W,
    M_TOTAL, TOTAL_K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M_TOTAL
    mask_n = offs_n < OUT_C

    HW = OUT_H * OUT_W
    batch = offs_m // HW
    rem = offs_m % HW
    out_h = rem // OUT_W
    out_w = rem % OUT_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOTAL_K

        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        ih = out_h[:, None] + kh[None, :] - PAD_H
        iw = out_w[:, None] + kw[None, :] - PAD_W

        valid = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & mask_m[:, None] & mask_k[None, :]

        x_idx = batch[:, None] * IN_C * IN_H * IN_W + in_c[None, :] * IN_H * IN_W + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=valid, other=0.0)

        w_idx = offs_n[None, :] * IN_C * KERNEL_H * KERNEL_W + in_c[:, None] * KERNEL_H * KERNEL_W + kh[:, None] * KERNEL_W + kw[:, None]
        w_tile = tl.load(W_ptr + w_idx, mask=mask_k[:, None] & mask_n[None, :], other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    y_idx = batch[:, None] * OUT_C * OUT_H * OUT_W + offs_n[None, :] * OUT_H * OUT_W + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(Y_ptr + y_idx, acc, mask=mask_m[:, None] & mask_n[None, :])


# ─── Cat kernel: write 4 tensors into output along channel dim ────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def cat4_kernel(
    a_ptr, b_ptr, c_ptr, d_ptr,
    out_ptr,
    Ca, Cb, Cc, Cd,
    HW,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # offs indexes into [N, C_total, HW] flattened
    C_total = Ca + Cb + Cc + Cd
    n = offs // (C_total * HW)
    rem = offs % (C_total * HW)
    c = rem // HW
    hw = rem % HW

    # Determine which branch
    in_a = c < Ca
    in_b = (c >= Ca) & (c < Ca + Cb)
    in_c = (c >= Ca + Cb) & (c < Ca + Cb + Cc)
    in_d = c >= Ca + Cb + Cc

    ca = c
    cb = c - Ca
    cc = c - Ca - Cb
    cd = c - Ca - Cb - Cc

    idx_a = n * Ca * HW + ca * HW + hw
    idx_b = n * Cb * HW + cb * HW + hw
    idx_c = n * Cc * HW + cc * HW + hw
    idx_d = n * Cd * HW + cd * HW + hw

    val_a = tl.load(a_ptr + idx_a, mask=mask & in_a, other=0.0)
    val_b = tl.load(b_ptr + idx_b, mask=mask & in_b, other=0.0)
    val_c = tl.load(c_ptr + idx_c, mask=mask & in_c, other=0.0)
    val_d = tl.load(d_ptr + idx_d, mask=mask & in_d, other=0.0)

    val = tl.where(in_a, val_a, tl.where(in_b, val_b, tl.where(in_c, val_c, val_d)))
    tl.store(out_ptr + offs, val, mask=mask)


def kernel_function(
):
    device = x.device
    dtype = x.dtype

    N, Ci, H, W = x.shape  # 10, 480, 224, 224
    HW = H * W
    M_total = N * HW  # 10 * 224 * 224 = 501760

    # ── Branch 1x1: conv(x, w1x1) + b1x1 ──────────────────────────────────────
    Co_1x1 = 192
    out_1x1 = torch.empty(N, Co_1x1, H, W, device=device, dtype=dtype)
    # x as [M, K], w as [K, N]
    x_2d = x.permute(0, 2, 3, 1).reshape(M_total, Ci)  # [M, Ci]
    w_1x1 = p_branch1x1_weight.reshape(Co_1x1, Ci)      # [Co, Ci]

    out_1x1_2d = torch.empty(M_total, Co_1x1, device=device, dtype=dtype)
    grid_1x1 = lambda meta: (triton.cdiv(M_total, meta['BLOCK_M']) * triton.cdiv(Co_1x1, meta['BLOCK_N']),)
    conv1x1_bias_kernel[grid_1x1](
        x_2d, w_1x1, p_branch1x1_bias, out_1x1_2d,
        M_total, Co_1x1, Ci,
    )
    # Wait - w_1x1 is [Co, Ci], we need A[M,K] x B[K,N] where K=Ci, N=Co
    # So B should be [Ci, Co] = w_1x1.T
    # Let me redo: stride_bk=1 (along Ci dim), stride_bn=Ci (along Co dim) for w_1x1[Co,Ci]
    # Actually w_1x1[k, n] where k=Ci index, n=Co index -> w_1x1.T[k,n] = w_1x1[n,k]
    # w_1x1 shape [Co, Ci]: element [co, ci] at offset co*Ci + ci
    # We want B[k, n] = w_1x1[n, k] -> stride_bk = 1 (ci stride in w_1x1), stride_bn = Ci
    out_1x1_2d2 = torch.empty(M_total, Co_1x1, device=device, dtype=dtype)
    conv1x1_bias_kernel[grid_1x1](
        x_2d, w_1x1, p_branch1x1_bias, out_1x1_2d2,
        M_total, Co_1x1, Ci,
        Ci, 1,    # stride_am=Ci (row stride), stride_ak=1 (col stride) for x_2d[M,Ci]
        1, Ci,    # stride_bk=1 (ci stride in w_1x1[Co,Ci] transposed), stride_bn=Ci
        Co_1x1, 1,
    )
    out_1x1 = out_1x1_2d2.reshape(N, H, W, Co_1x1).permute(0, 3, 1, 2).contiguous()

    # ── Branch 3x3 step 0: 1x1 conv ────────────────────────────────────────────
    Co_3x3_0 = 96
    w_3x3_0 = p_branch3x3_0_weight.reshape(Co_3x3_0, Ci)
    out_3x3_0_2d = torch.empty(M_total, Co_3x3_0, device=device, dtype=dtype)
    grid_3x3_0 = lambda meta: (triton.cdiv(M_total, meta['BLOCK_M']) * triton.cdiv(Co_3x3_0, meta['BLOCK_N']),)
    conv1x1_bias_kernel[grid_3x3_0](
        x_2d, w_3x3_0, p_branch3x3_0_bias, out_3x3_0_2d,
        M_total, Co_3x3_0, Ci,
        Ci, 1,
        1, Ci,
        Co_3x3_0, 1,
    )
    out_3x3_0 = out_3x3_0_2d.reshape(N, H, W, Co_3x3_0).permute(0, 3, 1, 2).contiguous()

    # ── Branch 3x3 step 1: 3x3 conv (pad=1) ────────────────────────────────────
    Co_3x3_1 = 208
    out_3x3_1 = torch.empty(N, Co_3x3_1, H, W, device=device, dtype=dtype)
    M3 = N * H * W
    TOTAL_K3 = Co_3x3_0 * 3 * 3  # 96*9=864
    grid_3x3_1 = (triton.cdiv(M3, 64), triton.cdiv(Co_3x3_1, 64))
    conv3x3_p1_s1_kernel[grid_3x3_1](
        out_3x3_0, p_branch3x3_1_weight, p_branch3x3_1_bias, out_3x3_1,
        N, Co_3x3_0, H, W,
        Co_3x3_1, H, W,
        3, 3,
        1, 1,
        M3, TOTAL_K3,
    )

    # ── Branch 5x5 step 0: 1x1 conv ────────────────────────────────────────────
    Co_5x5_0 = 16
    w_5x5_0 = p_branch5x5_0_weight.reshape(Co_5x5_0, Ci)
    out_5x5_0_2d = torch.empty(M_total, Co_5x5_0, device=device, dtype=dtype)
    grid_5x5_0 = lambda meta: (triton.cdiv(M_total, meta['BLOCK_M']) * triton.cdiv(Co_5x5_0, meta['BLOCK_N']),)
    conv1x1_bias_kernel[grid_5x5_0](
        x_2d, w_5x5_0, p_branch5x5_0_bias, out_5x5_0_2d,
        M_total, Co_5x5_0, Ci,
        Ci, 1,
        1, Ci,
        Co_5x5_0, 1,
    )
    out_5x5_0 = out_5x5_0_2d.reshape(N, H, W, Co_5x5_0).permute(0, 3, 1, 2).contiguous()

    # ── Branch 5x5 step 1: 5x5 conv (pad=2) ────────────────────────────────────
    Co_5x5_1 = 48
    out_5x5_1 = torch.empty(N, Co_5x5_1, H, W, device=device, dtype=dtype)
    M5 = N * H * W
    TOTAL_K5 = Co_5x5_0 * 5 * 5  # 16*25=400
    grid_5x5_1 = (triton.cdiv(M5, 64), triton.cdiv(Co_5x5_1, 32))
    conv5x5_p2_s1_kernel[grid_5x5_1](
        out_5x5_0, p_branch5x5_1_weight, p_branch5x5_1_bias, out_5x5_1,
        N, Co_5x5_0, H, W,
        Co_5x5_1, H, W,
        5, 5,
        2, 2,
        M5, TOTAL_K5,
    )

    # ── Branch pool: maxpool 3x3 s=1 p=1 ───────────────────────────────────────
    pool_out = torch.empty(N, Ci, H, W, device=device, dtype=dtype)
    BLOCK_M_pool = 128
    BLOCK_N_pool = 4
    grid_pool = (triton.cdiv(H