import torch
import triton
import triton.language as tl
import math

# ─── 1*1 conv as GEMM ────────────────────────────────────────────────────────
# Input x: (N, Ci, H, W) -> reshape to (N*H*W, Ci)
# Weight:  (Co, Ci, 1, 1) -> reshape to (Co, Ci)
# Output:  (N*H*W, Co) + bias -> reshape to (N, Co, H, W)

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bias_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
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
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk

        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)

        x = tl.load(x_ptrs, mask=x_mask, other=0.0)
        w = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x, w, acc=acc)

    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + b[None, :]

    # Store
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


# ─── Max Pool 3*3 stride=1 pad=1 ─────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def maxpool3x3_kernel(
    X_ptr, Y_ptr,
    N, C, H, W,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # Decode flat index -> (n, c, oh, ow)
    ow = offs % W
    tmp = offs // W
    oh = tmp % H
    tmp2 = tmp // H
    c = tmp2 % C
    n = tmp2 // C

    # Max pool 3*3, stride=1, pad=1
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh + kh - 1
            iw = ow + kw - 1
            valid = mask & (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W)
            idx = n * (C * H * W) + c * (H * W) + ih * W + iw
            val = tl.load(X_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(Y_ptr + offs, max_val, mask=mask)


# ─── Implicit GEMM for 3*3 conv (pad=1, stride=1) ────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv3x3_bias_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    M_TOTAL, TOTAL_K,
    PAD_H, PAD_W,
    STRIDE_H, STRIDE_W,
    KERNEL_H, KERNEL_W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_x = (
            (offs_m[:, None] < M_TOTAL) &
            (offs_k[None, :] < TOTAL_K) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W)
        )

        x_idx = batch[:, None] * (IN_C * IN_H * IN_W) + in_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x = tl.load(X_ptr + x_idx, mask=valid_x, other=0.0)

        # Weight: (OUT_C, IN_C, KH, KW) -> index [n, in_c, kh, kw]
        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_idx = offs_n[None, :] * TOTAL_K + offs_k[:, None]
        w = tl.load(W_ptr + w_idx, mask=valid_w, other=0.0)

        acc = tl.dot(x, w, acc=acc)

    # Add bias
    b = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    # Store in NCHW: output[batch, n, oh, ow]
    out_idx = batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(Y_ptr + out_idx, acc, mask=out_mask)


# ─── Implicit GEMM for 5*5 conv (pad=2, stride=1) ────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=4),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv5x5_bias_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    M_TOTAL, TOTAL_K,
    PAD_H, PAD_W,
    STRIDE_H, STRIDE_W,
    KERNEL_H, KERNEL_W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_x = (
            (offs_m[:, None] < M_TOTAL) &
            (offs_k[None, :] < TOTAL_K) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W)
        )

        x_idx = batch[:, None] * (IN_C * IN_H * IN_W) + in_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x = tl.load(X_ptr + x_idx, mask=valid_x, other=0.0)

        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_idx = offs_n[None, :] * TOTAL_K + offs_k[:, None]
        w = tl.load(W_ptr + w_idx, mask=valid_w, other=0.0)

        acc = tl.dot(x, w, acc=acc)

    b = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + b[None, :]

    out_mask = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    out_idx = batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(Y_ptr + out_idx, acc, mask=out_mask)


# ─── Cat kernel: concatenate along channel dim ───────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['total'],
)
@triton.jit
def cat_kernel(
    B1_ptr, B2_ptr, B3_ptr, B4_ptr,
    OUT_ptr,
    N, C1, C2, C3, C4,
    HW,
    total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    # Decode: total = N * (C1+C2+C3+C4) * HW
    C_total = C1 + C2 + C3 + C4
    hw_idx = offs % HW
    tmp = offs // HW
    c_idx = tmp % C_total
    n_idx = tmp // C_total

    # Determine which branch
    in_b1 = c_idx < C1
    in_b2 = (c_idx >= C1) & (c_idx < C1 + C2)
    in_b3 = (c_idx >= C1 + C2) & (c_idx < C1 + C2 + C3)
    in_b4 = c_idx >= C1 + C2 + C3

    c_b1 = c_idx
    c_b2 = c_idx - C1
    c_b3 = c_idx - C1 - C2
    c_b4 = c_idx - C1 - C2 - C3

    idx_b1 = n_idx * C1 * HW + c_b1 * HW + hw_idx
    idx_b2 = n_idx * C2 * HW + c_b2 * HW + hw_idx
    idx_b3 = n_idx * C3 * HW + c_b3 * HW + hw_idx
    idx_b4 = n_idx * C4 * HW + c_b4 * HW + hw_idx

    v1 = tl.load(B1_ptr + idx_b1, mask=mask & in_b1, other=0.0)
    v2 = tl.load(B2_ptr + idx_b2, mask=mask & in_b2, other=0.0)
    v3 = tl.load(B3_ptr + idx_b3, mask=mask & in_b3, other=0.0)
    v4 = tl.load(B4_ptr + idx_b4, mask=mask & in_b4, other=0.0)

    val = tl.where(in_b1, v1, tl.where(in_b2, v2, tl.where(in_b3, v3, v4)))
    tl.store(OUT_ptr + offs, val, mask=mask)


def run_conv1x1(x_flat, weight, bias):
    """x_flat: (M, K), weight: (Co, K), bias: (Co,) -> (M, Co)"""
    M, K = x_flat.shape
    Co = weight.shape[0]
    out = torch.empty((M, Co), device=x_flat.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(Co, meta['BLOCK_N']))
    conv1x1_bias_kernel[grid](
        x_flat, weight, bias, out,
        M, Co, K,
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
    N, C, H, W = x.shape  # 10, 480, 224, 224
    HW = H * W
    M = N * H * W  # 501760

    # Flatten x for 1x1 convolutions: (N*H*W, C) in NHWC order
    # x is NCHW, we need NHWC for the matmul
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
    x_flat = x_nhwc.reshape(M, C)  # (M, K)

    # ── Branch 1*1 ──────────────────────────────────────────────────────────
    w1 = p_branch1x1_weight.reshape(192, C)  # (192, 480)
    b1 = p_branch1x1_bias
    out1_flat = run_conv1x1(x_flat, w1, b1)  # (M, 192)
    branch1x1 = out1_flat.reshape(N, H, W, 192).permute(0, 3, 1, 2).contiguous()  # (N, 192, H, W)

    # ── Branch 3*3: reduce then 3*3 conv ────────────────────────────────────
    w3_0 = p_branch3x3_0_weight.reshape(96, C)
    b3_0 = p_branch3x3_0_bias
    out3_0_flat = run_conv1x1(x_flat, w3_0, b3_0)  # (M, 96)
    # Reshape to NCHW for 3*3 conv
    x3_0 = out3_0_flat.reshape(N, H, W, 96).permute(0, 3, 1, 2).contiguous()  # (N, 96, H, W)

    # 3*3 conv: IN_C=96, OUT_C=208, pad=1, stride=1
    IN_C3 = 96
    OUT_C3 = 208
    TOTAL_K3 = IN_C3 * 3 * 3  # 864
    M3 = N * H * W
    w3_1_flat = p_branch3x3_1_weight.reshape(OUT_C3, TOTAL_K3)  # (208, 864)
    # Transpose for kernel: (TOTAL_K, OUT_C)
    w3_1_T = w3_1_flat.t().contiguous()  # (864, 208)
    branch3x3 = torch.empty((N, OUT_C3, H, W), device=x.device, dtype=torch.float32)
    grid3 = (triton.cdiv(M3, 128), triton.cdiv(OUT_C3, 128))
    conv3x3_bias_kernel[lambda meta: (triton.cdiv(M3, meta['BLOCK_M']), triton.cdiv(OUT_C3, meta['BLOCK_N']))](
        x3_0, w3_1_T, p_branch3x3_1_bias, branch3x3,
        N, IN_C3, H, W,
        OUT_C3, H, W,
        M3, TOTAL_K3,
    )

    # ── Branch 5*5: reduce then 5*5 conv ────────────────────────────────────
    w5_0 = p_branch5x5_0_weight.reshape(16, C)
    b5_0 = p_branch5x5_0_bias
    out5_0_flat = run_conv1x1(x_flat, w5_0, b5_0)  # (M, 16)
    x5_0 = out5_0_flat.reshape(N, H, W, 16).permute(0, 3, 1, 2).contiguous()  # (N, 16, H, W)

    IN_C5 = 16
    OUT_C5 = 48
    TOTAL_K5 = IN_C5 * 5 * 5  # 400
    M5 = N * H * W
    w5_1_flat = p_branch5x5_1_weight.reshape(OUT_C5, TOTAL_K5)
    w5_1_T = w5_1_flat.t().contiguous()  # (400, 48)
    branch5x5 = torch.empty((N, OUT_C5, H, W), device=x.device, dtype=torch.float32)
    conv5x5_bias_kernel[lambda meta: (triton.cdiv(M5, meta['BLOCK_M']), triton.cdiv(OUT_C5, meta['BLOCK_N']))](
        x5_0, w5_1_T, p_branch5x5_1_bias, branch5x5,
        N, IN_C5, H, W,
        OUT_C5, H, W,
        M5, TOTAL_K5,
    )

    # ── Branch pool: maxpool then 1*1 conv ──────────────────────────────────
    pool_out = torch.empty_like(x)
    total_pool = N * C * H * W
    maxpool3x3_kernel[lambda meta: (triton.cdiv(total_pool, meta['BLOCK_SIZE']),)](
        x, pool_out,
        N, C, H, W,
        total_pool,
    )

    # 1*1 conv on pool output
    pool_nhwc = pool_out.permute(0, 2, 3, 1).contiguous()
    pool_flat = pool_nhwc.reshape(M, C)
    w_pool = p_branch_pool_1_weight.reshape(64, C)
    b_pool = p_branch_pool_1_bias
    out_pool_flat = run_conv1x1(pool_flat, w_pool, b_pool)  # (M, 64)
    branch_pool = out_pool_flat.reshape(N, H, W, 64).permute(0, 3, 1, 2).contiguous()

    # ── Concatenate ──────────────────────────────────────────────────────────
    C1, C2, C3, C4 = 192, 208, 48, 64
    C_total = C1 + C2 + C3 + C4  # 512
    total_out = N * C_total * HW
    out = torch.empty((N, C_total, H, W), device=x.device, dtype=torch.float32)

    cat_kernel[lambda meta: (triton.cdiv(total_out, meta['BLOCK_SIZE']),)](
        branch1x1, branch3x3, branch5x5, branch_pool,
        out,
        N, C1, C2, C3, C4,
        HW,
        total_out,
    )

    return out


def get_inputs():
    torch.manual_seed(0)
    p_branch1x1_weight = torch.randn(192, 480, 1, 1, device='cuda')
    p_branch1x1_bias = torch.randn(192, device='cuda')
    p_branch3x3_0_weight = torch.randn(96, 480, 1, 1, device='cuda')
    p_branch3x3_0_bias = torch.randn(96, device='cuda')
    p_branch3x3_1_weight = torch.randn(208, 96, 3, 3, device='cuda')
    p_branch3x3_1_bias = torch.randn(208, device='cuda')
    p_branch5x5_0_weight = torch.randn(16, 480, 1, 1, device='cuda')
    p_branch5x5_0_bias = torch.randn(16, device='cuda')
    p_branch5x5_1_weight = torch.randn(48, 16, 5, 5, device='cuda')
    p_branch5x5_1_bias = torch.randn(48, device='cuda')
    p_branch_pool_1_weight = torch.randn(64, 480, 1, 1, device='cuda')