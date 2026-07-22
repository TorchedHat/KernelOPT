import torch
import triton
import triton.language as tl
import math

# ============================================================
# Kernel 1: 1x1 convolution as GEMM + bias
# Input: x [N, Ci, H, W] (NCHW), weight [Co, Ci, 1, 1], bias [Co]
# Output: [N, Co, H, W]
# Treat as GEMM: M = N*H*W, K = Ci, N_out = Co
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv1x1_bias_kernel(
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

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        # Load x tile [BLOCK_M, BLOCK_K]
        x_ptrs = x_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K)
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load w tile [BLOCK_K, BLOCK_N] - weight is [Co, Ci], we need [Ci, Co]
        w_ptrs = w_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_mask = (offs_n[None, :] < N) & (k_offs[:, None] < K)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    b_offs = offs_n
    b_mask = offs_n < N
    bias = tl.load(b_ptr + b_offs, mask=b_mask, other=0.0)
    acc = acc + bias[None, :]

    # Store output
    out_ptrs = out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(out_ptrs, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Kernel 2: Max Pool 2D (3x3, stride=1, pad=1)
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def maxpool2d_kernel(
    x_ptr, out_ptr,
    N_total,
    B, C, H, W,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    # Decode linear index to (b, c, oh, ow)
    ow = offs % W
    tmp = offs // W
    oh = tmp % H
    tmp2 = tmp // H
    c = tmp2 % C
    b = tmp2 // C

    # Max pool 3x3, stride=1, pad=1
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(3):
        for kw in range(3):
            ih = oh + kh - 1
            iw = ow + kw - 1
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
            in_idx = b * (C * H * W) + c * (H * W) + ih * W + iw
            val = tl.load(x_ptr + in_idx, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(out_ptr + offs, max_val, mask=mask)


# ============================================================
# Kernel 3: 3x3 convolution (implicit GEMM)
# Input: x [N, Ci, H, W], weight [Co, Ci, 3, 3], bias [Co]
# Output: [N, Co, H, W] with padding=1
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=5, num_warps=4),
    ],
    key=['M_total', 'N_out', 'K_total'],
)
@triton.jit
def conv_nxn_bias_kernel(
    x_ptr, w_ptr, b_ptr, out_ptr,
    M_total, N_out, K_total,
    B, IN_C, IN_H, IN_W,
    OUT_H, OUT_W,
    KH, KW,
    PAD_H, PAD_W,
    STRIDE_H, STRIDE_W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index to (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K_total, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Decode K index to (in_c, kh, kw)
        kw_idx = offs_k % KW
        kh_idx = (offs_k // KW) % KH
        in_c = offs_k // (KW * KH)

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W) & \
                (offs_m[:, None] < M_total) & (offs_k[None, :] < K_total)

        x_idx = batch[:, None] * (IN_C * IN_H * IN_W) + in_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(x_ptr + x_idx, mask=valid, other=0.0)

        # Weight: [Co, Ci, KH, KW] -> index [n, k] = w[n, in_c, kh, kw]
        w_idx = offs_n[None, :] * (IN_C * KH * KW) + in_c[:, None] * (KH * KW) + kh_idx[:, None] * KW + kw_idx[:, None]
        w_mask = (offs_n[None, :] < N_out) & (offs_k[:, None] < K_total)
        w_tile = tl.load(w_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(b_ptr + offs_n, mask=offs_n < N_out, other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_idx = batch[:, None] * (N_out * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    out_mask = (offs_m[:, None] < M_total) & (offs_n[None, :] < N_out)
    tl.store(out_ptr + out_idx, acc.to(tl.float32), mask=out_mask)


# ============================================================
# Kernel 4: Cat along dim=1 (channel dimension)
# Concatenate [10,192,224,224], [10,208,224,224], [10,48,224,224], [10,64,224,224]
# -> [10, 512, 224, 224]
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_stages=2, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def cat_kernel(
    in0_ptr, in1_ptr, in2_ptr, in3_ptr,
    out_ptr,
    N_total,
    C0, C1, C2, C3,
    HW,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    # Decode: out is [B, C_total, H, W] where C_total = C0+C1+C2+C3
    C_total = C0 + C1 + C2 + C3
    hw_idx = offs % HW
    tmp = offs // HW
    c_idx = tmp % C_total
    b_idx = tmp // C_total

    # Determine which input tensor and local channel
    in_c0 = c_idx
    in_c1 = c_idx - C0
    in_c2 = c_idx - C0 - C1
    in_c3 = c_idx - C0 - C1 - C2

    is_0 = c_idx < C0
    is_1 = (c_idx >= C0) & (c_idx < C0 + C1)
    is_2 = (c_idx >= C0 + C1) & (c_idx < C0 + C1 + C2)
    is_3 = c_idx >= C0 + C1 + C2

    idx0 = b_idx * C0 * HW + in_c0 * HW + hw_idx
    idx1 = b_idx * C1 * HW + in_c1 * HW + hw_idx
    idx2 = b_idx * C2 * HW + in_c2 * HW + hw_idx
    idx3 = b_idx * C3 * HW + in_c3 * HW + hw_idx

    val0 = tl.load(in0_ptr + idx0, mask=is_0 & mask, other=0.0)
    val1 = tl.load(in1_ptr + idx1, mask=is_1 & mask, other=0.0)
    val2 = tl.load(in2_ptr + idx2, mask=is_2 & mask, other=0.0)
    val3 = tl.load(in3_ptr + idx3, mask=is_3 & mask, other=0.0)

    val = tl.where(is_0, val0, tl.where(is_1, val1, tl.where(is_2, val2, val3)))
    tl.store(out_ptr + offs, val, mask=mask)


def kernel_function(
    p_branch1x1_weight, p_branch1x1_bias,
    p_branch3x3_0_weight, p_branch3x3_0_bias,
    p_branch3x3_1_weight, p_branch3x3_1_bias,
    p_branch5x5_0_weight, p_branch5x5_0_bias,
    p_branch5x5_1_weight, p_branch5x5_1_bias,
    p_branch_pool_1_weight, p_branch_pool_1_bias,
):
    B, Ci, H, W = x.shape  # 10, 480, 224, 224
    device = x.device
    dtype = x.dtype

    # ---- Branch 1x1: conv2d(x, w1x1) ----
    # [10, 480, 224, 224] x [192, 480] -> [10, 192, 224, 224]
    out_1x1_C = p_branch1x1_weight.shape[0]  # 192
    M_1x1 = B * H * W  # 10 * 224 * 224 = 501760
    K_1x1 = Ci  # 480
    N_1x1 = out_1x1_C  # 192

    # x as [M, K]: x is [B, Ci, H, W] -> [B*H*W, Ci] but stored as [B, Ci, H, W]
    # We need x in [B, H, W, Ci] layout for row-major access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [B, H, W, Ci]
    w1x1_flat = p_branch1x1_weight.view(out_1x1_C, Ci)  # [Co, Ci]

    branch1x1_out = torch.empty(B, out_1x1_C, H, W, device=device, dtype=dtype)
    branch1x1_flat = torch.empty(M_1x1, N_1x1, device=device, dtype=dtype)

    grid_1x1 = (triton.cdiv(M_1x1, 128), triton.cdiv(N_1x1, 128))
    conv1x1_bias_kernel[grid_1x1](
        x_nhwc, w1x1_flat, p_branch1x1_bias, branch1x1_flat,
        M_1x1, N_1x1, K_1x1,
        Ci, 1,  # stride_xm=Ci, stride_xk=1
        Ci, 1,  # stride_wn=Ci (w is [Co, Ci], so w[n, k] = w_flat[n*Ci + k]), stride_wk=1
    )
    # Reshape to [B, H, W, Co] -> [B, Co, H, W]
    branch1x1_out = branch1x1_flat.view(B, H, W, out_1x1_C).permute(0, 3, 1, 2).contiguous()

    # ---- Branch 3x3 step 0: conv2d(x, w3x3_0) 1x1 ----
    out_3x3_0_C = p_branch3x3_0_weight.shape[0]  # 96
    N_3x3_0 = out_3x3_0_C
    w3x3_0_flat = p_branch3x3_0_weight.view(out_3x3_0_C, Ci)

    branch3x3_0_flat = torch.empty(M_1x1, N_3x3_0, device=device, dtype=dtype)
    grid_3x3_0 = (triton.cdiv(M_1x1, 128), triton.cdiv(N_3x3_0, 128))
    conv1x1_bias_kernel[grid_3x3_0](
        x_nhwc, w3x3_0_flat, p_branch3x3_0_bias, branch3x3_0_flat,
        M_1x1, N_3x3_0, K_1x1,
        Ci, 1,
        Ci, 1,
        N_3x3_0, 1,
    )
    branch3x3_0_out = branch3x3_0_flat.view(B, H, W, out_3x3_0_C).permute(0, 3, 1, 2).contiguous()

    # ---- Branch 3x3 step 1: conv2d(branch3x3_0, w3x3_1, pad=1) 3x3 ----
    out_3x3_1_C = p_branch3x3_1_weight.shape[0]  # 208
    M_3x3 = B * H * W
    K_3x3 = out_3x3_0_C * 3 * 3  # 96 * 9 = 864

    branch3x3_out = torch.empty(B, out_3x3_1_C, H, W, device=device, dtype=dtype)
    grid_3x3_1 = (triton.cdiv(M_3x3, 64), triton.cdiv(out_3x3_1_C, 64))
    conv_nxn_bias_kernel[grid_3x3_1](
        branch3x3_0_out, p_branch3x3_1_weight, p_branch3x3_1_bias, branch3x3_out,
        M_3x3, out_3x3_1_C, K_3x3,
        B, out_3x3_0_C, H, W,
        H, W,
        3, 3,
        1, 1,
        1, 1,
    )

    # ---- Branch 5x5 step 0: conv2d(x, w5x5_0) 1x1 ----
    out_5x5_0_C = p_branch5x5_0_weight.shape[0]  # 16
    N_5x5_0 = out_5x5_0_C
    w5x5_0_flat = p_branch5x5_0_weight.view(out_5x5_0_C, Ci)

    branch5x5_0_flat = torch.empty(M_1x1, N_5x5_0, device=device, dtype=dtype)
    grid_5x5_0 = (triton.cdiv(M_1x1, 128), triton.cdiv(N_5x5_0, 128))
    conv1x1_bias_kernel[grid_5x5_0](
        x_nhwc, w5x5_0_flat, p_branch5x5_0_bias, branch5x5_0_flat,
        M_1x1, N_5x5_0, K_1x1,
        Ci, 1,
        Ci, 1,
        N_5x5_0, 1,
    )
    branch5x5_0_out = branch5x5_0_flat.view(B, H, W, out_5x5_0_C).permute(0, 3, 1, 2).contiguous()

    # ---- Branch 5x5 step 1: conv2d(branch5x5_0, w5x5_1, pad=2) 5x5 ----
    out_5x5_1_C = p_branch5x5_1_weight.shape[0]  # 48
    K_5x5 = out_5x5_0_C * 5 * 5  # 16 * 25 = 400

    branch5x5_out = torch.empty(B, out_5x5_1_C, H, W, device=device, dtype=dtype)
    grid_5x5_1 = (triton.cdiv(M_3x3, 64), triton.cdiv(out_5x5_1_C, 64))
    conv_nxn_bias_kernel[grid_5x5_1](
        branch5x5_0_out, p_branch5x5_1_weight, p_branch5x5_1_bias, branch5x5_out,
        M_3x3, out_5x5_1_C, K_5x5,
        B, out_5x5_0_C, H, W,
        H, W,
        5, 5,
        2, 2,
        1, 1,
    )

    # ---- Branch pool: maxpool2d(x, 3x3, stride=1, pad=1) ----
    pool_out = torch.empty(B, Ci, H, W, device=device, dtype=dtype)
    N_pool = B * Ci * H * W
    grid_pool = (triton.cdiv(N_pool, 256),)
    maxpool2d_kernel[grid_pool](
        x, pool_out,
        N_pool,
        B, Ci, H, W,
    )

    # ---- Branch pool conv: conv2d(pool_out, wpool) 1x1 ----
    out_pool_C = p_branch_pool_1_weight.shape[0]  # 64
    N_pool_conv = out_pool_C
    w_pool_flat = p_branch_pool_1_weight.view(out_pool_C, Ci)

    pool_nhwc = pool_out.permute(0, 2, 3, 1).contiguous()
    branch_pool_flat = torch.empty(M_1x1, N_pool_conv, device=device, dtype=dtype)
    grid_pool_conv = (triton.cdiv(M_1x1, 128), triton.cdiv(N_pool_conv, 128))
    conv1x1_bias_kernel[grid_pool_conv](
        pool_nhwc, w_pool_flat, p_branch_pool_1_bias, branch_pool_flat,
        M_1x1, N_pool_conv, K_1x1,
        Ci, 1,
        Ci, 1,
        N_pool_conv, 1,
    )
    branch_pool_out = branch_pool_flat.view(B, H, W, out_pool_C).permute(0, 3, 1, 2).contiguous()

    # ---- Cat: [branch1x1, branch3x3, branch5x5, branch_pool] along dim=1 ----
    # [10,192,224,224], [10,208,224,224], [10,48,224,224], [10,64,224,224]
    # -> [10, 512, 224, 224]
    C_total = out_1x1_C + out_3x3_1_C + out_5x5_1_C + out_pool_C  # 192+208+48+64=512
    HW = H * W
    N_cat = B * C_total * HW

    cat_out = torch.empty(B, C_total, H, W, device=device, dtype=dtype)
    # Make contiguous flat views
    b1x1_flat = branch1x1_out.contiguous()
    b3x3_flat = branch3x3_out.contiguous()
    b5x5_flat = branch5x5_out.contiguous()
    bpool_flat = branch_pool_out.contiguous()

    grid_cat = (triton.cdiv(N_cat, 1024),)
    cat_kernel[grid_cat](
        b1x1_flat, b3x3_flat, b5x5_flat, bpool_flat,
        cat_out,
        N_cat,
        out_1x1_C, out_3x3_1_C, out_5x5_1_C, out_pool_C,
        HW,
    )

    return cat_out


def get_inputs():
    torch.manual_seed(42)
    p_branch1x1_weight = torch.randn(192, 480, 1, 1, dtype=torch.float32, device='cuda')
    p_branch1x1_bias = torch.randn(192, dtype=torch.float32, device='cuda')
    p_branch3x3_0_weight = torch.randn(96, 480, 1, 1, dtype=torch.float32, device='cuda')
    p_branch3x3_0_bias = torch.randn(96, dtype=torch.float32, device='cuda')
    p_branch3x3_1_weight = torch.randn(208, 96, 3, 3, dtype=torch.float32, device='cuda')
    p_branch3x3_1_bias = torch.randn(208, dtype=torch.float32, device='cuda')
    p_branch5x5_0_weight = torch.randn(16, 480, 1, 1, dtype=torch.float32, device='cuda')
    p_branch5x5_0_bias = torch.randn(16, dtype=torch.float32, device='cuda')
    p_branch5x5_1_weight = torch.randn(48, 16, 5, 5, dtype=torch.float32, device='cuda')
    p_branch5x5_1_bias = torch.randn(48, dtype=torch.float32, device='cuda')
    p_branch_pool_1_weight = torch.randn(64, 480, 1, 1, dtype=torch.float32, device='cuda')