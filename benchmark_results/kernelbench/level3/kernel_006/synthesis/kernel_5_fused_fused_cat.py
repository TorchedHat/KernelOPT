import torch
import triton
import triton.language as tl
import argparse

# ─── 1x1 conv as GEMM ────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
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
        acc = tl.dot(x, w, acc, allow_tf32=False)

    offs_m2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m2[:, None] < M) & (offs_n2[None, :] < N)

    bias = tl.load(B_ptr + offs_n2, mask=offs_n2 < N, other=0.0)
    acc = acc + bias[None, :]

    y_ptrs = Y_ptr + offs_m2[:, None] * N + offs_n2[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask)


# ─── 3x3 conv (implicit GEMM) ────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32,  'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 32,  'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 16}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv_nxn_bias_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N_batch, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M_TOTAL, TOTAL_K,
    s_xn, s_xc, s_xh, s_xw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch  = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw   = (offs_k // IN_C) % KERNEL_W
        kh   = offs_k // (IN_C * KERNEL_W)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_b  = batch[:, None] < N_batch
        x_mask = valid_m & valid_k & valid_ih & valid_iw & valid_b

        x_ptrs = X_ptr + batch[:, None] * s_xn + in_c[None, :] * s_xc + ih * s_xh + iw * s_xw
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> index [oc, in_c, kh, kw]
        # W_ptr layout: [OUT_C, IN_C, KH, KW] contiguous
        # offs_k encodes (kh, kw, in_c) as kh*KW*IN_C + kw*IN_C + in_c
        w_ptrs = W_ptr + offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) + \
                 kh[:, None] * (IN_C * KERNEL_W) + kw[:, None] * IN_C + in_c[:, None]
        w_mask = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    offs_m2 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n2 = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m2[:, None] < M_TOTAL) & (offs_n2[None, :] < OUT_C)

    bias = tl.load(B_ptr + offs_n2, mask=offs_n2 < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Output: [N, OUT_C, OUT_H, OUT_W] -> store at [batch, oc, out_h, out_w]
    out_w2 = offs_m2 % OUT_W
    out_h2 = (offs_m2 // OUT_W) % OUT_H
    batch2  = offs_m2 // (OUT_W * OUT_H)

    y_ptrs = Y_ptr + batch2[:, None] * (OUT_C * OUT_H * OUT_W) + \
             offs_n2[None, :] * (OUT_H * OUT_W) + \
             out_h2[:, None] * OUT_W + out_w2[:, None]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask)


# ─── MaxPool2d 3x3 stride=1 pad=1 ────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def maxpool3x3_s1_p1_kernel(
    X_ptr, Y_ptr,
    N_batch, C, H, W,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    # Decode flat index -> (n, c, h, w)
    w_idx = offs % W
    h_idx = (offs // W) % H
    c_idx = (offs // (W * H)) % C
    n_idx = offs // (W * H * C)

    # 3x3 max pool with stride=1, pad=1
    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for dh in range(-1, 2):
        for dw in range(-1, 2):
            ih = h_idx + dh
            iw = w_idx + dw
            valid = (ih >= 0) & (ih < H) & (iw >= 0) & (iw < W) & mask
            x_ptrs = X_ptr + n_idx * (C * H * W) + c_idx * (H * W) + ih * W + iw
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    tl.store(Y_ptr + offs, max_val, mask=mask)


# ─── Cat kernel: write 4 tensors into output along channel dim ───────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['N_total'],
)
@triton.jit
def cat4_kernel(
    A_ptr, B_ptr, C_ptr, D_ptr,
    Y_ptr,
    N_batch, CA, CB, CC, CD,
    H, W,
    N_total,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_total

    C_total = CA + CB + CC + CD
    # Decode: (n, c_out, h, w)
    w_idx = offs % W
    h_idx = (offs // W) % H
    c_idx = (offs // (W * H)) % C_total
    n_idx = offs // (W * H * C_total)

    # Determine which branch
    in_A = c_idx < CA
    in_B = (c_idx >= CA) & (c_idx < CA + CB)
    in_C = (c_idx >= CA + CB) & (c_idx < CA + CB + CC)
    in_D = c_idx >= CA + CB + CC

    c_A = c_idx
    c_B = c_idx - CA
    c_C = c_idx - CA - CB
    c_D = c_idx - CA - CB - CC

    flat_hw = h_idx * W + w_idx

    ptr_A = A_ptr + n_idx * (CA * H * W) + c_A * (H * W) + flat_hw
    ptr_B = B_ptr + n_idx * (CB * H * W) + c_B * (H * W) + flat_hw
    ptr_C = C_ptr + n_idx * (CC * H * W) + c_C * (H * W) + flat_hw
    ptr_D = D_ptr + n_idx * (CD * H * W) + c_D * (H * W) + flat_hw

    val_A = tl.load(ptr_A, mask=mask & in_A, other=0.0)
    val_B = tl.load(ptr_B, mask=mask & in_B, other=0.0)
    val_C = tl.load(ptr_C, mask=mask & in_C, other=0.0)
    val_D = tl.load(ptr_D, mask=mask & in_D, other=0.0)

    val = tl.where(in_A, val_A, tl.where(in_B, val_B, tl.where(in_C, val_C, val_D)))
    tl.store(Y_ptr + offs, val, mask=mask)


def run_conv1x1(x_nhwc, w, b, out_c):
    """x_nhwc: [M, K], w: [out_c, K], b: [out_c] -> y: [M, out_c]"""
    M, K = x_nhwc.shape
    N = out_c
    y = torch.empty((M, N), device=x_nhwc.device, dtype=torch.float32)
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']), triton.cdiv(N, meta['BLOCK_N']))
    conv1x1_bias_kernel[grid](
        x_nhwc, w, b, y,
        M, N, K,
        K, 1,   # stride_wn, stride_wk (w is [N, K], so stride_wn=K, stride_wk=1)
    )
    return y


def run_conv_nxn(x, w, b, pad_h, pad_w, stride_h=1, stride_w=1):
    """x: [N, C, H, W], w: [OC, IC, KH, KW], b: [OC]"""
    N_batch, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = w.shape
    OUT_H = (IN_H + 2 * pad_h - KERNEL_H) // stride_h + 1
    OUT_W = (IN_W + 2 * pad_w - KERNEL_W) // stride_w + 1
    M_TOTAL = N_batch * OUT_H * OUT_W
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    y = torch.empty((N_batch, OUT_C, OUT_H, OUT_W), device=x.device, dtype=torch.float32)

    s_xn = IN_C * IN_H * IN_W
    s_xc = IN_H * IN_W
    s_xh = IN_W
    s_xw = 1

    grid = lambda meta: (triton.cdiv(M_TOTAL, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))
    conv_nxn_bias_kernel[grid](
        x, w, b, y,
        N_batch, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        stride_h, stride_w,
        pad_h, pad_w,
        M_TOTAL, TOTAL_K,
        s_xn, s_xc, s_xh, s_xw,
    )
    return y


def kernel_function(
    p_branch1x1_weight, p_branch1x1_bias,
    p_branch3x3_0_weight, p_branch3x3_0_bias,
    p_branch3x3_1_weight, p_branch3x3_1_bias,
    p_branch5x5_0_weight, p_branch5x5_0_bias,
    p_branch5x5_1_weight, p_branch5x5_1_bias,
    p_branch_pool_1_weight, p_branch_pool_1_bias,
    x,
):
    N_batch, IN_C, H, W = x.shape  # [10, 480, 224, 224]

    # ── Branch 1x1 ──────────────────────────────────────────────────────────
    # x: [10, 480, 224, 224] -> reshape to [M, K] = [10*224*224, 480]
    M = N_batch * H * W
    x_flat = x.permute(0, 2, 3, 1).reshape(M, IN_C)  # [M, 480]

    # w: [192, 480] -> use as [N, K]
    w1 = p_branch1x1_weight.reshape(192, 480)
    b1 = p_branch1x1_bias
    branch1x1_flat = run_conv1x1(x_flat, w1, b1, 192)  # [M, 192]
    branch1x1 = branch1x1_flat.reshape(N_batch, H, W, 192).permute(0, 3, 1, 2).contiguous()

    # ── Branch 3x3: 1x1 then 3x3 ────────────────────────────────────────────
    w3_0 = p_branch3x3_0_weight.reshape(96, 480)
    b3_0 = p_branch3x3_0_bias
    branch3x3_0_flat = run_conv1x1(x_flat, w3_0, b3_0, 96)  # [M, 96]
    branch3x3_0 = branch3x3_0_flat.reshape(N_batch, H, W, 96).permute(0, 3, 1, 2).contiguous()

    # 3x3 conv with pad=1
    branch3x3 = run_conv_nxn(branch3x3_0, p_branch3x3_1_weight, p_branch3x3_1_bias, pad_h=1, pad_w=1)

    # ── Branch 5x5: 1x1 then 5x5 ────────────────────────────────────────────
    w5_0 = p_branch5x5_0_weight.reshape(16, 480)
    b5_0 = p_branch5x5_0_bias
    branch5x5_0_flat = run_conv1x1(x_flat, w5_0, b5_0, 16)  # [M, 16]
    branch5x5_0 = branch5x5_0_flat.reshape(N_batch, H, W, 16).permute(0, 3, 1, 2).contiguous()

    # 5x5 conv with pad=2
    branch5x5 = run_conv_nxn(branch5x5_0, p_branch5x5_1_weight, p_branch5x5_1_bias, pad_h=2, pad_w=2)

    # ── MaxPool branch ───────────────────────────────────────────────────────
    pool_out = torch.empty_like(x)
    N_total_pool = N_batch * IN_C * H * W
    grid_pool = lambda meta: (triton.cdiv(N_total_pool, meta['BLOCK_SIZE']),)
    maxpool3x3_s1_p1_kernel[grid_pool](
        x, pool_out,
        N_batch, IN_C, H, W,
        N_total_pool,
    )

    # 1x1 conv on pool output
    pool_flat = pool_out.permute(0, 2, 3, 1).reshape(M, IN_C)
    w_pool = p_branch_pool_1_weight.reshape(64, 480)
    b_pool = p_branch_pool_1_bias
    branch_pool_flat = run_conv1x1(pool_flat, w_pool, b_pool, 64)
    branch_pool = branch_pool_flat.reshape(N_batch, H, W, 64).permute(0, 3, 1, 2).contiguous()

    # ── Cat ──────────────────────────────────────────────────────────────────
    # Output: [10, 512, 224, 224]
    C_total = 192 + 208 + 48 + 64  # = 512
    out = torch.empty((N_batch, C_total, H, W), device=x.device, dtype=torch.float32)
    N_total_cat = N_batch * C_total * H * W
    grid_cat = lambda meta: (triton.cdiv(N_total_cat, meta['BLOCK_SIZE']),)
    cat4_kernel[grid_cat](
        branch1x1, branch3x3, branch5x5, branch_pool,
        out,
        N_batch, 192, 208, 48, 64,
        H, W,
        N_total_cat,
    )
    return out


def get_inputs():
    torch.manual_seed(42)
    p_branch1x1_weight = torch.randn(192, 480, 1, 1)
    p_branch1x1_bias = torch.randn(192)
    p_branch3x3_0_weight = torch.randn(96, 480, 1, 1)
    p_branch3x3_0_bias = torch.randn(96)
    p_branch3x3_1_weight = torch.randn(208, 96, 3, 3)
    p_branch3x3_1_bias = torch.randn(208)
    p_branch5x5_0_weight = torch.randn(16, 480, 1, 1)
    p_branch5x5_0_bias = torch.randn(16)
    p_branch5x5_1_weight = torch.randn(48, 16, 5, 5)
    p_branch5x5_1_bias = torch.randn(48)
    p_branch_pool_1_weight = torch.randn(64, 480, 1, 1)
    p_branch_pool_1_bias = torch.randn(64)
    x = torch.randn(10, 480, 224, 224)
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]

    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn
        import torch.nn.functional as F

        # Reference implementation
        (p_branch1x1_weight, p_branch1x1_bias,
         p_branch3x3_0_weight, p_branch3x3_0_bias,
         p_branch3x3_1_weight, p_branch3x3_1_bias,
         p_branch5x5_0_weight, p_branch5x5_0_bias,
         p_branch5x5_1_weight, p_branch5x5_1_bias,
         p_branch_pool_1_weight, p_branch_pool_1_bias,
         x) = inputs_cuda

        branch1x1_ref = F.conv2d(x, p_branch1x1_weight, p_branch1x1_bias)
        branch3x3_ref = F.conv2d(
            F.conv2d(x, p_branch3x3_0_weight, p_branch3x3_0_bias),
            p_branch3x3_1_weight, p_branch3x3_1_bias, padding=1
        )
        branch5x5_ref = F.conv2d(
            F.conv2d(x, p_branch5x5_0_weight, p_branch5x5_0_bias),
            p_branch5x5_1_weight, p_branch5x5_1_bias, padding=2
        )
        pool_ref = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool_ref = F.conv2d(pool_ref, p_branch_pool_1_weight, p_branch_pool_1_bias)
        ref = torch.cat([branch1x1_ref, branch3x3_ref, branch5x5_ref, branch_pool_ref], dim=1)

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")