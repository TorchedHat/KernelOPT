import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Implicit-GEMM Conv2d + bias + ReLU
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'IN_C', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    stride_xn, stride_xc, stride_xh, stride_xw,
    stride_wco, stride_wci, stride_wkh, stride_wkw,
    M_TOTAL,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    IN_C_CONST: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp   = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    mask_m = offs_m < M_TOTAL
    mask_n = offs_n < OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C_CONST * KERNEL_H * KERNEL_W

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C_CONST
        kw   = (offs_k // IN_C_CONST) % KERNEL_W
        kh   = offs_k // (IN_C_CONST * KERNEL_W)

        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (
            mask_m[:, None] &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid, other=0.0)

        w_ptrs = (W_ptr
        w_mask = (offs_k[:, None] < TOTAL_K) & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store: output layout (BATCH, OUT_C, OUT_H, OUT_W)
    out_idx = (batch[:, None] * (OUT_C * OUT_H * OUT_W)
    store_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(Y_ptr + out_idx, acc.to(tl.float32), mask=store_mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: MaxPool2d 2x2 stride 2
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['TOTAL'],
)
@triton.jit
def maxpool2x2_kernel(
    X_ptr, Y_ptr,
    BATCH, C, IN_H, IN_W, OUT_H, OUT_W,
    TOTAL,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL

    # Decode: (batch, c, out_h, out_w)
    out_w = offs % OUT_W
    tmp   = offs // OUT_W
    out_h = tmp % OUT_H
    tmp2  = tmp // OUT_H
    c     = tmp2 % C
    batch = tmp2 // C

    base = batch * (C * IN_H * IN_W) + c * (IN_H * IN_W)
    ih0 = out_h * 2
    iw0 = out_w * 2

    v00 = tl.load(X_ptr + base + ih0 * IN_W + iw0,         mask=mask, other=-1e38)
    v01 = tl.load(X_ptr + base + ih0 * IN_W + iw0 + 1,     mask=mask, other=-1e38)
    v10 = tl.load(X_ptr + base + (ih0+1) * IN_W + iw0,     mask=mask, other=-1e38)
    v11 = tl.load(X_ptr + base + (ih0+1) * IN_W + iw0 + 1, mask=mask, other=-1e38)

    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    tl.store(Y_ptr + offs, result, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Linear (GEMM) + bias + optional ReLU
# Input: X (M, K), W (N, K), B (N,) -> Y (M, N)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 64}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + offs_k[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=mask_n[None, :] & mask_k[:, None], other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


# ─────────────────────────────────────────────────────────────────────────────
# Main wrapper
# ─────────────────────────────────────────────────────────────────────────────
def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_fc1_weight, p_fc1_bias,
    p_fc2_weight, p_fc2_bias,
    p_fc3_weight, p_fc3_bias,
    x,
):
    BATCH = x.shape[0]
    device = x.device
    dtype  = x.dtype

    # ── Conv1: (BATCH,1,32,32) -> (BATCH,6,28,28) + ReLU ──────────────────
    IN_C1, IN_H1, IN_W1 = 1, 32, 32
    OUT_C1, OUT_H1, OUT_W1 = 6, 28, 28
    KH1, KW1 = 5, 5
    M1 = BATCH * OUT_H1 * OUT_W1

    conv1_out = torch.empty(BATCH, OUT_C1, OUT_H1, OUT_W1, device=device, dtype=dtype)

    grid1 = (triton.cdiv(M1, 128), triton.cdiv(OUT_C1, 32))
    conv2d_relu_kernel[grid1](
        x, p_conv1_weight, p_conv1_bias, conv1_out,
        BATCH, IN_C1, IN_H1, IN_W1,
        OUT_C1, OUT_H1, OUT_W1,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        p_conv1_weight.stride(0), p_conv1_weight.stride(1),
        p_conv1_weight.stride(2), p_conv1_weight.stride(3),
        M1,
        KH1, KW1, 1, 1, 0, 0,
        IN_C1,
        True,
    )

    # ── MaxPool1: (BATCH,6,28,28) -> (BATCH,6,14,14) ───────────────────────
    OUT_H1p, OUT_W1p = 14, 14
    pool1_out = torch.empty(BATCH, OUT_C1, OUT_H1p, OUT_W1p, device=device, dtype=dtype)
    TOTAL1p = BATCH * OUT_C1 * OUT_H1p * OUT_W1p

    grid_p1 = (triton.cdiv(TOTAL1p, 512),)
    maxpool2x2_kernel[grid_p1](
        conv1_out, pool1_out,
        BATCH, OUT_C1, OUT_H1, OUT_W1, OUT_H1p, OUT_W1p,
        TOTAL1p,
    )

    # ── Conv2: (BATCH,6,14,14) -> (BATCH,16,10,10) + ReLU ─────────────────
    IN_C2, IN_H2, IN_W2 = 6, 14, 14
    OUT_C2, OUT_H2, OUT_W2 = 16, 10, 10
    KH2, KW2 = 5, 5
    M2 = BATCH * OUT_H2 * OUT_W2

    conv2_out = torch.empty(BATCH, OUT_C2, OUT_H2, OUT_W2, device=device, dtype=dtype)

    grid2 = (triton.cdiv(M2, 128), triton.cdiv(OUT_C2, 32))
    conv2d_relu_kernel[grid2](
        pool1_out, p_conv2_weight, p_conv2_bias, conv2_out,
        BATCH, IN_C2, IN_H2, IN_W2,
        OUT_C2, OUT_H2, OUT_W2,
        pool1_out.stride(0), pool1_out.stride(1), pool1_out.stride(2), pool1_out.stride(3),
        p_conv2_weight.stride(0), p_conv2_weight.stride(1),
        p_conv2_weight.stride(2), p_conv2_weight.stride(3),
        M2,
        KH2, KW2, 1, 1, 0, 0,
        IN_C2,
        True,
    )

    # ── MaxPool2: (BATCH,16,10,10) -> (BATCH,16,5,5) ───────────────────────
    OUT_H2p, OUT_W2p = 5, 5
    pool2_out = torch.empty(BATCH, OUT_C2, OUT_H2p, OUT_W2p, device=device, dtype=dtype)
    TOTAL2p = BATCH * OUT_C2 * OUT_H2p * OUT_W2p

    grid_p2 = (triton.cdiv(TOTAL2p, 512),)
    maxpool2x2_kernel[grid_p2](
        conv2_out, pool2_out,
        BATCH, OUT_C2, OUT_H2, OUT_W2, OUT_H2p, OUT_W2p,
        TOTAL2p,
    )

    # ── Flatten: (BATCH,16,5,5) -> (BATCH,400) ─────────────────────────────
    flat = pool2_out.reshape(BATCH, 400)

    # ── FC1: (BATCH,400) -> (BATCH,120) + ReLU ─────────────────────────────
    M_fc1, N_fc1, K_fc1 = BATCH, 120, 400
    fc1_out = torch.empty(M_fc1, N_fc1, device=device, dtype=dtype)
    grid_fc1 = (triton.cdiv(M_fc1, 64), triton.cdiv(N_fc1, 64))
    linear_relu_kernel[grid_fc1](
        flat, p_fc1_weight, p_fc1_bias, fc1_out,
        M_fc1, N_fc1, K_fc1,
        flat.stride(0), flat.stride(1),
        p_fc1_weight.stride(0), p_fc1_weight.stride(1),
        True,
    )

    # ── FC2: (BATCH,120) -> (BATCH,84) + ReLU ──────────────────────────────
    M_fc2, N_fc2, K_fc2 = BATCH, 84, 120
    fc2_out = torch.empty(M_fc2, N_fc2, device=device, dtype=dtype)
    grid_fc2 = (triton.cdiv(M_fc2, 64), triton.cdiv(N_fc2, 64))
    linear_relu_kernel[grid_fc2](
        fc1_out, p_fc2_weight, p_fc2_bias, fc2_out,
        M_fc2, N_fc2, K_fc2,
        fc1_out.stride(0), fc1_out.stride(1),
        p_fc2_weight.stride(0), p_fc2_weight.stride(1),
        True,
    )

    # ── FC3: (BATCH,84) -> (BATCH,20) ──────────────────────────────────────
    M_fc3, N_fc3, K_fc3 = BATCH, 20, 84
    fc3_out = torch.empty(M_fc3, N_fc3, device=device, dtype=dtype)
    grid_fc3 = (triton.cdiv(M_fc3, 64), triton.cdiv(N_fc3, 64))
    linear_relu_kernel[grid_fc3](
        fc2_out, p_fc3_weight, p_fc3_bias, fc3_out,
        M_fc3, N_fc3, K_fc3,
        fc2_out.stride(0), fc2_out.stride(1),
        p_fc3_weight.stride(0), p_fc3_weight.stride(1),
        False,
    )

    return fc3_out


def get_inputs():
    batch_size = 4096
    num_classes = 20
    p_conv1_weight = torch.randn(6, 1, 5, 5, device='cuda', dtype=torch.float32)
    p_conv1_bias   = torch.randn(6, device='cuda', dtype=torch.float32)
    p_conv2_weight = torch.randn(16, 6, 5, 5, device='cuda', dtype=torch.float32)
    p_conv2_bias   = torch.randn(16, device='cuda', dtype=torch.float32)
    p_fc1_weight   = torch.randn(120, 400, device='cuda', dtype=torch.float32)
    p_fc1_bias     = torch.randn(120, device='cuda', dtype=torch.float32)
    p_fc2_weight   = torch.randn(84, 120, device='cuda', dtype=torch.float32)
    p_fc2_bias     = torch.randn(84, device='cuda', dtype=torch.float32)
    p_fc3_weight   = torch.randn(20, 84, device='cuda', dtype=torch.float32)
    p_fc3_bias     = torch.randn(20, device='cuda', dtype=torch.float32)
    x = torch.randn(batch_size, 1, 32, 32, device='cuda', dtype=torch.float32)
    return [
        p_conv1_weight, p_conv1_bias,
        p_conv2_weight, p_conv2_bias,
        p_fc1_weight, p_fc1_bias,
        p_fc2_weight, p_fc2_bias,
        p_fc3_weight, p_fc3_bias,
        x,
    ]


if __name__ == "__main__":
    import argparse
    import torch.nn as nn
    import torch.nn.functional as F

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        (p_conv1_weight, p_conv1_bias,
         p_conv2_weight, p_conv2_bias,
         p_fc1_weight, p_fc1_bias,
         p_fc2_weight, p_fc2_bias,
         p_fc3_weight, p_fc3_bias,
         x) = inputs

        # Reference using PyTorch eager
        x_ref = x.clone()
        x_ref = F.relu(F.conv2d(x_ref, p_conv1_weight, p_conv1_bias))
        x_ref = F.max_pool2d(x_ref, kernel_size=2, stride=2)
        x_ref = F.relu(F.conv2d(x_ref, p_conv2_weight, p_conv2_bias))
        x_ref = F.max_pool2d(x_ref, kernel_size=2, stride=2)
        x_ref = x_ref.view(-1, 400)
        x_ref = F.relu(F.linear(x_ref, p_fc1_weight, p_fc1_bias))
        x_ref = F.relu(F.linear(x_ref, p_fc2_weight, p_fc2_bias))
        x_ref = F.linear(x_ref, p_fc3_weight, p_fc3_bias)

        assert torch.allclose(output, x_ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - x_ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")