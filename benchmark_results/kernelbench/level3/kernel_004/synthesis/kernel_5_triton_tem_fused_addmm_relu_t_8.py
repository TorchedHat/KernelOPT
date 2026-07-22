import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Implicit-GEMM Conv2d  (generic, used for both conv1 and conv2)
# Input layout: NCHW  Weight layout: (Co, Ci, KH, KW)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W',
         'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W,  N = OUT_C,  K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    M_TOTAL = BATCH * OUT_H * OUT_W
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw

        # X in NCHW: stride = (IN_C*IN_H*IN_W, IN_H*IN_W, IN_W, 1)
        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # W in (Co, Ci, KH, KW): stride = (Ci*KH*KW, KH*KW, KW, 1)
        mask_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store: output in NCHW -> (BATCH, OUT_C, OUT_H, OUT_W)
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


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
    BATCH: tl.constexpr, C: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < TOTAL

    # Decode flat index -> (n, c, oh, ow)
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    # Gather 4 input values
    ih0 = oh * 2
    iw0 = ow * 2

    base = n * (C * IN_H * IN_W) + c * (IN_H * IN_W)

    p00 = base + ih0 * IN_W + iw0
    p01 = base + ih0 * IN_W + (iw0 + 1)
    p10 = base + (ih0 + 1) * IN_W + iw0
    p11 = base + (ih0 + 1) * IN_W + (iw0 + 1)

    v00 = tl.load(X_ptr + p00, mask=mask, other=-float('inf'))
    v01 = tl.load(X_ptr + p01, mask=mask, other=-float('inf'))
    v10 = tl.load(X_ptr + p10, mask=mask, other=-float('inf'))
    v11 = tl.load(X_ptr + p11, mask=mask, other=-float('inf'))

    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    tl.store(Y_ptr + offs, result, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Linear (GEMM) + optional bias + optional ReLU
# A: (M, K),  W: (N, K),  B: (N,)  ->  Y: (M, N)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=8, num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=4, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 128}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    A_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        a_ptrs = A_ptr + offs_m[:, None] * K + offs_k[None, :]
        a_tile = tl.load(a_ptrs, mask=mask_a, other=0.0)

        mask_w = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        w_ptrs = W_ptr + offs_n[:, None] * K + offs_k[None, :]
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(a_tile, tl.trans(w_tile), acc, allow_tf32=False)

    # Bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Wrapper
# ─────────────────────────────────────────────────────────────────────────────
def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_fc1_weight, p_fc1_bias,
    p_fc2_weight, p_fc2_bias,
    p_fc3_weight, p_fc3_bias,
    x,
):
    BATCH = x.shape[0]  # 4096
    device = x.device

    # ── Conv1: (4096,1,32,32) -> (4096,6,28,28), ReLU ──────────────────────
    OUT_C1, IN_C1, KH1, KW1 = 6, 1, 5, 5
    OUT_H1, OUT_W1 = 28, 28
    M1 = BATCH * OUT_H1 * OUT_W1
    conv1_out = torch.empty((BATCH, OUT_C1, OUT_H1, OUT_W1), device=device, dtype=torch.float32)

    grid1 = (triton.cdiv(M1, 128), triton.cdiv(OUT_C1, 32))
    conv2d_relu_kernel[grid1](
        x, p_conv1_weight, p_conv1_bias, conv1_out,
        BATCH, IN_C1, 32, 32,
        OUT_C1, OUT_H1, OUT_W1,
        KH1, KW1, 1, 1, 0, 0,
        BATCH=BATCH, IN_C=IN_C1, IN_H=32, IN_W=32,
        OUT_C=OUT_C1, OUT_H=OUT_H1, OUT_W=OUT_W1,
        KERNEL_H=KH1, KERNEL_W=KW1,
    )

    # ── MaxPool1: (4096,6,28,28) -> (4096,6,14,14) ─────────────────────────
    OUT_H_P1, OUT_W_P1 = 14, 14
    pool1_out = torch.empty((BATCH, OUT_C1, OUT_H_P1, OUT_W_P1), device=device, dtype=torch.float32)
    TOTAL_P1 = BATCH * OUT_C1 * OUT_H_P1 * OUT_W_P1
    grid_p1 = (triton.cdiv(TOTAL_P1, 512),)
    maxpool2x2_kernel[grid_p1](
        conv1_out, pool1_out,
        BATCH, OUT_C1, OUT_H1, OUT_W1, OUT_H_P1, OUT_W_P1,
        TOTAL=TOTAL_P1,
    )

    # ── Conv2: (4096,6,14,14) -> (4096,16,10,10), ReLU ─────────────────────
    OUT_C2, IN_C2, KH2, KW2 = 16, 6, 5, 5
    OUT_H2, OUT_W2 = 10, 10
    M2 = BATCH * OUT_H2 * OUT_W2
    conv2_out = torch.empty((BATCH, OUT_C2, OUT_H2, OUT_W2), device=device, dtype=torch.float32)

    grid2 = (triton.cdiv(M2, 128), triton.cdiv(OUT_C2, 32))
    conv2d_relu_kernel[grid2](
        pool1_out, p_conv2_weight, p_conv2_bias, conv2_out,
        BATCH, IN_C2, OUT_H_P1, OUT_W_P1,
        OUT_C2, OUT_H2, OUT_W2,
        KH2, KW2, 1, 1, 0, 0,
        BATCH=BATCH, IN_C=IN_C2, IN_H=OUT_H_P1, IN_W=OUT_W_P1,
        OUT_C=OUT_C2, OUT_H=OUT_H2, OUT_W=OUT_W2,
        KERNEL_H=KH2, KERNEL_W=KW2,
    )

    # ── MaxPool2: (4096,16,10,10) -> (4096,16,5,5) ─────────────────────────
    OUT_H_P2, OUT_W_P2 = 5, 5
    pool2_out = torch.empty((BATCH, OUT_C2, OUT_H_P2, OUT_W_P2), device=device, dtype=torch.float32)
    TOTAL_P2 = BATCH * OUT_C2 * OUT_H_P2 * OUT_W_P2
    grid_p2 = (triton.cdiv(TOTAL_P2, 512),)
    maxpool2x2_kernel[grid_p2](
        conv2_out, pool2_out,
        BATCH, OUT_C2, OUT_H2, OUT_W2, OUT_H_P2, OUT_W_P2,
        TOTAL=TOTAL_P2,
    )

    # ── Flatten: (4096,16,5,5) -> (4096,400) ───────────────────────────────
    flat = pool2_out.reshape(BATCH, 400)

    # ── FC1: (4096,400) -> (4096,120), ReLU ────────────────────────────────
    M_fc = BATCH
    fc1_out = torch.empty((M_fc, 120), device=device, dtype=torch.float32)
    grid_fc1 = (triton.cdiv(M_fc, 128), triton.cdiv(120, 64))
    linear_relu_kernel[grid_fc1](
        flat, p_fc1_weight, p_fc1_bias, fc1_out,
        M_fc, 120, 400,
        M=M_fc, N=120, K=400,
    )

    # ── FC2: (4096,120) -> (4096,84), ReLU ─────────────────────────────────
    fc2_out = torch.empty((M_fc, 84), device=device, dtype=torch.float32)
    grid_fc2 = (triton.cdiv(M_fc, 128), triton.cdiv(84, 64))
    linear_relu_kernel[grid_fc2](
        fc1_out, p_fc2_weight, p_fc2_bias, fc2_out,
        M_fc, 84, 120,
        M=M_fc, N=84, K=120,
    )

    # ── FC3: (4096,84) -> (4096,20) ────────────────────────────────────────
    fc3_out = torch.empty((M_fc, 20), device=device, dtype=torch.float32)
    grid_fc3 = (triton.cdiv(M_fc, 128), triton.cdiv(20, 64))
    linear_relu_kernel[grid_fc3](
        fc2_out, p_fc3_weight, p_fc3_bias, fc3_out,
        M_fc, 20, 84,
        M=M_fc, N=20, K=84,
    )

    return fc3_out


def get_inputs():
    torch.manual_seed(42)
    p_conv1_weight = torch.randn(6, 1, 5, 5, device='cuda')
    p_conv1_bias   = torch.randn(6, device='cuda')
    p_conv2_weight = torch.randn(16, 6, 5, 5, device='cuda')
    p_conv2_bias   = torch.randn(16, device='cuda')
    p_fc1_weight   = torch.randn(120, 400, device='cuda')
    p_fc1_bias     = torch.randn(120, device='cuda')
    p_fc2_weight   = torch.randn(84, 120, device='cuda')
    p_fc2_bias     = torch.randn(84, device='cuda')
    p_fc3_weight   = torch.randn(20, 84, device='cuda')
    p_fc3_bias     = torch.randn(20, device='cuda')
    x              = torch.randn(4096, 1, 32, 32, device='cuda')
    return [p_conv1_weight, p_conv1_bias,
            p_conv2_weight, p_conv2_bias,
            p_fc1_weight, p_fc1_bias,
            p_fc2_weight, p_fc2_bias,
            p_fc3_weight, p_fc3_bias,


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
        x_ref = x
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