import torch
import triton
import triton.language as tl

# ============================================================
# Kernel 1: Conv2d (implicit GEMM) for conv1: (4096,1,32,32) -> (4096,6,28,28)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Decode M index -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_idx in range(BLOCK_K_COUNT):
        k_off = k_idx * BLOCK_K
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        kw = offs_k % KW
        tmp_k = offs_k // KW
        kh = tmp_k % KH
        in_c = tmp_k // KH

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m[:, None] & valid_k[None, :] & valid_ih & valid_iw

        x_ptrs = X_ptr + batch[:, None] * (IN_C * IN_H * IN_W) + \
                 in_c[None, :] * (IN_H * IN_W) + \
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: (OUT_C, IN_C, KH, KW) -> index [n, in_c, kh, kw]
        # offs_k encodes (in_c, kh, kw) as in_c*KH*KW + kh*KW + kw
        w_ptrs = W_ptr + offs_n[None, :] * (IN_C * KH * KW) + offs_k[:, None]
        mask_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store: output layout (BATCH, OUT_C, OUT_H, OUT_W)
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + batch[:, None] * (OUT_C * OUT_H * OUT_W) + \
             offs_n[None, :] * (OUT_H * OUT_W) + \
             out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ============================================================
# Kernel 2: Max Pool 2x2 stride 2
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64},  num_warps=2),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH: tl.constexpr, C: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    N_TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: N_TOTAL = BATCH * C * OUT_H * OUT_W
    out_w = offs % OUT_W
    tmp = offs // OUT_W
    out_h = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    batch = tmp2 // C

    max_val = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for kh in range(KH):
        for kw in range(KW):
            ih = out_h * STRIDE_H + kh
            iw = out_w * STRIDE_W + kw
            valid = mask & (ih < IN_H) & (iw < IN_W)
            x_ptrs = X_ptr + batch * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
            val = tl.load(x_ptrs, mask=valid, other=float('-inf'))
            max_val = tl.maximum(max_val, val)

    y_ptrs = Y_ptr + batch * (C * OUT_H * OUT_W) + c * (OUT_H * OUT_W) + out_h * OUT_W + out_w
    tl.store(y_ptrs, max_val, mask=mask)


# ============================================================
# Kernel 3: Fused Linear + optional ReLU (GEMM tiled)
# M x K @ K x N + bias -> M x N
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 64},  num_warps=4,  num_stages=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    APPLY_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_idx in range(0, K, BLOCK_K):
        offs_k = k_idx + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        x_ptrs = X_ptr + offs_m[:, None] * K + offs_k[None, :]
        w_ptrs = W_ptr + offs_k[:, None] * N + offs_n[None, :]

        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = Y_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_fc1_weight, p_fc1_bias,
    p_fc2_weight, p_fc2_bias,
    p_fc3_weight, p_fc3_bias,
):
    BATCH = x.shape[0]
    device = x.device

    # --- Conv1: (BATCH,1,32,32) -> (BATCH,6,28,28), then ReLU ---
    OUT_C1, IN_C1, KH1, KW1 = 6, 1, 5, 5
    OUT_H1, OUT_W1 = 28, 28
    M1 = BATCH * OUT_H1 * OUT_W1
    conv1_out = torch.empty((BATCH, OUT_C1, OUT_H1, OUT_W1), device=device, dtype=torch.float32)

    grid1 = (triton.cdiv(M1, 256), triton.cdiv(OUT_C1, 16))
    conv2d_relu_kernel[grid1](
        x, p_conv1_weight, p_conv1_bias, conv1_out,
        BATCH, IN_C1, 32, 32,
        OUT_C1, OUT_H1, OUT_W1,
        KH1, KW1, 1, 1, 0, 0,
    )

    # --- MaxPool1: (BATCH,6,28,28) -> (BATCH,6,14,14) ---
    OUT_H_P1, OUT_W_P1 = 14, 14
    pool1_out = torch.empty((BATCH, OUT_C1, OUT_H_P1, OUT_W_P1), device=device, dtype=torch.float32)
    N_POOL1 = BATCH * OUT_C1 * OUT_H_P1 * OUT_W_P1

    grid_p1 = (triton.cdiv(N_POOL1, 256),)
    maxpool2d_kernel[grid_p1](
        conv1_out, pool1_out,
        BATCH, OUT_C1, OUT_H1, OUT_W1, OUT_H_P1, OUT_W_P1,
        2, 2, 2, 2,
        N_POOL1,
    )

    # --- Conv2: (BATCH,6,14,14) -> (BATCH,16,10,10), then ReLU ---
    OUT_C2, IN_C2, KH2, KW2 = 16, 6, 5, 5
    OUT_H2, OUT_W2 = 10, 10
    M2 = BATCH * OUT_H2 * OUT_W2
    conv2_out = torch.empty((BATCH, OUT_C2, OUT_H2, OUT_W2), device=device, dtype=torch.float32)

    grid2 = (triton.cdiv(M2, 256), triton.cdiv(OUT_C2, 16))
    conv2d_relu_kernel[grid2](
        pool1_out, p_conv2_weight, p_conv2_bias, conv2_out,
        BATCH, IN_C2, OUT_H_P1, OUT_W_P1,
        OUT_C2, OUT_H2, OUT_W2,
        KH2, KW2, 1, 1, 0, 0,
    )

    # --- MaxPool2: (BATCH,16,10,10) -> (BATCH,16,5,5) ---
    OUT_H_P2, OUT_W_P2 = 5, 5
    pool2_out = torch.empty((BATCH, OUT_C2, OUT_H_P2, OUT_W_P2), device=device, dtype=torch.float32)
    N_POOL2 = BATCH * OUT_C2 * OUT_H_P2 * OUT_W_P2

    grid_p2 = (triton.cdiv(N_POOL2, 256),)
    maxpool2d_kernel[grid_p2](
        conv2_out, pool2_out,
        BATCH, OUT_C2, OUT_H2, OUT_W2, OUT_H_P2, OUT_W_P2,
        2, 2, 2, 2,
        N_POOL2,
    )

    # --- Flatten: (BATCH, 16, 5, 5) -> (BATCH, 400) ---
    flat = pool2_out.reshape(BATCH, 400)

    # --- FC1: (BATCH,400) -> (BATCH,120), ReLU ---
    # Weight shape: (120, 400), need to transpose for GEMM: X(BATCH,400) @ W^T(400,120)
    w_fc1_t = p_fc1_weight.t().contiguous()  # (400, 120)
    fc1_out = torch.empty((BATCH, 120), device=device, dtype=torch.float32)
    grid_fc1 = (triton.cdiv(BATCH, 128), triton.cdiv(120, 64))
    linear_relu_kernel[grid_fc1](
        flat, w_fc1_t, p_fc1_bias, fc1_out,
        BATCH, 120, 400,
    )

    # --- FC2: (BATCH,120) -> (BATCH,84), ReLU ---
    w_fc2_t = p_fc2_weight.t().contiguous()  # (120, 84)
    fc2_out = torch.empty((BATCH, 84), device=device, dtype=torch.float32)
    grid_fc2 = (triton.cdiv(BATCH, 128), triton.cdiv(84, 64))
    linear_relu_kernel[grid_fc2](
        fc1_out, w_fc2_t, p_fc2_bias, fc2_out,
        BATCH, 84, 120,
    )

    # --- FC3: (BATCH,84) -> (BATCH,20), no ReLU ---
    w_fc3_t = p_fc3_weight.t().contiguous()  # (84, 20)
    fc3_out = torch.empty((BATCH, 20), device=device, dtype=torch.float32)
    grid_fc3 = (triton.cdiv(BATCH, 128), triton.cdiv(20, 64))
    linear_relu_kernel[grid_fc3](
        fc2_out, w_fc3_t, p_fc3_bias, fc3_out,
        BATCH, 20, 84,
    )

    return fc3_out


def get_inputs():
    batch_size = 4096
    num_classes = 20
    p_conv1_weight = torch.randn(6, 1, 5, 5, dtype=torch.float32, device='cuda')
    p_conv1_bias = torch.randn(6, dtype=torch.float32, device='cuda')
    p_conv2_weight = torch.randn(16, 6, 5, 5, dtype=torch.float32, device='cuda')
    p_conv2_bias = torch.randn(16, dtype=torch.float32, device='cuda')
    p_fc1_weight = torch.randn(120, 400, dtype=torch.float32, device='cuda')
    p_fc1_bias = torch.randn(120, dtype=torch.float32, device='cuda')
    p_fc2_weight = torch.randn(84, 120, dtype=torch.float32, device='cuda')
    p_fc2_bias = torch.randn(84, dtype=torch.float32, device='cuda')
    p_fc3_weight = torch.randn(20, 84, dtype=torch.float32, device='cuda')
    p_fc3_bias = torch.randn(20, dtype=torch.float32, device='cuda')
    x = torch.randn(batch_size, 1, 32, 32, dtype=torch.float32, device='cuda')
    return [
        p_conv1_weight, p_conv1_bias,
        p_conv2_weight, p_conv2_bias,
        p_fc1_weight, p_fc1_bias,
        p_fc2_weight, p_fc2_bias,
        p_fc3_weight, p_fc3_bias,
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

        class LeNet5(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(1, 6, 5)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16*5*5, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, num_classes)

            def forward(self, x):
                x = F.relu(self.conv1(x))
                x = F.max_pool2d(x, 2, 2)
                x = F.relu(self.conv2(x))
                x = F.max_pool2d(x, 2, 2)
                x = x.view(-1, 16*5*5)
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x

        (p_conv1_weight, p_conv1_bias,
         p_conv2_weight, p_conv2_bias,
         p_fc1_weight, p_fc1_bias,
         p_fc2_weight, p_fc2_bias,
         p_fc3_weight, p_fc3_bias,
         x) = inputs

        model = LeNet5(20).cuda()
        with torch.no_grad():
            model.conv1.weight.copy_(p_conv1_weight)
            model.conv1.bias.copy_(p_conv1_bias)
            model.conv2.weight.copy_(p_conv2_weight)
            model.conv2.bias.copy_(p_conv2_bias)
            model.fc1.weight.copy_(p_fc1_weight)
            model.fc1.bias.copy_(p_fc1_bias)
            model.fc2.weight.copy_(p_fc2_weight)
            model.fc2.bias.copy_(p_fc2_bias)
            model.fc3.weight.copy_(p_fc3_weight)
            model.fc3.bias.copy_(p_fc3_bias)
            ref = model(x)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")