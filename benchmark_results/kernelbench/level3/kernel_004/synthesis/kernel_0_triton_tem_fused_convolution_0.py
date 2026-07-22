import torch
import triton
import triton.language as tl

# ============================================================
# Kernel 1: Conv2d (implicit GEMM) for conv1
# Input: (N, 1, 32, 32), Weight: (6, 1, 5, 5)
# Output: (N, 6, 28, 28)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1024, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=1),
    ],
    key=['BATCH', 'OUT_H', 'OUT_W', 'IN_C', 'OUT_C', 'KERNEL_H', 'KERNEL_W'],
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
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Decode M index -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_k = offs_k < TOTAL_K
        valid_m = offs_m < M_TOTAL
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_batch = batch_idx < BATCH

        mask_x = valid_m[:, None] & valid_batch[:, None] & valid_ih & valid_iw & valid_k[None, :]

        x_ptrs = X_ptr + (
            batch_idx[:, None] * (IN_C * IN_H * IN_W) +
            in_c[None, :] * (IN_H * IN_W) +
        )
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: (OUT_C, IN_C, KERNEL_H, KERNEL_W) -> index: oc * IN_C*KH*KW + in_c * KH*KW + kh * KW + kw
        # offs_k encodes (in_c, kh, kw) as in_c * KH * KW + kh * KW + kw
        # But we decoded as: kw = offs_k % KW, kh = (offs_k // KW) % KH, in_c = offs_k // (KW * KH)
        # So offs_k = in_c * KH * KW + kh * KW + kw  (same encoding)
        mask_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = W_ptr + (
            offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) +
            offs_k[:, None]
        )
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

    # Add bias
    bias_mask = offs_n < OUT_C
    bias = tl.load(B_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    # Apply ReLU if needed
    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store output: Y shape (BATCH, OUT_C, OUT_H, OUT_W)
    # Y[batch, oc, oh, ow] = Y[batch * OUT_C * OUT_H * OUT_W + oc * OUT_H * OUT_W + oh * OUT_W + ow]
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = Y_ptr + (
        batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
        offs_n[None, :] * (OUT_H * OUT_W) +
        out_h_idx[:, None] * OUT_W +
        out_w_idx[:, None]
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ============================================================
# Kernel 2: Max Pool 2x2 stride 2
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def maxpool2d_kernel(
    X_ptr, Y_ptr,
    BATCH: tl.constexpr, C: tl.constexpr,
    IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    N_TOTAL: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: offs -> (batch, c, oh, ow)
    ow = offs % OUT_W
    tmp = offs // OUT_W
    oh = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    batch = tmp2 // C

    # Gather 2x2 window
    ih0 = oh * 2
    iw0 = ow * 2

    def load_val(dh, dw):
        ih = ih0 + dh
        iw = iw0 + dw
        valid = mask & (batch < BATCH) & (c < C) & (ih < IN_H) & (iw < IN_W)
        ptr = X_ptr + batch * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw
        return tl.load(ptr, mask=valid, other=float('-inf'))

    v00 = load_val(0, 0)
    v01 = load_val(0, 1)
    v10 = load_val(1, 0)
    v11 = load_val(1, 1)

    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))

    out_ptr = Y_ptr + batch * (C * OUT_H * OUT_W) + c * (OUT_H * OUT_W) + oh * OUT_W + ow
    tl.store(out_ptr, result, mask=mask)


# ============================================================
# Kernel 3: Fused Linear + ReLU (GEMM)
# A: (M, K), B: (N, K) -> C: (M, N)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    A_ptr, B_ptr, Bias_ptr, C_ptr,
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

        mask_b = (offs_n[:, None] < N) & (offs_k[None, :] < K)
        b_ptrs = B_ptr + offs_n[:, None] * K + offs_k[None, :]
        b_tile = tl.load(b_ptrs, mask=mask_b, other=0.0)

        acc = tl.dot(a_tile, b_tile.T, acc=acc, allow_tf32=False)

    # Add bias
    bias_mask = offs_n < N
    bias = tl.load(Bias_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    if APPLY_RELU:
        acc = tl.maximum(acc, 0.0)

    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    c_ptrs = C_ptr + offs_m[:, None] * N + offs_n[None, :]
    tl.store(c_ptrs, acc.to(tl.float32), mask=mask_c)


def conv2d_relu_launch(x, weight, bias, apply_relu, stride=1, pad=0):
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    OUT_H = (IN_H + 2 * pad - KH) // stride + 1
    OUT_W = (IN_W + 2 * pad - KW) // stride + 1

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = N * OUT_H * OUT_W
    BLOCK_N_FIXED = 16

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, BLOCK_N_FIXED),
    )

    conv2d_relu_kernel[grid](
        x, weight, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride, stride,
        pad, pad,
        apply_relu,
        BLOCK_N=BLOCK_N_FIXED,
        BLOCK_K=16,
    )
    return y


def maxpool2d_launch(x):
    N, C, IN_H, IN_W = x.shape
    OUT_H = IN_H // 2
    OUT_W = IN_W // 2
    y = torch.empty((N, C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)
    N_TOTAL = N * C * OUT_H * OUT_W

    grid = lambda meta: (triton.cdiv(N_TOTAL, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, y,
        N, C, IN_H, IN_W, OUT_H, OUT_W,
        N_TOTAL,
    )
    return y


def linear_relu_launch(x, weight, bias, apply_relu):
    M, K = x.shape
    N, K2 = weight.shape
    assert K == K2
    y = torch.empty((M, N), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_relu_kernel[grid](
        x, weight, bias, y,
        M, N, K,
        apply_relu,
    )
    return y


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_fc1_weight, p_fc1_bias,
    p_fc2_weight, p_fc2_bias,
    p_fc3_weight, p_fc3_bias,
):
    # Conv1 + ReLU: (N,1,32,32) -> (N,6,28,28)
    x = conv2d_relu_launch(x, p_conv1_weight, p_conv1_bias, apply_relu=True, stride=1, pad=0)

    # MaxPool: (N,6,28,28) -> (N,6,14,14)
    x = maxpool2d_launch(x)

    # Conv2 + ReLU: (N,6,14,14) -> (N,16,10,10)
    x = conv2d_relu_launch(x, p_conv2_weight, p_conv2_bias, apply_relu=True, stride=1, pad=0)

    # MaxPool: (N,16,10,10) -> (N,16,5,5)
    x = maxpool2d_launch(x)

    # Flatten: (N,16,5,5) -> (N,400)
    x = x.reshape(x.shape[0], -1)

    # FC1 + ReLU: (N,400) -> (N,120)
    x = linear_relu_launch(x, p_fc1_weight, p_fc1_bias, apply_relu=True)

    # FC2 + ReLU: (N,120) -> (N,84)
    x = linear_relu_launch(x, p_fc2_weight, p_fc2_bias, apply_relu=True)

    # FC3: (N,84) -> (N,20)
    x = linear_relu_launch(x, p_fc3_weight, p_fc3_bias, apply_relu=False)

    return x


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
    import torch.nn as nn
    import torch.nn.functional as F

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference implementation
        (p_conv1_weight, p_conv1_bias,
         p_conv2_weight, p_conv2_bias,
         p_fc1_weight, p_fc1_bias,
         p_fc2_weight, p_fc2_bias,
         p_fc3_weight, p_fc3_bias,
         x) = inputs

        ref = F.relu(F.conv2d(x, p_conv1_weight, p_conv1_bias))
        ref = F.max_pool2d(ref, kernel_size=2, stride=2)
        ref = F.relu(F.conv2d(ref, p_conv2_weight, p_conv2_bias))
        ref = F.max_pool2d(ref, kernel_size=2, stride=2)
        ref = ref.view(ref.shape[0], -1)
        ref = F.relu(F.linear(ref, p_fc1_weight, p_fc1_bias))
        ref = F.relu(F.linear(ref, p_fc2_weight, p_fc2_bias))
        ref = F.linear(ref, p_fc3_weight, p_fc3_bias)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")