import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Conv2d (implicit GEMM) + bias + ReLU + MaxPool2d (2x2, stride 2)
# Fuses: conv -> bias_add -> relu -> maxpool into one kernel
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W',
         'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_bias_relu_maxpool_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    POOL_H: tl.constexpr, POOL_W: tl.constexpr,
    POOL_OUT_H: tl.constexpr, POOL_OUT_W: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Computes conv2d (no padding, stride=1) + bias + relu + maxpool(2x2, stride=2).
    Each program computes a BLOCK_M x BLOCK_N tile of the conv output,
    
    M = BATCH * OUT_H * OUT_W  (conv output spatial positions)
    N = OUT_C
    K = IN_C * KERNEL_H * KERNEL_W
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Output channel indices
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < OUT_C

    # M indices: each corresponds to (batch, out_h, out_w)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    mask_m = offs_m < (BATCH * OUT_H * OUT_W)

    # Decode M -> (n, oh, ow)
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    n  = tmp // OUT_H

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K = IN_C * KERNEL_H * KERNEL_W
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOTAL_K

        # Decode K -> (ic, kh, kw)
        ic = offs_k // (KERNEL_H * KERNEL_W)
        krem = offs_k % (KERNEL_H * KERNEL_W)
        kh = krem // KERNEL_W
        kw = krem % KERNEL_W

        # Input coordinates
        ih = oh[:, None] + kh[None, :]   # [BLOCK_M, BLOCK_K]
        iw = ow[:, None] + kw[None, :]   # [BLOCK_M, BLOCK_K]

        # Input valid mask
        valid_x = (mask_m[:, None] & mask_k[None, :] &
                   (ih >= 0) & (ih < IN_H) &
                   (iw >= 0) & (iw < IN_W))

        # X: shape [BATCH, IN_C, IN_H, IN_W], stride [IN_C*IN_H*IN_W, IN_H*IN_W, IN_W, 1]
        x_idx = (n[:, None] * (IN_C * IN_H * IN_W) +
                 ic[None, :] * (IN_H * IN_W) +
        x_tile = tl.load(X_ptr + x_idx, mask=valid_x, other=0.0)

        # W: shape [OUT_C, IN_C, KERNEL_H, KERNEL_W]
        # stride [IN_C*KERNEL_H*KERNEL_W, KERNEL_H*KERNEL_W, KERNEL_W, 1]
        w_idx = (offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) +
                 ic[:, None] * (KERNEL_H * KERNEL_W) +
                 kh[:, None] * KERNEL_W + kw[:, None])
        valid_w = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(W_ptr + w_idx, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # MaxPool: for each conv output position (n, oh, ow, oc),
    # we need to find which pool output it belongs to and do a max reduction.
    # Pool output: pool_oh = oh // POOL_H, pool_ow = ow // POOL_W
    # Only positions where oh % POOL_H == 0 and ow % POOL_W == 0 are "anchor" positions.
    # We need to gather 2x2 windows. This is complex to do in a single pass here,
    # so we write the conv+relu output and do pooling separately.
    # Actually, let's just write the conv+relu output here.
    
    # Write conv+relu output: Y shape [BATCH, OUT_C, OUT_H, OUT_W]
    out_idx = (n[:, None] * (OUT_C * OUT_H * OUT_W) +
               offs_n[None, :] * (OUT_H * OUT_W) +
               oh[:, None] * OUT_W + ow[:, None])
    valid_out = mask_m[:, None] & mask_n[None, :]
    tl.store(Y_ptr + out_idx, acc.to(tl.float32), mask=valid_out)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: MaxPool2d (2x2, stride=2) pointwise
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2),
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
    """MaxPool2d with kernel=2, stride=2."""
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode: offs -> (n, c, ph, pw)
    pw = offs % OUT_W
    tmp = offs // OUT_W
    ph = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    n = tmp2 // C

    # Gather 2x2 window
    ih0 = ph * 2
    iw0 = pw * 2

    idx00 = n * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih0 * IN_W + iw0
    idx01 = idx00 + 1
    idx10 = idx00 + IN_W
    idx11 = idx00 + IN_W + 1

    v00 = tl.load(X_ptr + idx00, mask=mask, other=float('-inf'))
    v01 = tl.load(X_ptr + idx01, mask=mask, other=float('-inf'))
    v10 = tl.load(X_ptr + idx10, mask=mask, other=float('-inf'))
    v11 = tl.load(X_ptr + idx11, mask=mask, other=float('-inf'))

    result = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    tl.store(Y_ptr + offs, result, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: Linear + bias + ReLU (fused)
# ─────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 64},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 64},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32},  num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 32},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32},  num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    WITH_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """
    Y = relu(X @ W.T + B)  if WITH_RELU else X @ W.T + B
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < K

        # Load X tile [BLOCK_M, BLOCK_K]
        x_idx = offs_m[:, None] * K + offs_k[None, :]
        x_tile = tl.load(X_ptr + x_idx, mask=mask_m[:, None] & mask_k[None, :], other=0.0)

        # Load W tile [BLOCK_N, BLOCK_K] -> transpose to [BLOCK_K, BLOCK_N]
        w_idx = offs_n[:, None] * K + offs_k[None, :]
        w_tile = tl.load(W_ptr + w_idx, mask=mask_n[:, None] & mask_k[None, :], other=0.0)

        acc = tl.dot(x_tile, tl.trans(w_tile), acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=mask_n, other=0.0)
    acc = acc + bias[None, :]

    # Optional ReLU
    if WITH_RELU:
        acc = tl.maximum(acc, tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32))

    # Store
    out_idx = offs_m[:, None] * N + offs_n[None, :]
    tl.store(Y_ptr + out_idx, acc.to(tl.float32),
             mask=mask_m[:, None] & mask_n[None, :])


# ─────────────────────────────────────────────────────────────────────────────
# Main kernel_function
# ─────────────────────────────────────────────────────────────────────────────

def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_fc1_weight, p_fc1_bias,
    p_fc2_weight, p_fc2_bias,
    p_fc3_weight, p_fc3_bias,
    x,
):
    device = x.device
    dtype = x.dtype
    BATCH = x.shape[0]  # 4096

    # ── Conv1: [4096,1,32,32] -> [4096,6,28,28] ──────────────────────────────
    IN_C1, IN_H1, IN_W1 = 1, 32, 32
    OUT_C1, KH1, KW1 = 6, 5, 5
    OUT_H1 = IN_H1 - KH1 + 1  # 28
    OUT_W1 = IN_W1 - KW1 + 1  # 28

    conv1_out = torch.empty((BATCH, OUT_C1, OUT_H1, OUT_W1), device=device, dtype=dtype)

    M1 = BATCH * OUT_H1 * OUT_W1
    grid1 = lambda meta: (
        triton.cdiv(M1, meta['BLOCK_M']),
        triton.cdiv(OUT_C1, meta['BLOCK_N']),
    )
    conv2d_bias_relu_maxpool_kernel[grid1](
        x, p_conv1_weight, p_conv1_bias, conv1_out,
        BATCH=BATCH, IN_C=IN_C1, IN_H=IN_H1, IN_W=IN_W1,
        OUT_C=OUT_C1, OUT_H=OUT_H1, OUT_W=OUT_W1,
        KERNEL_H=KH1, KERNEL_W=KW1,
        POOL_H=2, POOL_W=2,
        POOL_OUT_H=OUT_H1 // 2, POOL_OUT_W=OUT_W1 // 2,
    )

    # ── MaxPool1: [4096,6,28,28] -> [4096,6,14,14] ───────────────────────────
    POOL_OUT_H1, POOL_OUT_W1 = 14, 14
    pool1_out = torch.empty((BATCH, OUT_C1, POOL_OUT_H1, POOL_OUT_W1), device=device, dtype=dtype)
    N_POOL1 = BATCH * OUT_C1 * POOL_OUT_H1 * POOL_OUT_W1

    grid_pool1 = lambda meta: (triton.cdiv(N_POOL1, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid_pool1](
        conv1_out, pool1_out,
        BATCH=BATCH, C=OUT_C1,
        IN_H=OUT_H1, IN_W=OUT_W1,
        OUT_H=POOL_OUT_H1, OUT_W=POOL_OUT_W1,
        N_TOTAL=N_POOL1,
    )

    # ── Conv2: [4096,6,14,14] -> [4096,16,10,10] ─────────────────────────────
    IN_C2, IN_H2, IN_W2 = 6, 14, 14
    OUT_C2, KH2, KW2 = 16, 5, 5
    OUT_H2 = IN_H2 - KH2 + 1  # 10
    OUT_W2 = IN_W2 - KW2 + 1  # 10

    conv2_out = torch.empty((BATCH, OUT_C2, OUT_H2, OUT_W2), device=device, dtype=dtype)

    M2 = BATCH * OUT_H2 * OUT_W2
    grid2 = lambda meta: (
        triton.cdiv(M2, meta['BLOCK_M']),
        triton.cdiv(OUT_C2, meta['BLOCK_N']),
    )
    conv2d_bias_relu_maxpool_kernel[grid2](
        pool1_out, p_conv2_weight, p_conv2_bias, conv2_out,
        BATCH=BATCH, IN_C=IN_C2, IN_H=IN_H2, IN_W=IN_W2,
        OUT_C=OUT_C2, OUT_H=OUT_H2, OUT_W=OUT_W2,
        KERNEL_H=KH2, KERNEL_W=KW2,
        POOL_H=2, POOL_W=2,
        POOL_OUT_H=OUT_H2 // 2, POOL_OUT_W=OUT_W2 // 2,
    )

    # ── MaxPool2: [4096,16,10,10] -> [4096,16,5,5] ───────────────────────────
    POOL_OUT_H2, POOL_OUT_W2 = 5, 5
    pool2_out = torch.empty((BATCH, OUT_C2, POOL_OUT_H2, POOL_OUT_W2), device=device, dtype=dtype)
    N_POOL2 = BATCH * OUT_C2 * POOL_OUT_H2 * POOL_OUT_W2

    grid_pool2 = lambda meta: (triton.cdiv(N_POOL2, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid_pool2](
        conv2_out, pool2_out,
        BATCH=BATCH, C=OUT_C2,
        IN_H=OUT_H2, IN_W=OUT_W2,
        OUT_H=POOL_OUT_H2, OUT_W=POOL_OUT_W2,
        N_TOTAL=N_POOL2,
    )

    # ── Flatten: [4096,16,5,5] -> [4096,400] ─────────────────────────────────
    flat = pool2_out.reshape(BATCH, 400)

    # ── FC1: [4096,400] -> [4096,120] + ReLU ─────────────────────────────────
    fc1_out = torch.empty((BATCH, 120), device=device, dtype=dtype)
    grid_fc1 = lambda meta: (
        triton.cdiv(BATCH, meta['BLOCK_M']),
        triton.cdiv(120, meta['BLOCK_N']),
    )
    linear_relu_kernel[grid_fc1](
        flat, p_fc1_weight, p_fc1_bias, fc1_out,
        M=BATCH, N=120, K=400,
        WITH_RELU=True,
    )

    # ── FC2: [4096,120] -> [4096,84] + ReLU ──────────────────────────────────
    fc2_out = torch.empty((BATCH, 84), device=device, dtype=dtype)
    grid_fc2 = lambda meta: (
        triton.cdiv(BATCH, meta['BLOCK_M']),
        triton.cdiv(84, meta['BLOCK_N']),
    )
    linear_relu_kernel[grid_fc2](
        fc1_out, p_fc2_weight, p_fc2_bias, fc2_out,
        M=BATCH, N=84, K=120,
        WITH_RELU=True,
    )

    # ── FC3: [4096,84] -> [4096,20] (no ReLU) ────────────────────────────────
    fc3_out = torch.empty((BATCH, 20), device=device, dtype=dtype)
    grid_fc3 = lambda meta: (
        triton.cdiv(BATCH, meta['BLOCK_M']),
        triton.cdiv(20, meta['BLOCK_N']),
    )
    linear_relu_kernel[grid_fc3](
        fc2_out, p_fc3_weight, p_fc3_bias, fc3_out,
        M=BATCH, N=20, K=84,
        WITH_RELU=False,
    )

    return fc3_out


def get_inputs():
    batch_size = 4096
    num_classes = 20
    p_conv1_weight = torch.randn(6, 1, 5, 5)
    p_conv1_bias   = torch.randn(6)
    p_conv2_weight = torch.randn(16, 6, 5, 5)
    p_conv2_bias   = torch.randn(16)
    p_fc1_weight   = torch.randn(120, 400)
    p_fc1_bias     = torch.randn(120)
    p_fc2_weight   = torch.randn(84, 120)
    p_fc2_bias     = torch.randn(84)
    p_fc3_weight   = torch.randn(20, 84)
    p_fc3_bias     = torch.randn(20)
    x              = torch.rand(batch_size, 1, 32, 32)
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
    inputs_cuda = [t.cuda() for t in inputs]

    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Reference implementation
        (p_conv1_weight, p_conv1_bias,
         p_conv2_weight, p_conv2_bias,
         p_fc1_weight, p_fc1_bias,
         p_fc2_weight, p_fc2_bias,
         p_fc3_weight, p_fc3_bias,
         x) = inputs_cuda

        ref = F.relu(F.conv2d(x, p_conv1_weight, p_conv1_bias))
        ref = F.max_pool2d(ref, kernel_size=2, stride=2)
        ref = F.relu(F.conv2d(ref, p_conv2_weight, p_conv2_bias))
        ref = F.max_pool2d(ref, kernel_size=2, stride=2)
        ref = ref.view(-1, 400)
        ref = F.relu(F.linear(ref, p_fc1_weight, p_fc1_bias))
        ref = F.relu(F.linear(ref, p_fc2_weight, p_fc2_bias))
        ref = F.linear(ref, p_fc3_weight, p_fc3_bias)

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")