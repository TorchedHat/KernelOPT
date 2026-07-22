import torch
import triton
import triton.language as tl

# ============================================================
# Kernel 1: Fused bias add + ReLU (pointwise)
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements', 'n_channels'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    n_elements, n_channels,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    c = offsets % n_channels
    bias = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + bias
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


# ============================================================
# Kernel 2: Max Pool 2x2 stride 2
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['n_out'],
)
@triton.jit
def maxpool2d_kernel(
    x_ptr, out_ptr,
    N, C, H_in, W_in, H_out, W_out,
    n_out,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_out

    # Decode: out index -> (n, c, oh, ow)
    ow = offsets % W_out
    tmp = offsets // W_out
    oh = tmp % H_out
    tmp2 = tmp // H_out
    c = tmp2 % C
    n = tmp2 // C

    # 2x2 max pool, stride 2
    ih0 = oh * 2
    iw0 = ow * 2

    base = n * (C * H_in * W_in) + c * (H_in * W_in)

    def load_val(ih, iw):
        idx = base + ih * W_in + iw
        valid = mask & (ih < H_in) & (iw < W_in)
        return tl.load(x_ptr + idx, mask=valid, other=-1e9)

    v00 = load_val(ih0,     iw0)
    v01 = load_val(ih0,     iw0 + 1)
    v10 = load_val(ih0 + 1, iw0)
    v11 = load_val(ih0 + 1, iw0 + 1)

    val = tl.maximum(tl.maximum(v00, v01), tl.maximum(v10, v11))
    tl.store(out_ptr + offsets, val, mask=mask)


# ============================================================
# Kernel 3: Conv2d (implicit GEMM) - general
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Out_ptr,
    N_batch, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KW
        kh = offs_k // (IN_C * KW)

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        x_mask = valid_m & valid_k & valid_ih & valid_iw

        x_idx = batch[:, None] * (IN_C * IN_H * IN_W) + in_c[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=x_mask, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> index [oc, in_c, kh, kw]
        # offs_n = oc, offs_k = (kh*KW*IN_C + kw*IN_C + in_c)
        w_oc = offs_n[None, :]  # [1, BLOCK_N]
        w_k = offs_k[:, None]   # [BLOCK_K, 1]
        w_in_c = in_c[:, None]
        w_kh = kh[:, None]
        w_kw = kw[:, None]
        w_idx = w_oc * (IN_C * KH * KW) + w_kh * (KW * IN_C) + w_kw * IN_C + w_in_c
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_tile = tl.load(W_ptr + w_idx, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    out_idx = offs_m[:, None] * OUT_C + offs_n[None, :]
    tl.store(Out_ptr + out_idx, acc, mask=out_mask)


# ============================================================
# Kernel 4: Linear (GEMM) with optional bias and ReLU
# ============================================================
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    X_ptr, W_ptr, B_ptr, Out_ptr,
    M, N, K,
    apply_relu: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)

        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=x_mask, other=0.0)
        w_tile = tl.load(W_ptr + offs_k[:, None] * N + offs_n[None, :], mask=w_mask, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    b_mask = offs_n < N
    bias = tl.load(B_ptr + offs_n, mask=b_mask, other=0.0)
    acc = acc + bias[None, :]

    if apply_relu:
        acc = tl.where(acc > 0.0, acc, 0.0)

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Out_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=out_mask)


def run_conv2d(x, weight, bias, stride=1, padding=0):
    """Run conv2d using implicit GEMM kernel, output in NCHW format."""
    N_batch, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    OUT_H = (IN_H + 2 * padding - KH) // stride + 1
    OUT_W = (IN_W + 2 * padding - KW) // stride + 1

    M = N_batch * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KH * KW

    # Output in (M, N) = (N_batch*OUT_H*OUT_W, OUT_C) layout
    out_flat = torch.empty((M, N), device=x.device, dtype=torch.float32)

    # Weight needs to be in (OUT_C, IN_C*KH*KW) but our kernel indexes as
    # w[oc, kh, kw, in_c] = w[oc * IN_C*KH*KW + kh*KW*IN_C + kw*IN_C + in_c]
    # PyTorch weight is [OUT_C, IN_C, KH, KW]
    # We need to reorder to [OUT_C, KH, KW, IN_C] for our K decomposition
    # K index: in_c + kw*IN_C + kh*KW*IN_C
    # So weight should be indexed as w[oc, kh, kw, in_c]
    # PyTorch: w[oc, ic, kh, kw] -> permute to [oc, kh, kw, ic] then flatten
    w_reordered = weight.permute(0, 2, 3, 1).contiguous()  # [OUT_C, KH, KW, IN_C]

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    conv2d_implicit_gemm_kernel[grid](
        x, w_reordered, out_flat,
        N_batch, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride, stride,
        padding, padding,
        M, N, K,
    )

    # out_flat is (N_batch*OUT_H*OUT_W, OUT_C), reshape to (N_batch, OUT_H, OUT_W, OUT_C)
    # then permute to (N_batch, OUT_C, OUT_H, OUT_W)
    out = out_flat.view(N_batch, OUT_H, OUT_W, OUT_C).permute(0, 3, 1, 2).contiguous()

    # Add bias
    if bias is not None:
        n_elements = out.numel()
        n_channels = OUT_C
        # bias is per output channel, out is (N, C, H, W)
        # We need to add bias[c] to out[:, c, :, :]
        # Use the bias_relu kernel but without relu for now
        # Actually let's just do it inline
        out = out + bias.view(1, OUT_C, 1, 1)

    return out


def run_maxpool2d(x, kernel_size=2, stride=2):
    N, C, H_in, W_in = x.shape
    H_out = (H_in - kernel_size) // stride + 1
    W_out = (W_in - kernel_size) // stride + 1
    n_out = N * C * H_out * W_out
    out = torch.empty((N, C, H_out, W_out), device=x.device, dtype=x.dtype)

    grid = lambda meta: (triton.cdiv(n_out, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, out,
        N, C, H_in, W_in, H_out, W_out,
        n_out,
    )
    return out


def run_relu(x):
    out = torch.empty_like(x)
    n_elements = x.numel()
    # Simple pointwise relu
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # Reuse bias_relu_kernel with zero bias? No, let's make a separate relu kernel
    # Actually let's just use bias_relu with a zero bias trick - but that's wasteful
    # Let's inline relu using the bias_relu_kernel with a dummy bias of 0
    # Better: use a simple pointwise kernel
    relu_pointwise_kernel[grid](x, out, n_elements)
    return out


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_pointwise_kernel(x_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    out = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def run_linear(x, weight, bias, apply_relu=False):
    """x: (M, K), weight: (N, K), bias: (N,) -> out: (M, N)"""
    M, K = x.shape
    N = weight.shape[0]

    # weight is (N, K), we need (K, N) for the GEMM
    w_t = weight.t().contiguous()  # (K, N)

    out = torch.empty((M, N), device=x.device, dtype=torch.float32)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    linear_relu_kernel[grid](
        x, w_t, bias, out,
        M, N, K,
        apply_relu,
    )
    return out


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_fc1_weight, p_fc1_bias,
    p_fc2_weight, p_fc2_bias,
    p_fc3_weight, p_fc3_bias,
    x,
):
    # Conv1: (4096, 1, 32, 32) -> (4096, 6, 28, 28)
    x = run_conv2d(x, p_conv1_weight, p_conv1_bias, stride=1, padding=0)
    # ReLU
    x = run_relu(x)
    # MaxPool: (4096, 6, 28, 28) -> (4096, 6, 14, 14)
    x = run_maxpool2d(x, kernel_size=2, stride=2)

    # Conv2: (4096, 6, 14, 14) -> (4096, 16, 10, 10)
    x = run_conv2d(x, p_conv2_weight, p_conv2_bias, stride=1, padding=0)
    # ReLU
    x = run_relu(x)
    # MaxPool: (4096, 16, 10, 10) -> (4096, 16, 5, 5)
    x = run_maxpool2d(x, kernel_size=2, stride=2)

    # Flatten: (4096, 16, 5, 5) -> (4096, 400)
    x = x.contiguous().view(4096, 400)

    # FC1: (4096, 400) -> (4096, 120) + ReLU
    x = run_linear(x, p_fc1_weight, p_fc1_bias, apply_relu=True)

    # FC2: (4096, 120) -> (4096, 84) + ReLU
    x = run_linear(x, p_fc2_weight, p_fc2_bias, apply_relu=True)

    # FC3: (4096, 84) -> (4096, 20)
    x = run_linear(x, p_fc3_weight, p_fc3_bias, apply_relu=False)

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
        # Reference implementation
        (p_conv1_weight, p_conv1_bias,
         p_conv2_weight, p_conv2_bias,
         p_fc1_weight, p_fc1_bias,
         p_fc2_weight, p_fc2_bias,
         p_fc3_weight, p_fc3_bias,
         x) = inputs

        ref = F.relu(F.conv2d(x, p_conv1_weight, p_conv1_bias, stride=1, padding=0))
        ref = F.max_pool2d(ref, kernel_size=2, stride=2)
        ref = F.relu(F.conv2d(ref, p_conv2_weight, p_conv2_bias, stride=1, padding=0))
        ref = F.max_pool2d(ref, kernel_size=2, stride=2)
        ref = ref.view(4096, 400)
        ref = F.relu(F.linear(ref, p_fc1_weight, p_fc1_bias))
        ref = F.relu(F.linear(ref, p_fc2_weight, p_fc2_bias))
        ref = F.linear(ref, p_fc3_weight, p_fc3_bias)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")