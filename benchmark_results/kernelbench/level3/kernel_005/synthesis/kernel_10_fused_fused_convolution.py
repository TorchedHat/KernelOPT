import torch
import triton
import triton.language as tl
import math
import argparse

# ─── Fused bias + ReLU pointwise kernel ───────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel', 'C'],
)
@triton.jit
def bias_relu_kernel(
    x_ptr, bias_ptr, out_ptr,
    numel, C,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    c = offs % C
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    out = x + b
    out = tl.where(out > 0.0, out, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


# ─── ReLU pointwise kernel ────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
    ],
    key=['numel'],
)
@triton.jit
def relu_kernel(
    x_ptr, out_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < numel
    x = tl.load(x_ptr + offs, mask=mask, other=0.0)
    out = tl.where(x > 0.0, x, 0.0)
    tl.store(out_ptr + offs, out, mask=mask)


# ─── MaxPool2d kernel (kernel=3, stride=2, padding=0) ─────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 64}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
    ],
    key=['N', 'C', 'OH', 'OW', 'IH', 'IW'],
)
@triton.jit
def maxpool2d_kernel(
    inp_ptr, out_ptr,
    N, C, IH, IW, OH, OW,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    total = N * C * OH * OW
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total

    ow = offs % OW
    tmp = offs // OW
    oh = tmp % OH
    tmp2 = tmp // OH
    c = tmp2 % C
    n = tmp2 // C

    max_val = tl.full([BLOCK_SIZE], float('-inf'), dtype=tl.float32)

    for kh in tl.static_range(KH):
        for kw in tl.static_range(KW):
            ih = oh * SH + kh - PH
            iw = ow * SW + kw - PW
            valid = mask & (ih >= 0) & (ih < IH) & (iw >= 0) & (iw < IW)
            idx = n * (C * IH * IW) + c * (IH * IW) + ih * IW + iw
            val = tl.load(inp_ptr + idx, mask=valid, other=float('-inf'))
            max_val = tl.where(valid & (val > max_val), val, max_val)

    tl.store(out_ptr + offs, max_val, mask=mask)


# ─── Implicit GEMM Conv2d kernel ──────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH: tl.constexpr, KW: tl.constexpr,
    SH: tl.constexpr, SW: tl.constexpr,
    PH: tl.constexpr, PW: tl.constexpr,
    M_TOTAL, TOTAL_K,
    HAS_BIAS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (n, oh, ow)
    ow = offs_m % OUT_W
    tmp = offs_m // OUT_W
    oh = tmp % OUT_H
    n = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, ic)
        ic = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KW
        kh = tmp_k // KW

        ih = oh[:, None] * SH + kh[None, :] - PH
        iw = ow[:, None] * SW + kw[None, :] - PW

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_n = n[:, None] < N
        mask_x = valid_m & valid_k & valid_ih & valid_iw & valid_n

        x_idx = n[:, None] * (IN_C * IN_H * IN_W) + ic[None, :] * (IN_H * IN_W) + ih * IN_W + iw
        x_tile = tl.load(X_ptr + x_idx, mask=mask_x, other=0.0)

        # W shape: (OUT_C, IN_C, KH, KW) -> index: oc * (IN_C*KH*KW) + ic * (KH*KW) + kh * KW + kw
        w_idx = offs_n[None, :] * TOTAL_K + ic[:, None] * (KH * KW) + kh[:, None] * KW + kw[:, None]
        mask_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_tile = tl.load(W_ptr + w_idx, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias and store
    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + bias[None, :]

    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    # Output in NCHW: Y[n, oc, oh, ow]
    out_idx = n[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + oh[:, None] * OUT_W + ow[:, None]
    tl.store(Y_ptr + out_idx, acc, mask=mask_out)


# ─── Fused Linear + Bias + ReLU using tl.dot ──────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 128}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def linear_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    M, N, K,
    HAS_BIAS: tl.constexpr,
    DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_x = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        mask_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        x_tile = tl.load(X_ptr + offs_m[:, None] * K + offs_k[None, :], mask=mask_x, other=0.0)
        w_tile = tl.load(W_ptr + offs_k[:, None] * N + offs_n[None, :], mask=mask_w, other=0.0)
        acc = tl.dot(x_tile, w_tile, acc=acc)

    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < N, other=0.0)
        acc = acc + bias[None, :]

    if DO_RELU:
        acc = tl.where(acc > 0.0, acc, 0.0)

    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(Y_ptr + offs_m[:, None] * N + offs_n[None, :], acc, mask=mask_out)


def conv2d_triton(x, weight, bias, stride, padding):
    """Run conv2d using implicit GEMM Triton kernel."""
    N, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    SH, SW = stride
    PH, PW = padding

    OUT_H = (IN_H + 2 * PH - KH) // SH + 1
    OUT_W = (IN_W + 2 * PW - KW) // SW + 1
    M_TOTAL = N * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW

    # Flatten weight to (OUT_C, TOTAL_K)
    w_flat = weight.reshape(OUT_C, TOTAL_K).contiguous()
    x_cont = x.contiguous()

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    has_bias = bias is not None
    b_ptr = bias if has_bias else x  # dummy

    conv2d_implicit_gemm_kernel[grid](
        x_cont, w_flat, b_ptr, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW, SH, SW, PH, PW,
        M_TOTAL, TOTAL_K,
        HAS_BIAS=has_bias,
    )
    return y


def maxpool2d_triton(x, kernel_size=3, stride=2, padding=0):
    N, C, IH, IW = x.shape
    OH = (IH + 2 * padding - kernel_size) // stride + 1
    OW = (IW + 2 * padding - kernel_size) // stride + 1
    out = torch.empty((N, C, OH, OW), device=x.device, dtype=x.dtype)
    total = N * C * OH * OW
    grid = lambda meta: (triton.cdiv(total, meta['BLOCK_SIZE']),)
    maxpool2d_kernel[grid](
        x, out,
        N, C, IH, IW, OH, OW,
        kernel_size, kernel_size,
        stride, stride,
        padding, padding,
    )
    return out


def relu_triton(x):
    out = torch.empty_like(x)
    numel = x.numel()
    grid = lambda meta: (triton.cdiv(numel, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, out, numel)
    return out


def linear_triton(x, weight, bias, do_relu=False):
    """x: (M, K), weight: (N, K), bias: (N,) -> (M, N)"""
    M, K = x.shape
    N = weight.shape[0]
    # weight is (N, K), we need (K, N) for the matmul
    w_t = weight.t().contiguous()
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N, meta['BLOCK_N']),
    )
    has_bias = bias is not None
    b_ptr = bias if has_bias else x
    linear_relu_kernel[grid](
        x, w_t, b_ptr, out,
        M, N, K,
        HAS_BIAS=has_bias,
        DO_RELU=do_relu,
    )
    return out


def kernel_function(
    p_conv1_weight, p_conv1_bias,
    p_conv2_weight, p_conv2_bias,
    p_conv3_weight, p_conv3_bias,
    p_conv4_weight, p_conv4_bias,
    p_conv5_weight, p_conv5_bias,
    p_fc1_weight, p_fc1_bias,
    p_fc2_weight, p_fc2_bias,
    p_fc3_weight, p_fc3_bias,
    x,
):
    # Conv1: (1024,3,224,224) -> (1024,96,55,55), stride=4, pad=2, kernel=11
    x = conv2d_triton(x, p_conv1_weight, p_conv1_bias, stride=(4, 4), padding=(2, 2))
    x = relu_triton(x)
    x = maxpool2d_triton(x, kernel_size=3, stride=2, padding=0)

    # Conv2: (1024,96,27,27) -> (1024,256,27,27), stride=1, pad=2, kernel=5
    x = conv2d_triton(x, p_conv2_weight, p_conv2_bias, stride=(1, 1), padding=(2, 2))
    x = relu_triton(x)
    x = maxpool2d_triton(x, kernel_size=3, stride=2, padding=0)

    # Conv3: (1024,256,13,13) -> (1024,384,13,13), stride=1, pad=1, kernel=3
    x = conv2d_triton(x, p_conv3_weight, p_conv3_bias, stride=(1, 1), padding=(1, 1))
    x = relu_triton(x)

    # Conv4: (1024,384,13,13) -> (1024,384,13,13), stride=1, pad=1, kernel=3
    x = conv2d_triton(x, p_conv4_weight, p_conv4_bias, stride=(1, 1), padding=(1, 1))
    x = relu_triton(x)

    # Conv5: (1024,384,13,13) -> (1024,256,13,13), stride=1, pad=1, kernel=3
    x = conv2d_triton(x, p_conv5_weight, p_conv5_bias, stride=(1, 1), padding=(1, 1))
    x = relu_triton(x)
    x = maxpool2d_triton(x, kernel_size=3, stride=2, padding=0)

    # Flatten: (1024, 9216)
    x = x.reshape(x.shape[0], -1).contiguous()

    # FC1: (1024, 9216) -> (1024, 4096) + ReLU
    x = linear_triton(x, p_fc1_weight, p_fc1_bias, do_relu=True)

    # FC2: (1024, 4096) -> (1024, 4096) + ReLU
    x = linear_triton(x, p_fc2_weight, p_fc2_bias, do_relu=True)

    # FC3: (1024, 4096) -> (1024, 1000)
    x = linear_triton(x, p_fc3_weight, p_fc3_bias, do_relu=False)

    return x


def get_inputs():
    torch.manual_seed(42)
    batch_size = 1024
    p_conv1_weight = torch.randn(96, 3, 11, 11, device='cuda')
    p_conv1_bias = torch.randn(96, device='cuda')
    p_conv2_weight = torch.randn(256, 96, 5, 5, device='cuda')
    p_conv2_bias = torch.randn(256, device='cuda')
    p_conv3_weight = torch.randn(384, 256, 3, 3, device='cuda')
    p_conv3_bias = torch.randn(384, device='cuda')
    p_conv4_weight = torch.randn(384, 384, 3, 3, device='cuda')
    p_conv4_bias = torch.randn(384, device='cuda')
    p_conv5_weight = torch.randn(256, 384, 3, 3, device='cuda')
    p_conv5_bias = torch.randn(256, device='cuda')
    p_fc1_weight = torch.randn(4096, 9216, device='cuda')
    p_fc1_bias = torch.randn(4096, device='cuda')
    p_fc2_weight = torch.randn(4096, 4096, device='cuda')
    p_fc2_bias = torch.randn(4096, device='cuda')
    p_fc3_weight = torch.randn(1000, 4096, device='cuda')
    p_fc3_bias = torch.randn(1000, device='cuda')
    x = torch.randn(batch_size, 3, 224, 224, device='cuda')
    return [
        p_conv1_weight, p_conv1_bias,
        p_conv2_weight, p_conv2_bias,
        p_conv3_weight, p_conv3_bias,
        p_conv4_weight, p_conv4_bias,
        p_conv5_weight, p_conv5_bias,
        p_fc1_weight, p_fc1_bias,
        p_fc2_weight, p_fc2_bias,
        p_fc3_weight, p_fc3_bias,
        x,
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F

        (p_conv1_weight, p_conv1_bias,
         p_conv2_weight, p_conv2_bias,
         p_conv3_weight, p_conv3_bias,
         p_conv4_weight, p_conv4_bias,
         p_conv5_weight, p_conv5_bias,
         p_fc1_weight, p_fc1_bias,
         p_fc2_weight, p_fc2_bias,
         p_fc3_weight, p_fc3_bias,
         x) = inputs

        # Reference using PyTorch
        ref = F.conv2d(x, p_conv1_weight, p_conv1_bias, stride=4, padding=2)
        ref = F.relu(ref)
        ref = F.max_pool2d(ref, kernel_size=3, stride=2)
        ref = F.conv2d(ref, p_conv2_weight, p_conv2_bias, stride=1, padding=2)
        ref = F.relu(ref)
        ref = F.max_pool2d(ref, kernel_size=3, stride=2)
        ref = F.conv2d(ref, p_conv3_weight, p_conv3_bias, stride=1, padding=1)
        ref = F.relu(ref)
        ref = F.conv2d(ref, p_conv4_weight, p_conv4_bias, stride=1, padding=1)
        ref = F.relu(ref)
        ref = F.conv2d(ref, p_conv5_weight, p_conv5_bias, stride=1, padding=1)
        ref = F.relu(ref)
        ref = F.max_pool2d(ref, kernel_size=3, stride=2)
        ref = ref.flatten(1)
        ref = F.linear(ref, p_fc1_weight, p_fc1_bias)
        ref = F.relu(ref)
        ref = F.linear(ref, p_fc2_weight, p_fc2_bias)
        ref = F.relu(ref)
        ref = F.linear(ref, p_fc3_weight, p_fc3_bias)

        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")