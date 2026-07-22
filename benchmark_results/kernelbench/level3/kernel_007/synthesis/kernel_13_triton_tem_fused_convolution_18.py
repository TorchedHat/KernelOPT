import torch
import triton
import triton.language as tl
import math

# ============================================================
# Triton kernel: fused conv2d (implicit GEMM) + bias + relu
# ============================================================

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_relu_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    HAS_BIAS: tl.constexpr, DO_RELU: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    tmp_m = offs_m
    out_w_idx = tmp_m % OUT_W
    tmp_m2 = tmp_m // OUT_W
    out_h_idx = tmp_m2 % OUT_H
    batch_idx = tmp_m2 // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        kw_idx = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh_idx = tmp_k % KERNEL_H
        in_c_idx = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M
        valid_k = offs_k[None, :] < TOTAL_K
        valid_h = (ih >= 0) & (ih < IN_H)
        valid_w = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_h & valid_w

        x_idx = (batch_idx[:, None] * (IN_C * IN_H * IN_W) +
                 in_c_idx[None, :] * (IN_H * IN_W) +
        x_tile = tl.load(X_ptr + x_idx, mask=mask_x, other=0.0)

        # Weight: [OUT_C, IN_C, KH, KW] -> index [oc, in_c, kh, kw]
        w_idx = (offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) +
                 in_c_idx[:, None] * (KERNEL_H * KERNEL_W) +
                 kh_idx[:, None] * KERNEL_W +
                 kw_idx[:, None])
        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n
        w_tile = tl.load(W_ptr + w_idx, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    if HAS_BIAS:
        bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
        acc = acc + bias[None, :]

    # ReLU
    if DO_RELU:
        acc = tl.maximum(acc, 0.0)

    # Store
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < OUT_C)
    # Output layout: [BATCH, OUT_C, OUT_H, OUT_W] -> index [n, c, h, w]
    y_idx = (batch_idx[:, None] * (OUT_C * OUT_H * OUT_W) +
             offs_n[None, :] * (OUT_H * OUT_W) +
             out_h_idx[:, None] * OUT_W +
             out_w_idx[:, None])
    tl.store(Y_ptr + y_idx, acc.to(tl.float32), mask=mask_out)


def conv2d_triton(x, weight, bias, stride=(1,1), padding=(0,0), do_relu=False):
    """Triton conv2d with optional bias and relu."""
    N, C_in, H, W = x.shape
    C_out, C_in_w, KH, KW = weight.shape
    assert C_in == C_in_w

    SH, SW = stride if isinstance(stride, (tuple, list)) else (stride, stride)
    PH, PW = padding if isinstance(padding, (tuple, list)) else (padding, padding)

    OH = (H + 2 * PH - KH) // SH + 1
    OW = (W + 2 * PW - KW) // SW + 1

    y = torch.empty((N, C_out, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OH * OW
    BLOCK_M = 64
    BLOCK_N = 64

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(C_out, BLOCK_N))

    conv2d_relu_kernel[grid](
        y,
        N, C_in, H, W,
        C_out, OH, OW,
        KH, KW,
        SH, SW,
        PH, PW,
        bias is not None, do_relu,
        BLOCK_M, BLOCK_N, 32,
    )
    return y