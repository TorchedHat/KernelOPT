import torch
import triton
import triton.language as tl


# Squeeze conv: 1x1, in_channels=3, out_channels=6
# Implicit GEMM: M = N*H*W = 128*256*256, N_out=6, K=3
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 8, 'BLOCK_K': 8}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N_OUT', 'K_IN'],
)
@triton.jit
def squeeze_conv_kernel(
    X_ptr, W_ptr, B_ptr, OUT_ptr,
    M, N_OUT, K_IN,
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

    for k in range(0, tl.cdiv(K_IN, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        # Load X tile [BLOCK_M, BLOCK_K]
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K_IN)
        x_ptrs = X_ptr + offs_m[:, None] * stride_xm + k_offs[None, :] * stride_xk
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        # Load W tile [BLOCK_K, BLOCK_N] - weight is (N_OUT, K_IN), we need (K_IN, N_OUT)
        w_mask = (k_offs[:, None] < K_IN) & (offs_n[None, :] < N_OUT)
        w_ptrs = W_ptr + offs_n[None, :] * stride_wn + k_offs[:, None] * stride_wk
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias and ReLU
    bias_mask = offs_n < N_OUT
    bias = tl.load(B_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    # Store output
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
    out_ptrs = OUT_ptr + offs_m[:, None] * N_OUT + offs_n[None, :]
    tl.store(out_ptrs, acc, mask=out_mask)


# Expand1x1 conv: 1x1, in_channels=6, out_channels=64
# Writes to output[:, :64, :, :] in NCHW layout
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N_OUT', 'K_IN'],
)
@triton.jit
def expand1x1_conv_kernel(
    X_ptr, W_ptr, B_ptr, OUT_ptr,
    M, N_OUT, K_IN,
    OUT_CHANNELS_TOTAL,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K_IN, BLOCK_K)):
        k_offs = k * BLOCK_K + offs_k
        x_mask = (offs_m[:, None] < M) & (k_offs[None, :] < K_IN)
        x_ptrs = X_ptr + offs_m[:, None] * K_IN + k_offs[None, :]
        x_tile = tl.load(x_ptrs, mask=x_mask, other=0.0)

        w_mask = (k_offs[:, None] < K_IN) & (offs_n[None, :] < N_OUT)
        w_ptrs = W_ptr + offs_n[None, :] * K_IN + k_offs[:, None]
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    bias_mask = offs_n < N_OUT
    bias = tl.load(B_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    # Output is in NCHW layout: [128, 128, 256, 256]
    # offs_m indexes spatial position (batch * H * W)
    # offs_n indexes channel (0..63 for expand1x1)
    # In NCHW: index = batch * (128 * H * W) + channel * (H * W) + spatial
    # spatial = offs_m % (H*W), batch = offs_m // (H*W)
    # But we can use a different layout: store as (M, N_OUT) then reshape
    # Actually, let's write directly to NCHW output
    # output[batch, channel, h, w] = output[batch * 128 * HW + channel * HW + hw_idx]
    # offs_m = batch * HW + hw_idx
    # batch = offs_m // HW, hw_idx = offs_m % HW
    # linear_idx = batch * OUT_CHANNELS_TOTAL * HW + channel * HW + hw_idx
    HW = M // 128  # H*W = 256*256 = 65536
    batch = offs_m // HW
    hw_idx = offs_m % HW

    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
    # channel offset = offs_n (0..63)
    linear_idx = batch[:, None] * (OUT_CHANNELS_TOTAL * HW) + offs_n[None, :] * HW + hw_idx[:, None]
    tl.store(OUT_ptr + linear_idx, acc, mask=out_mask)


# Expand3x3 conv: 3x3 pad=1, in_channels=6, out_channels=64
# Implicit GEMM with K=6*3*3=54
# Writes to output[:, 64:, :, :] in NCHW layout
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 64}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N_OUT', 'K_TOTAL'],
)
@triton.jit
def expand3x3_conv_kernel(
    X_ptr, W_ptr, B_ptr, OUT_ptr,
    M, N_OUT, K_TOTAL,
    IN_C, IN_H, IN_W,
    OUT_H, OUT_W,
    BATCH,
    OUT_CHANNELS_TOTAL,
    CHANNEL_OFFSET,
    PAD_H, PAD_W,
    STRIDE_H, STRIDE_W,
    KERNEL_H, KERNEL_W,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode offs_m -> (batch, out_h, out_w)
    HW = OUT_H * OUT_W
    batch = offs_m // HW
    hw_rem = offs_m % HW
    out_h = hw_rem // OUT_W
    out_w = hw_rem % OUT_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k in range(0, tl.cdiv(K_TOTAL, BLOCK_K)):
        offs_k = k * BLOCK_K + tl.arange(0, BLOCK_K)

        # Decode offs_k -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kh = (offs_k // IN_C) % KERNEL_H
        kw = offs_k // (IN_C * KERNEL_H)

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid = (
            (offs_m[:, None] < M) &
            (offs_k[None, :] < K_TOTAL) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W)
        )

        # X is stored as (M, IN_C) where M = BATCH * IN_H * IN_W
        # x[batch, in_c, ih, iw] -> linear: batch * IN_H * IN_W * IN_C + ih * IN_W * IN_C + iw * IN_C + in_c
        x_linear = batch[:, None] * (IN_H * IN_W * IN_C) + ih * IN_W * IN_C + iw * IN_C + in_c[None, :]
        x_tile = tl.load(X_ptr + x_linear, mask=valid, other=0.0)

        # W is (N_OUT, IN_C, KH, KW) -> we need (K_TOTAL, N_OUT)
        # w[n, c, kh, kw] -> linear: n * IN_C * KH * KW + c * KH * KW + kh * KW + kw
        # K encoding: k = kw * IN_C + c + kh * IN_C * KW  -- wait, let me re-check
        # offs_k = kw * IN_C * KERNEL_H + kh * IN_C + in_c  -- no
        # in_c = offs_k % IN_C
        # kh = (offs_k // IN_C) % KERNEL_H
        # kw = offs_k // (IN_C * KERNEL_H)
        # So w index: n * IN_C * KH * KW + in_c * KH * KW + kh * KW + kw
        w_mask = (offs_k[:, None] < K_TOTAL) & (offs_n[None, :] < N_OUT)
        w_linear = offs_n[None, :] * (IN_C * KERNEL_H * KERNEL_W) + in_c[:, None] * (KERNEL_H * KERNEL_W) + kh[:, None] * KERNEL_W + kw[:, None]
        w_tile = tl.load(W_ptr + w_linear, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    bias_mask = offs_n < N_OUT
    bias = tl.load(B_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]
    acc = tl.maximum(acc, 0.0)

    # Write to output NCHW at channel offset CHANNEL_OFFSET
    out_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
    linear_idx = batch[:, None] * (OUT_CHANNELS_TOTAL * HW) + (CHANNEL_OFFSET + offs_n[None, :]) * HW + hw_rem[:, None]
    tl.store(OUT_ptr + linear_idx, acc, mask=out_mask)


def kernel_function(
    p_squeeze_weight, p_squeeze_bias,
    p_expand1x1_weight, p_expand1x1_bias,
    p_expand3x3_weight, p_expand3x3_bias,
):
    BATCH, IN_C, IN_H, IN_W = x.shape  # 128, 3, 256, 256
    SQ_C = p_squeeze_weight.shape[0]   # 6
    EX1_C = p_expand1x1_weight.shape[0]  # 64
    EX3_C = p_expand3x3_weight.shape[0]  # 64
    OUT_C = EX1_C + EX3_C  # 128

    device = x.device
    dtype = x.dtype

    # Reshape x to (M, IN_C) for squeeze conv (1x1)
    # x is NCHW, we need NHWC for the GEMM
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [128, 256, 256, 3]
    M = BATCH * IN_H * IN_W  # 8388608

    # Squeeze conv output: (M, SQ_C) in NHWC layout
    sq_out = torch.empty((M, SQ_C), device=device, dtype=dtype)

    # Squeeze weight: (SQ_C, IN_C, 1, 1) -> (SQ_C, IN_C)
    sq_w = p_squeeze_weight.reshape(SQ_C, IN_C).contiguous()

    grid_sq = (triton.cdiv(M, 128), triton.cdiv(SQ_C, 16))
    squeeze_conv_kernel[grid_sq](
        x_nhwc, sq_w, p_squeeze_bias, sq_out,
        M, SQ_C, IN_C,
    )

    # Output tensor: [128, 128, 256, 256] NCHW
    out = torch.empty((BATCH, OUT_C, IN_H, IN_W), device=device, dtype=dtype)

    # Expand1x1 conv: sq_out (M, SQ_C) -> out[:, :64, :, :]
    ex1_w = p_expand1x1_weight.reshape(EX1_C, SQ_C).contiguous()
    HW = IN_H * IN_W

    grid_ex1 = (triton.cdiv(M, 64), triton.cdiv(EX1_C, 64))
    expand1x1_conv_kernel[grid_ex1](
        sq_out, ex1_w, p_expand1x1_bias, out,
        M, EX1_C, SQ_C,
        OUT_C,
    )

    # Expand3x3 conv: sq_out (M, SQ_C) -> out[:, 64:, :, :]
    # sq_out is (M, SQ_C) = (BATCH*H*W, 6) in NHWC layout
    K_TOTAL = SQ_C * 3 * 3  # 54

    grid_ex3 = (triton.cdiv(M, 64), triton.cdiv(EX3_C, 64))
    expand3x3_conv_kernel[grid_ex3](
        sq_out, p_expand3x3_weight, p_expand3x3_bias, out,
        M, EX3_C, K_TOTAL,
        SQ_C, IN_H, IN_W,
        IN_H, IN_W,
        BATCH,
        OUT_C,
    )

    return out


def get_inputs():
    p_squeeze_weight = torch.randn(6, 3, 1, 1)
    p_squeeze_bias = torch.randn(6)
    p_expand1x1_weight = torch.randn(64, 6, 1, 1)
    p_expand1x1_bias = torch.randn(64)
    p_expand3x3_weight = torch.randn(64, 6, 3, 3)
    p_expand3x3_bias = torch.randn(64)
    x = torch.randn(128, 3, 256, 256)
    return [p_squeeze_weight, p_squeeze_bias, p_expand1x1_weight, p_expand1x1_bias,


if __name__ == "__main__":
    import argparse
    import torch.nn.functional as F

    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]

    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()

    if not args.profile_only:
        p_sq_w, p_sq_b, p_ex1_w, p_ex1_b, p_ex3_w, p_ex3_b, x = inputs_cuda

        # Reference computation
        sq = F.relu(F.conv2d(x, p_sq_w, p_sq_b))
        ex1 = F.relu(F.conv2d(sq, p_ex1_w, p_ex1_b))
        ex3 = F.relu(F.conv2d(sq, p_ex3_w, p_ex3_b, padding=1))
        ref = torch.cat([ex1, ex3], dim=1)

        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")