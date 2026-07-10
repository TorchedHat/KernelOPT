import torch
import triton
import triton.language as tl

# ─────────────────────────────────────────────────────────────────────────────
# Kernel 1: Conv3d (implicit GEMM)
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1024, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'IN_C'],
)
@triton.jit
def conv3d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH, IN_C, IN_D, IN_H, IN_W,
    OUT_C, OUT_D, OUT_H, OUT_W,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    M_TOTAL,
    # strides for X: (batch, c, d, h, w)
    sx_n, sx_c, sx_d, sx_h, sx_w,
    # strides for W: (out_c, in_c, kd, kh, kw)
    sw_oc, sw_ic, sw_kd, sw_kh, sw_kw,
    # strides for Y: (batch, c, d, h, w)
    sy_n, sy_c, sy_d, sy_h, sy_w,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: (batch, out_d, out_h, out_w)
    tmp = offs_m
    out_w = tmp % OUT_W
    tmp = tmp // OUT_W
    out_h = tmp % OUT_H
    tmp = tmp // OUT_H
    out_d = tmp % OUT_D
    batch = tmp // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KD * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K: (in_c, kd, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        tmp_k = tmp_k // KH
        kd_idx = tmp_k % KD
        ic_idx = tmp_k // KD

        # Input coordinates
        id_ = out_d[:, None] * STRIDE_D + kd_idx[None, :] - PAD_D
        ih_ = out_h[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw_ = out_w[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = (batch < BATCH) & (out_d < OUT_D) & (out_h < OUT_H) & (out_w < OUT_W)
        valid_k = offs_k < TOTAL_K
        valid_spatial = (id_ >= 0) & (id_ < IN_D) & (ih_ >= 0) & (ih_ < IN_H) & (iw_ >= 0) & (iw_ < IN_W)
        mask_x = valid_m[:, None] & valid_k[None, :] & valid_spatial

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight tile [BLOCK_K, BLOCK_N]
        mask_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=(offs_n < OUT_C), other=0.0)
    acc = acc + bias[None, :]

    # Store
    mask_out = ((batch < BATCH) & (out_d < OUT_D) & (out_h < OUT_H) & (out_w < OUT_W))[:, None] & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 2: Softmax over channel dim (dim=1)
# Each program handles one (batch, d, h, w) position, reduces over C=16
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 16, 'XBLOCK': 64}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_C': 16, 'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_C': 16, 'XBLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_C': 16, 'XBLOCK': 512}, num_warps=8, num_stages=1),
    ],
    key=['N_SPATIAL', 'C'],
)
@triton.jit
def softmax_channel_kernel(
    X_ptr, Y_ptr,
    C, N_SPATIAL,
    stride_c,
    BLOCK_C: tl.constexpr,
    XBLOCK: tl.constexpr,
):
    # Each program handles XBLOCK spatial positions
    pid = tl.program_id(0)
    offs_spatial = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask_spatial = offs_spatial < N_SPATIAL

    # Load all C channels for each spatial position
    offs_c = tl.arange(0, BLOCK_C)  # BLOCK_C >= C

    # X layout: (batch, C, D, H, W) contiguous
    # spatial index encodes (batch, d, h, w) flattened over (D*H*W) per batch
    # stride_c = D*H*W (stride between channels)
    # base pointer for each spatial position
    x_base = offs_spatial  # index into the DHW*batch flat space

    # Load [XBLOCK, BLOCK_C]
    ptrs = X_ptr + x_base[:, None] + offs_c[None, :] * stride_c
    mask = mask_spatial[:, None] & (offs_c[None, :] < C)
    vals = tl.load(ptrs, mask=mask, other=-float('inf'))

    # Softmax
    row_max = tl.max(vals, axis=1)  # [XBLOCK]
    vals_shifted = vals - row_max[:, None]
    exp_vals = tl.exp(vals_shifted)
    # mask out-of-range channels for sum
    exp_vals = tl.where(mask, exp_vals, 0.0)
    row_sum = tl.sum(exp_vals, axis=1)  # [XBLOCK]
    softmax_vals = exp_vals / row_sum[:, None]

    # Store
    out_ptrs = Y_ptr + x_base[:, None] + offs_c[None, :] * stride_c
    tl.store(out_ptrs, softmax_vals, mask=mask)


# ─────────────────────────────────────────────────────────────────────────────
# Kernel 3: MaxPool3d with kernel=2, stride=2
# ─────────────────────────────────────────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
    ],
    key=['N_OUT'],
)
@triton.jit
def maxpool3d_k2_kernel(
    X_ptr, Y_ptr,
    BATCH, C,
    IN_D, IN_H, IN_W,
    OUT_D, OUT_H, OUT_W,
    # strides for input (contiguous NCDHW)
    sx_n, sx_c, sx_d, sx_h, sx_w,
    # strides for output
    sy_n, sy_c, sy_d, sy_h, sy_w,
    N_OUT,
    XBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offs < N_OUT

    # Decode output index: (batch, c, od, oh, ow)
    tmp = offs
    ow = tmp % OUT_W
    tmp = tmp // OUT_W
    oh = tmp % OUT_H
    tmp = tmp // OUT_H
    od = tmp % OUT_D
    tmp = tmp // OUT_D
    c = tmp % C
    n = tmp // C

    # Pool over 2x2x2
    base = n * sx_n + c * sx_c
    id0 = od * 2
    ih0 = oh * 2
    iw0 = ow * 2

    v = tl.full((XBLOCK,), float('-inf'), dtype=tl.float32)
    for dd in range(2):
        for dh in range(2):
            for dw in range(2):
                id_ = id0 + dd
                ih_ = ih0 + dh
                iw_ = iw0 + dw
                valid = mask & (id_ < IN_D) & (ih_ < IN_H) & (iw_ < IN_W)
                ptrs = X_ptr + base + id_ * sx_d + ih_ * sx_h + iw_ * sx_w
                val = tl.load(ptrs, mask=valid, other=float('-inf'))
                v = tl.maximum(v, val)

    out_ptrs = Y_ptr + n * sy_n + c * sy_c + od * sy_d + oh * sy_h + ow * sy_w
    tl.store(out_ptrs, v, mask=mask)


def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: (128, 3, 16, 32, 32)
    # weight: (16, 3, 3, 3, 3)
    # bias: (16,)

    BATCH, IN_C, IN_D, IN_H, IN_W = x.shape
    OUT_C, _, KD, KH, KW = p_conv_weight.shape
    STRIDE_D, STRIDE_H, STRIDE_W = 1, 1, 1
    PAD_D, PAD_H, PAD_W = 0, 0, 0

    OUT_D = (IN_D + 2 * PAD_D - KD) // STRIDE_D + 1  # 14
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1  # 30
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1  # 30

    # Allocate conv output
    conv_out = torch.empty((BATCH, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_D * OUT_H * OUT_W

    # Launch conv3d kernel
    BLOCK_N = 16
    grid_conv = (
        triton.cdiv(M_TOTAL, 1),  # will be overridden by autotune BLOCK_M
        triton.cdiv(OUT_C, BLOCK_N),
    )

    # We need to compute grid based on BLOCK_M from autotune
    # Use a lambda grid
    def grid_fn(meta):
        return (triton.cdiv(M_TOTAL, meta['BLOCK_M']), triton.cdiv(OUT_C, meta['BLOCK_N']))

    conv3d_kernel[grid_fn](
        x, p_conv_weight, p_conv_bias, conv_out,
        BATCH, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        M_TOTAL,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3), x.stride(4),
        p_conv_weight.stride(0), p_conv_weight.stride(1), p_conv_weight.stride(2),
        p_conv_weight.stride(3), p_conv_weight.stride(4),
        conv_out.stride(0), conv_out.stride(1), conv_out.stride(2),
        conv_out.stride(3), conv_out.stride(4),
    )

    # Softmax over channel dim=1
    # Layout: (BATCH, OUT_C, OUT_D, OUT_H, OUT_W)
    # For each (batch, d, h, w) position, reduce over C=16
    # N_SPATIAL = BATCH * OUT_D * OUT_H * OUT_W
    # stride_c = OUT_D * OUT_H * OUT_W (stride between channels in flat spatial index)
    N_SPATIAL = BATCH * OUT_D * OUT_H * OUT_W
    stride_c = OUT_D * OUT_H * OUT_W  # stride between channels

    softmax_out = torch.empty_like(conv_out)

    def grid_softmax(meta):
        return (triton.cdiv(N_SPATIAL, meta['XBLOCK']),)

    softmax_channel_kernel[grid_softmax](
        conv_out, softmax_out,
        OUT_C, N_SPATIAL,
        stride_c,
    )

    # MaxPool3d 1: (128, 16, 14, 30, 30) -> (128, 16, 7, 15, 15)
    P1_D, P1_H, P1_W = 7, 15, 15
    pool1_out = torch.empty((BATCH, OUT_C, P1_D, P1_H, P1_W), device=x.device, dtype=x.dtype)
    N_OUT1 = BATCH * OUT_C * P1_D * P1_H * P1_W

    def grid_pool1(meta):
        return (triton.cdiv(N_OUT1, meta['XBLOCK']),)

    maxpool3d_k2_kernel[grid_pool1](
        softmax_out, pool1_out,
        BATCH, OUT_C,
        OUT_D, OUT_H, OUT_W,
        P1_D, P1_H, P1_W,
        softmax_out.stride(0), softmax_out.stride(1), softmax_out.stride(2),
        softmax_out.stride(3), softmax_out.stride(4),
        pool1_out.stride(0), pool1_out.stride(1), pool1_out.stride(2),
        pool1_out.stride(3), pool1_out.stride(4),
        N_OUT1,
    )

    # MaxPool3d 2: (128, 16, 7, 15, 15) -> (128, 16, 3, 7, 7)
    P2_D, P2_H, P2_W = 3, 7, 7
    pool2_out = torch.empty((BATCH, OUT_C, P2_D, P2_H, P2_W), device=x.device, dtype=x.dtype)
    N_OUT2 = BATCH * OUT_C * P2_D * P2_H * P2_W

    def grid_pool2(meta):
        return (triton.cdiv(N_OUT2, meta['XBLOCK']),)

    maxpool3d_k2_kernel[grid_pool2](
        pool1_out, pool2_out,
        BATCH, OUT_C,
        P1_D, P1_H, P1_W,
        P2_D, P2_H, P2_W,
        pool1_out.stride(0), pool1_out.stride(1), pool1_out.stride(2),
        pool1_out.stride(3), pool1_out.stride(4),
        pool2_out.stride(0), pool2_out.stride(1), pool2_out.stride(2),
        pool2_out.stride(3), pool2_out.stride(4),
        N_OUT2,
    )

    return pool2_out


def get_inputs():
    batch_size = 128
    in_channels = 3
    depth, height, width = 16, 32, 32
    x = torch.rand(batch_size, in_channels, depth, height, width, device='cuda')

    # Initialize weights like nn.Conv3d would
    import math
    out_channels = 16
    kernel_size = 3
    weight = torch.empty(out_channels, in_channels, kernel_size, kernel_size, kernel_size, device='cuda')
    bias = torch.empty(out_channels, device='cuda')
    # kaiming uniform init
    fan_in = in_channels * kernel_size * kernel_size * kernel_size
    std = math.sqrt(2.0 / fan_in)
    weight.uniform_(-std * math.sqrt(3), std * math.sqrt(3))
    bias.uniform_(-1.0 / math.sqrt(fan_in), 1.0 / math.sqrt(fan_in))

    return [weight, bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify against PyTorch reference
        import torch.nn as nn
        weight, bias, x = inputs
        conv = nn.Conv3d(3, 16, 3, bias=True).cuda()
        conv.weight.data.copy_(weight)
        conv.bias.data.copy_(bias)
        pool = nn.MaxPool3d(2)

        with torch.no_grad():
            ref = conv(x)
            ref = torch.softmax(ref, dim=1)
            ref = pool(ref)
            ref = pool(ref)

        print(f"Output shape: {output.shape}")
        print(f"Reference shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        if max_diff < 1e-3:
            print("PASSED: Results match within tolerance.")
        else:
            print("FAILED: Results do not match!")
        print("Kernel executed successfully.")