import torch
import triton
import triton.language as tl

# ─── Kernel 1: Implicit GEMM Conv2d ───────────────────────────────────────────
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=2),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # X strides: NHWC layout
    s_xn: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr, s_xc: tl.constexpr,
    # W strides: (OUT_C, IN_C, KH, KW) -> NHWC-like for W: (OUT_C, KH, KW, IN_C)
    s_wo: tl.constexpr, s_wkh: tl.constexpr, s_wkw: tl.constexpr, s_wi: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_tile in range(BLOCK_K_COUNT):
        offs_k = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)
        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kh_idx = (offs_k // IN_C) % KH
        kw_idx = offs_k // (IN_C * KH)

        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_ih & valid_iw

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        valid_n = offs_n[None, :] < OUT_C
        mask_w = valid_k[:, None] & valid_n
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Store output: Y is (BATCH, OUT_C, OUT_H, OUT_W) in NCHW
    # stride: (OUT_C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1)
    s_yn = OUT_C * OUT_H * OUT_W
    s_yc = OUT_H * OUT_W
    s_yh = OUT_W
    s_yw = 1

    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)
    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# ─── Kernel 2: AvgPool2d + Sigmoid + Sum (per batch item) ────────────────────
# Each program handles one (batch, channel) pair, reducing over spatial dims
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},  num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},  num_stages=3, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128},  num_stages=4, num_warps=2),
    ],
    key=['OUT_H', 'OUT_W', 'POOL_H', 'POOL_W', 'POOL_SH', 'POOL_SW', 'N_CHANNELS'],
)
@triton.jit
def avgpool_sigmoid_sum_kernel(
    BATCH: tl.constexpr,
    N_CHANNELS: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    POOL_H: tl.constexpr,
    POOL_W: tl.constexpr,
    POOL_SH: tl.constexpr,
    POOL_SW: tl.constexpr,
    POOL_OUT_H: tl.constexpr,
    POOL_OUT_W: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one batch item
    batch_id = tl.program_id(0)

    # Total elements in pooled output per batch: N_CHANNELS * POOL_OUT_H * POOL_OUT_W
    total_pool = N_CHANNELS * POOL_OUT_H * POOL_OUT_W
    pool_area = POOL_H * POOL_W
    pool_scale = 1.0 / pool_area

    # Strides for conv output (NCHW)
    s_xn = N_CHANNELS * OUT_H * OUT_W
    s_xc = OUT_H * OUT_W
    s_xh = OUT_W
    s_xw = 1

    batch_base = batch_id * s_xn

    acc_sum = tl.zeros((), dtype=tl.float32)

    for tile_start in range(0, total_pool, BLOCK_SIZE):
        offs = tile_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < total_pool

        # Decode offs -> (channel, pool_h, pool_w)
        pw = offs % POOL_OUT_W
        tmp2 = offs // POOL_OUT_W
        ph = tmp2 % POOL_OUT_H
        ch = tmp2 // POOL_OUT_H

        # Compute avg pool: sum over POOL_H x POOL_W window
        pool_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for ki in range(POOL_H):
            for kj in range(POOL_W):
                ih = ph * POOL_SH + ki
                iw = pw * POOL_SW + kj
                valid = mask & (ih < OUT_H) & (iw < OUT_W) & (ch < N_CHANNELS)
                x_ptrs = X_ptr + batch_base + ch * s_xc + ih * s_xh + iw * s_xw
                val = tl.load(x_ptrs, mask=valid, other=0.0)
                pool_acc = pool_acc + val

        pool_out = pool_acc * pool_scale

        # Sigmoid
        sig_out = 1.0 / (1.0 + tl.exp(-pool_out))

        # Accumulate sum (mask out-of-bounds)
        sig_out = tl.where(mask, sig_out, 0.0)
        acc_sum += tl.sum(sig_out, axis=0)

    tl.store(Y_ptr + batch_id, acc_sum)


def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: (128, 8, 384, 384) NCHW
    # weight: (64, 8, 3, 3)
    # bias: (64,)
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = p_conv_weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1

    POOL_H, POOL_W = 4, 4
    POOL_SH, POOL_SW = 4, 4
    POOL_OUT_H = (OUT_H - POOL_H) // POOL_SH + 1  # 95
    POOL_OUT_W = (OUT_W - POOL_W) // POOL_SW + 1  # 95

    # Convert x to NHWC for coalesced access in conv kernel
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # Convert weight to (OUT_C, KH, KW, IN_C) for coalesced access
    w_khwc = p_conv_weight.permute(0, 2, 3, 1).contiguous()

    # Allocate conv output (NCHW)
    conv_out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # NHWC strides for x_nhwc: (H*W*C, W*C, C, 1)
    s_xn = IN_H * IN_W * IN_C
    s_xh = IN_W * IN_C
    s_xw = IN_C
    s_xc = 1

    # (OUT_C, KH, KW, IN_C) strides for w_khwc
    s_wo = KH * KW * IN_C
    s_wkh = KW * IN_C
    s_wkw = IN_C
    s_wi = 1

    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_K = 16

    grid_conv = (
        (M_TOTAL + BLOCK_M - 1) // BLOCK_M,
        (OUT_C + BLOCK_N - 1) // BLOCK_N,
    )

    conv2d_implicit_gemm_kernel[grid_conv](
        x_nhwc, w_khwc, p_conv_bias, conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xh, s_xw, s_xc,
        s_wo, s_wkh, s_wkw, s_wi,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )

    # Allocate output
    result = torch.empty((BATCH,), device=x.device, dtype=x.dtype)

    grid_pool = (BATCH,)
    avgpool_sigmoid_sum_kernel[grid_pool](
        conv_out, result,
        BATCH, OUT_C, OUT_H, OUT_W,
        POOL_H, POOL_W, POOL_SH, POOL_SW,
        POOL_OUT_H, POOL_OUT_W,
    )

    return result


def get_inputs():
    p_conv_weight = torch.randn(64, 8, 3, 3, device='cuda')
    p_conv_bias = torch.randn(64, device='cuda')
    x = torch.rand(128, 8, 384, 384, device='cuda')
    return [p_conv_weight, p_conv_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F
        p_conv_weight, p_conv_bias, x = inputs
        ref_conv = F.conv2d(x, p_conv_weight, p_conv_bias, stride=1, padding=0)
        ref_pool = F.avg_pool2d(ref_conv, kernel_size=4, stride=4)
        ref_sig = torch.sigmoid(ref_pool)
        ref_sum = torch.sum(ref_sig, dim=[1, 2, 3])

        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - ref_sum).abs().max().item():.4e}")
        assert torch.allclose(output, ref_sum, rtol=1e-3, atol=1e-2), \
            f"Results don't match! max_diff={(output - ref_sum).abs().max().item():.4e}"
        print("Kernel executed successfully.")