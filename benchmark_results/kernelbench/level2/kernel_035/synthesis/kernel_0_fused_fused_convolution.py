import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

# ─── Kernel 1: Implicit-GEMM Conv2d (NHWC layout internally) ─────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W', 'KH', 'KW'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # strides for X in NHWC: (IN_H*IN_W*IN_C, IN_W*IN_C, IN_C, 1)
    s_xn: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr,
    # strides for W in (OUT_C, KH, KW, IN_C): (KH*KW*IN_C, KW*IN_C, IN_C, 1)
    s_wn: tl.constexpr, s_wh: tl.constexpr, s_ww: tl.constexpr,
    M_TOTAL: tl.constexpr,
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

    TOTAL_K = IN_C * KH * KW
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_idx in range(BLOCK_K_COUNT):
        k_off = k_idx * BLOCK_K
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kh_kw = offs_k // IN_C
        kh = kh_kw // KW
        kw = kh_kw % KW

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = (offs_m < M_TOTAL)[:, None]
        valid_k = (offs_k < TOTAL_K)[None, :]
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_spatial

        x_ptrs = X_ptr + batch[:, None] * s_xn + ih * s_xh + iw * s_xw + in_c[None, :]
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        valid_n = (offs_n < OUT_C)[None, :]
        valid_k2 = (offs_k < TOTAL_K)[:, None]
        mask_w = valid_k2 & valid_n

        w_ptrs = W_ptr + offs_n[None, :] * s_wn + kh[:, None] * s_wh + kw[:, None] * s_ww + in_c[:, None]
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=(offs_n < OUT_C), other=0.0)
    acc = acc + bias[None, :]

    # Store: output in NCHW layout (batch, out_c, out_h, out_w)
    mask_out = (offs_m < M_TOTAL)[:, None] & (offs_n < OUT_C)[None, :]
    out_ptrs = Y_ptr + batch[:, None] * (OUT_C * OUT_H * OUT_W) + \
               offs_n[None, :] * (OUT_H * OUT_W) + \
               out_h[:, None] * OUT_W + out_w[:, None]
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


# ─── Kernel 2: Fused sub + hardswish + maxpool2d + mish ──────────────────────

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_SIZE': 64},  num_stages=3, num_warps=2),
    ],
    key=['N_OUT'],
)
@triton.jit
def fused_sub_hardswish_maxpool_mish_kernel(
    IN_ptr, OUT_ptr,
    BATCH: tl.constexpr,
    C: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    POOL_KH: tl.constexpr,
    POOL_KW: tl.constexpr,
    POOL_STRIDE_H: tl.constexpr,
    POOL_STRIDE_W: tl.constexpr,
    SUBTRACT_VAL: tl.constexpr,
    N_OUT: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_OUT

    # Decode output index -> (batch, c, out_h, out_w)
    out_w = offs % OUT_W
    tmp = offs // OUT_W
    out_h = tmp % OUT_H
    tmp2 = tmp // OUT_H
    c = tmp2 % C
    batch = tmp2 // C

    # MaxPool: find max over pool window in the hardswish output
    pool_max = tl.full((BLOCK_SIZE,), float('-inf'), dtype=tl.float32)

    for ph in range(POOL_KH):
        for pw in range(POOL_KW):
            ih = out_h * POOL_STRIDE_H + ph
            iw = out_w * POOL_STRIDE_W + pw

            valid_spatial = (ih < IN_H) & (iw < IN_W)
            in_idx = batch * (C * IN_H * IN_W) + c * (IN_H * IN_W) + ih * IN_W + iw

            val = tl.load(IN_ptr + in_idx, mask=mask & valid_spatial, other=float('-inf'))

            # sub
            val = val - SUBTRACT_VAL

            # hardswish: x * clamp(x+3, 0, 6) / 6
            val_plus3 = val + 3.0
            clamped = tl.minimum(tl.maximum(val_plus3, 0.0), 6.0)
            hs_val = val * clamped * (1.0 / 6.0)

            pool_max = tl.maximum(pool_max, hs_val)

    # mish: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)) for x <= 20, else x
    x = pool_max
    sp = tl.where(x > 20.0, x, tl.log(1.0 + tl.exp(x)))
    mish_val = x * libdevice.tanh(sp)

    tl.store(OUT_ptr + offs, mish_val, mask=mask)


# ─── Wrapper ──────────────────────────────────────────────────────────────────

def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: (128, 64, 128, 128) NCHW
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = p_conv_weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1

    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)

    # Convert weight to (OUT_C, KH, KW, IN_C) for coalesced access
    w_khwc = p_conv_weight.permute(0, 2, 3, 1).contiguous()  # (OUT_C, KH, KW, IN_C)

    # Allocate conv output in NCHW
    conv_out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W
    s_xn = IN_H * IN_W * IN_C
    s_xh = IN_W * IN_C
    s_xw = IN_C
    s_wn = KH * KW * IN_C
    s_wh = KW * IN_C
    s_ww = IN_C

    def grid(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )

    conv2d_kernel[grid](
        x_nhwc, w_khwc, p_conv_bias, conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xh, s_xw,
        s_wn, s_wh, s_ww,
        M_TOTAL,
    )

    # MaxPool params
    POOL_KH, POOL_KW = 2, 2
    POOL_STRIDE_H, POOL_STRIDE_W = 2, 2
    POOL_OUT_H = (OUT_H) // POOL_STRIDE_H
    POOL_OUT_W = (OUT_W) // POOL_STRIDE_W

    N_OUT = BATCH * OUT_C * POOL_OUT_H * POOL_OUT_W
    final_out = torch.empty((BATCH, OUT_C, POOL_OUT_H, POOL_OUT_W), device=x.device, dtype=x.dtype)

    grid2 = lambda meta: (triton.cdiv(N_OUT, meta['BLOCK_SIZE']),)

    fused_sub_hardswish_maxpool_mish_kernel[grid2](
        conv_out, final_out,
        BATCH, OUT_C,
        OUT_H, OUT_W,
        POOL_OUT_H, POOL_OUT_W,
        POOL_KH, POOL_KW,
        POOL_STRIDE_H, POOL_STRIDE_W,
        0.5,  # SUBTRACT_VAL
        N_OUT,
    )

    return final_out


def get_inputs():
    p_conv_weight = torch.randn(128, 64, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias = torch.randn(128, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 64, 128, 128, device='cuda', dtype=torch.float32)
    return [p_conv_weight, p_conv_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_conv_weight, p_conv_bias, x = inputs

    output = kernel_function(p_conv_weight, p_conv_bias, x)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F

        # Reference computation
        conv_ref = F.conv2d(x, p_conv_weight, p_conv_bias, stride=1, padding=0)
        sub_ref = conv_ref - 0.5
        hs_ref = F.hardswish(sub_ref)
        pool_ref = F.max_pool2d(hs_ref, kernel_size=2, stride=2)
        mish_ref = F.mish(pool_ref)

        print(f"Output shape: {output.shape}")
        print(f"Ref shape:    {mish_ref.shape}")
        max_diff = (output - mish_ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, mish_ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")