import torch
import triton
import triton.language as tl

# Kernel 1: Implicit GEMM convolution (Conv2d)
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # Input strides (NHWC layout)
    s_xn: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr, s_xc: tl.constexpr,
    # Weight strides (NHWC-like: [OUT_C, KH, KW, IN_C])
    s_wo: tl.constexpr, s_wkh: tl.constexpr, s_wkw: tl.constexpr, s_wc: tl.constexpr,
    # Output strides (NHWC layout)
    s_yn: tl.constexpr, s_yh: tl.constexpr, s_yw: tl.constexpr, s_yc: tl.constexpr,
    M_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_block in range(BLOCK_K_COUNT):
        k_start = k_block * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kh, kw)
        in_c_idx = offs_k % IN_C
        kh_idx = (offs_k // IN_C) % KH
        kw_idx = offs_k // (IN_C * KH)

        # Input spatial coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            (batch_idx[:, None] < BATCH) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_ptrs = (
            batch_idx[:, None] * s_xn +
            in_c_idx[None, :] * s_xc
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (
            offs_n[None, :] * s_wo +
            kh_idx[:, None] * s_wkh +
            kw_idx[:, None] * s_wkw +
            in_c_idx[:, None] * s_wc
        )
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Store output
    mask_out = (
        (batch_idx[:, None] < BATCH) &
        (out_h_idx[:, None] < OUT_H) &
        (out_w_idx[:, None] < OUT_W) &
        (offs_n[None, :] < OUT_C)
    )

    y_ptrs = (
        batch_idx[:, None] * s_yn +
        out_h_idx[:, None] * s_yh +
        out_w_idx[:, None] * s_yw +
        offs_n[None, :] * s_yc
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


# Kernel 2: Fused avg_pool2d + sigmoid + sum over [C, H, W] per batch
# Each program handles one batch item, iterating over channels
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_C': 1,  'BLOCK_SPATIAL': 256}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_C': 1,  'BLOCK_SPATIAL': 512}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_C': 1,  'BLOCK_SPATIAL': 128}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_C': 1,  'BLOCK_SPATIAL': 1024}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'OUT_C', 'POOL_OH', 'POOL_OW'],
)
@triton.jit
def avgpool_sigmoid_sum_kernel(
    CONV_ptr, OUT_ptr,
    BATCH: tl.constexpr, OUT_C: tl.constexpr,
    CONV_H: tl.constexpr, CONV_W: tl.constexpr,
    POOL_KH: tl.constexpr, POOL_KW: tl.constexpr,
    POOL_SH: tl.constexpr, POOL_SW: tl.constexpr,
    POOL_OH: tl.constexpr, POOL_OW: tl.constexpr,
    # Conv output strides (NHWC)
    s_cn: tl.constexpr, s_ch: tl.constexpr, s_cw: tl.constexpr, s_cc: tl.constexpr,
    BLOCK_C: tl.constexpr, BLOCK_SPATIAL: tl.constexpr,
):
    batch_id = tl.program_id(0)

    total_spatial = POOL_OH * POOL_OW
    pool_area = POOL_KH * POOL_KW
    inv_pool_area = 1.0 / pool_area

    batch_sum = tl.zeros((), dtype=tl.float32)

    for c in range(OUT_C):
        # For each channel, sum over all pooled spatial positions
        c_sum = tl.zeros((), dtype=tl.float32)

        for sp_start in range(0, total_spatial, BLOCK_SPATIAL):
            sp_offs = sp_start + tl.arange(0, BLOCK_SPATIAL)
            sp_mask = sp_offs < total_spatial

            pool_oh = sp_offs // POOL_OW
            pool_ow = sp_offs % POOL_OW

            # For each pooled position, sum over pool window
            pool_sum = tl.zeros((BLOCK_SPATIAL,), dtype=tl.float32)

            for kh in range(POOL_KH):
                for kw in range(POOL_KW):
                    ih = pool_oh * POOL_SH + kh
                    iw = pool_ow * POOL_SW + kw

                    valid = sp_mask & (ih < CONV_H) & (iw < CONV_W)

                    ptrs = (
                    )
                    vals = tl.load(ptrs, mask=valid, other=0.0)
                    pool_sum = pool_sum + vals

            avg_val = pool_sum * inv_pool_area
            # sigmoid
            sig_val = 1.0 / (1.0 + tl.exp(-avg_val))
            c_sum = c_sum + tl.sum(tl.where(sp_mask, sig_val, 0.0), axis=0)

        batch_sum = batch_sum + c_sum

    tl.store(OUT_ptr + batch_id, batch_sum)


def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: [128, 8, 384, 384] NCHW
    # weight: [64, 8, 3, 3]
    # bias: [64]

    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = p_conv_weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1

    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]
    # Convert weight to [OUT_C, KH, KW, IN_C] for coalesced access
    w_nhwc = p_conv_weight.permute(0, 2, 3, 1).contiguous()  # [OUT_C, KH, KW, IN_C]

    # Allocate conv output in NHWC
    conv_out = torch.empty((BATCH, OUT_H, OUT_W, OUT_C), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Strides for NHWC input
    s_xn = x_nhwc.stride(0)
    s_xh = x_nhwc.stride(1)
    s_xw = x_nhwc.stride(2)
    s_xc = x_nhwc.stride(3)

    # Strides for weight [OUT_C, KH, KW, IN_C]
    s_wo = w_nhwc.stride(0)
    s_wkh = w_nhwc.stride(1)
    s_wkw = w_nhwc.stride(2)
    s_wc = w_nhwc.stride(3)

    # Strides for NHWC output
    s_yn = conv_out.stride(0)
    s_yh = conv_out.stride(1)
    s_yw = conv_out.stride(2)
    s_yc = conv_out.stride(3)

    def grid(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )

    conv2d_implicit_gemm_kernel[grid](
        x_nhwc, w_nhwc, p_conv_bias, conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xh, s_xw, s_xc,
        s_wo, s_wkh, s_wkw, s_wc,
        s_yn, s_yh, s_yw, s_yc,
        M_TOTAL,
    )

    # Pool parameters
    POOL_KH, POOL_KW = 4, 4
    POOL_SH, POOL_SW = 4, 4
    POOL_OH = (OUT_H) // POOL_KH  # 382 // 4 = 95
    POOL_OW = (OUT_W) // POOL_KW  # 382 // 4 = 95

    # Output: sum per batch item
    out = torch.empty((BATCH,), device=x.device, dtype=x.dtype)

    # Strides for conv_out (NHWC)
    s_cn = conv_out.stride(0)
    s_ch = conv_out.stride(1)
    s_cw = conv_out.stride(2)
    s_cc = conv_out.stride(3)

    grid2 = (BATCH,)
    avgpool_sigmoid_sum_kernel[grid2](
        conv_out, out,
        BATCH, OUT_C,
        OUT_H, OUT_W,
        POOL_KH, POOL_KW,
        POOL_SH, POOL_SW,
        POOL_OH, POOL_OW,
        s_cn, s_ch, s_cw, s_cc,
    )

    return out


def get_inputs():
    batch_size = 128
    in_channels = 8
    height, width = 384, 384
    x = torch.rand(batch_size, in_channels, height, width, device='cuda')
    weight = torch.rand(64, 8, 3, 3, device='cuda')
    bias = torch.rand(64, device='cuda')
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
        import torch.nn as nn
        p_conv_weight, p_conv_bias, x = inputs

        # Reference computation
        conv = nn.Conv2d(8, 64, 3, bias=True).cuda()
        conv.weight.data = p_conv_weight
        conv.bias.data = p_conv_bias
        avg_pool = nn.AvgPool2d(4)

        with torch.no_grad():
            ref_x = conv(x)
            ref_x = avg_pool(ref_x)
            ref_x = torch.sigmoid(ref_x)
            ref = torch.sum(ref_x, dim=[1, 2, 3])

        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")