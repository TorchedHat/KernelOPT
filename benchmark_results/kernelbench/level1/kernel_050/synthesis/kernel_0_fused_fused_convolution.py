import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'OUT_H', 'OUT_W', 'IN_C', 'OUT_C', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_bias_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # strides for X: NCHW
    s_xn: tl.constexpr, s_xc: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr,
    # strides for W: OIHW
    s_wo: tl.constexpr, s_wi: tl.constexpr, s_wh: tl.constexpr, s_ww: tl.constexpr,
    # strides for Y: NCHW
    s_yn: tl.constexpr, s_yc: tl.constexpr, s_yh: tl.constexpr, s_yw: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W (output spatial positions)
    # N = OUT_C (output channels)
    # K = IN_C * KERNEL_H * KERNEL_W (reduction)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_idx in range(BLOCK_K_COUNT):
        k_off = k_idx * BLOCK_K
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c = offs_k % IN_C
        kw = (offs_k // IN_C) % KERNEL_W
        kh = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_m = (batch[:, None] < BATCH) & (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        valid_k = offs_k[None, :] < TOTAL_K

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid_m & valid_k, other=0.0)

        # Weight: [BLOCK_K, BLOCK_N]
        valid_n = offs_n[None, :] < OUT_C
        valid_k2 = offs_k[:, None] < TOTAL_K

        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=valid_k2 & valid_n, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Store output: Y[batch, out_c, out_h, out_w]
    mask_m = (batch < BATCH) & (out_h < OUT_H) & (out_w < OUT_W)
    mask_n = offs_n < OUT_C

    y_ptrs = (Y_ptr
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_m[:, None] & mask_n[None, :])


def kernel_function(weight, bias, x):
    # x: [N, C, H, W], weight: [OC, IC, KH, KW], bias: [OC]
    N, IC, H, W = x.shape
    OC, IC2, KH, KW = weight.shape
    assert IC == IC2

    STRIDE_H, STRIDE_W = 4, 4
    PAD_H, PAD_W = 2, 2

    OUT_H = (H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (W + 2 * PAD_W - KW) // STRIDE_W + 1

    # Output in NCHW format
    y = torch.empty((N, OC, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M = N * OUT_H * OUT_W

    # Strides for x (NCHW)
    s_xn, s_xc, s_xh, s_xw = x.stride()
    # Strides for weight (OIHW)
    s_wo, s_wi, s_wh, s_ww = weight.stride()
    # Strides for y (NCHW)
    s_yn, s_yc, s_yh, s_yw = y.stride()

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(OC, meta['BLOCK_N']),
        )

    conv2d_bias_kernel[grid](
        x, weight, bias, y,
        N, IC, H, W,
        OC, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xc, s_xh, s_xw,
        s_wo, s_wi, s_wh, s_ww,
        s_yn, s_yc, s_yh, s_yw,
    )

    return y


def get_inputs():
    weight = torch.rand(96, 3, 11, 11, device='cuda', dtype=torch.float32)
    bias = torch.rand(96, device='cuda', dtype=torch.float32)
    x = torch.rand(256, 3, 224, 224, device='cuda', dtype=torch.float32)
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
        # Verify correctness
        import torch.nn.functional as F
        weight, bias, x = inputs
        ref = F.conv2d(x, weight, bias, stride=4, padding=2)
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match.")