import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 8,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=8, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'IN_C', 'KD', 'KH', 'KW'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, Y_ptr,
    # Input strides (N, Ci, D, H, W)
    s_xn, s_xc, s_xd, s_xh, s_xw,
    # Weight strides (Ci, Co, KD, KH, KW)
    s_wci, s_wco, s_wkd, s_wkh, s_wkw,
    # Output strides (N, Co, OD, OH, OW)
    s_yn, s_yco, s_yd, s_yh, s_yw,
    # Dimensions
    BATCH, IN_C, IN_D, IN_H, IN_W,
    OUT_C, OUT_D, OUT_H, OUT_W,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    DIL_D, DIL_H, DIL_W,
    M_TOTAL,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index -> (batch, out_d, out_h, out_w)
    tmp = offs_m
    out_w = tmp % OUT_W
    tmp = tmp // OUT_W
    out_h = tmp % OUT_H
    tmp = tmp // OUT_H
    out_d = tmp % OUT_D
    batch = tmp // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Total K = IN_C * KD * KH * KW
    TOTAL_K = IN_C * KD * KH * KW

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kd, kh, kw)
        tmp_k = offs_k
        kw_idx = tmp_k % KW
        tmp_k = tmp_k // KW
        kh_idx = tmp_k % KH
        tmp_k = tmp_k // KH
        kd_idx = tmp_k % KD
        in_c = tmp_k // KD

        # For transposed conv: out = in * stride - pad + k * dilation
        # So: in_d = (out_d + pad - kd * dilation) / stride (must be integer and in range)
        # We need: out_d = in_d * stride - pad + kd * dilation
        # => in_d * stride = out_d + pad - kd * dilation
        # => in_d = (out_d + pad - kd * dilation) / stride

        # Compute numerator for each (out_d, kd) pair
        num_d = out_d[:, None] + PAD_D - kd_idx[None, :] * DIL_D
        num_h = out_h[:, None] + PAD_H - kh_idx[None, :] * DIL_H
        num_w = out_w[:, None] + PAD_W - kw_idx[None, :] * DIL_W

        # Check divisibility and range
        valid_d = (num_d % STRIDE_D == 0)
        valid_h = (num_h % STRIDE_H == 0)
        valid_w = (num_w % STRIDE_W == 0)

        in_d_idx = num_d // STRIDE_D
        in_h_idx = num_h // STRIDE_H
        in_w_idx = num_w // STRIDE_W

        valid_d = valid_d & (in_d_idx >= 0) & (in_d_idx < IN_D)
        valid_h = valid_h & (in_h_idx >= 0) & (in_h_idx < IN_H)
        valid_w = valid_w & (in_w_idx >= 0) & (in_w_idx < IN_W)

        valid_k = offs_k[None, :] < TOTAL_K
        valid_m = offs_m[:, None] < M_TOTAL
        valid_batch = batch[:, None] < BATCH
        valid_ic = in_c[None, :] < IN_C

        valid_x = valid_d & valid_h & valid_w & valid_k & valid_m & valid_batch & valid_ic

        # Load input tile [BLOCK_M, BLOCK_K]
        x_base = (batch[:, None] * s_xn
        x_tile = tl.load(X_ptr + x_base, mask=valid_x, other=0.0)

        # Load weight tile [BLOCK_K, BLOCK_N]
        # weight shape: (Ci, Co, KD, KH, KW)
        # For transposed conv, weight[ci, co, kd, kh, kw]
        valid_n = offs_n[None, :] < OUT_C
        valid_w_mask = (offs_k[:, None] < TOTAL_K) & valid_n

        w_base = (in_c[:, None] * s_wci
        w_tile = tl.load(W_ptr + w_base, mask=valid_w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Store output
    valid_out_m = offs_m[:, None] < M_TOTAL
    valid_out_n = offs_n[None, :] < OUT_C

    y_base = (batch[:, None] * s_yn

    tl.store(Y_ptr + y_base, acc.to(tl.float32), mask=valid_out_m & valid_out_n)


def kernel_function(weight, x):
    # weight: (in_channels, out_channels, KD, KH, KW) = (32, 64, 3, 3, 3)
    # x: (batch, in_channels, D, H, W) = (16, 32, 16, 32, 32)
    BATCH, IN_C, IN_D, IN_H, IN_W = x.shape
    IN_C_w, OUT_C, KD, KH, KW = weight.shape

    STRIDE_D = STRIDE_H = STRIDE_W = 2
    PAD_D = PAD_H = PAD_W = 1
    DIL_D = DIL_H = DIL_W = 2

    OUT_D = (IN_D - 1) * STRIDE_D - 2 * PAD_D + DIL_D * (KD - 1) + 1
    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + DIL_H * (KH - 1) + 1
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + DIL_W * (KW - 1) + 1

    y = torch.zeros(BATCH, OUT_C, OUT_D, OUT_H, OUT_W, device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_D * OUT_H * OUT_W

    s_xn, s_xc, s_xd, s_xh, s_xw = x.stride()
    s_wci, s_wco, s_wkd, s_wkh, s_wkw = weight.stride()
    s_yn, s_yco, s_yd, s_yh, s_yw = y.stride()

    def grid(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )

    conv_transpose3d_kernel[grid](
        x, weight, y,
        s_xn, s_xc, s_xd, s_xh, s_xw,
        s_wci, s_wco, s_wkd, s_wkh, s_wkw,
        s_yn, s_yco, s_yd, s_yh, s_yw,
        BATCH, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        DIL_D, DIL_H, DIL_W,
        M_TOTAL,
    )

    return y


def get_inputs():
    weight = torch.randn(32, 64, 3, 3, 3, dtype=torch.float32, device='cuda')
    x = torch.randn(16, 32, 16, 32, 32, dtype=torch.float32, device='cuda')
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, x = inputs

    # Verify correctness
    if not args.profile_only:
        import torch.nn as nn
        # Reference: ConvTranspose3d with stride=2, padding=1, dilation=2
        ref_conv = nn.ConvTranspose3d(32, 64, 3, stride=2, padding=1, dilation=2, bias=False)
        # Set weight to match our weight tensor
        with torch.no_grad():
            ref_conv.weight.copy_(weight)
        ref_conv = ref_conv.cuda()
        ref_out = ref_conv(x)

        our_out = kernel_function(weight, x)

        print(f"Output shape: {our_out.shape}")
        print(f"Reference shape: {ref_out.shape}")
        max_diff = (our_out - ref_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        if max_diff < 1e-3:
            print("PASSED: Results match within tolerance")
        else:
            print("FAILED: Results do not match")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")