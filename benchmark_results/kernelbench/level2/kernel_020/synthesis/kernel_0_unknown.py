import torch
import triton
import triton.language as tl

# ConvTranspose3d: for each input position (n, ci, d, h, w), scatter to output positions
# Output shape: (N, Co, OD, OH, OW) where OD = (D-1)*S - 2*P + KD + output_padding
# We implement as: output[n, co, d*S + kd - P, h*S + kh - P, w*S + kw - P] += x[n,ci,d,h,w] * weight[ci,co,kd,kh,kw]
# Equivalently: treat as GEMM where we iterate over input positions

# Strategy: Use implicit GEMM for ConvTranspose3d
# M = N * OD * OH * OW (output spatial)
# N_out = Co
# K = Ci * KD * KH * KW
# For each output position (n, od, oh, ow) and output channel co:
#   out[n,co,od,oh,ow] = sum over (ci,kd,kh,kw) where (od+P-kd)%S==0 etc.
#                         of x[n,ci,(od+P-kd)/S,(oh+P-kh)/S,(ow+P-kw)/S] * weight[ci,co,kd,kh,kw]

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N_OUT', 'K'],
)
@triton.jit
def conv_transpose3d_epilogue_kernel(
    X_ptr, W_ptr, B_ptr, BIAS_ptr, OUT_ptr,
    N, Ci, Co,
    ID, IH, IW,
    OD, OH, OW,
    KD, KH, KW,
    stride_d, stride_h, stride_w,
    pad_d, pad_h, pad_w,
    # strides for X: (N, Ci, ID, IH, IW)
    x_sn, x_sc, x_sd, x_sh, x_sw,
    # strides for W: (Ci, Co, KD, KH, KW)
    w_sci, w_sco, w_skd, w_skh, w_skw,
    M, N_OUT, K,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode output positions from offs_m
    # M = N * OD * OH * OW
    OHW = OH * OW
    ODH = OD * OH
    ODHW = OD * OH * OW

    out_w = offs_m % OW
    out_h = (offs_m // OW) % OH
    out_d = (offs_m // OHW) % OD
    batch = offs_m // ODHW

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = Ci * KD * KH * KW
    KDHW = KD * KH * KW
    KHW = KH * KW

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K: (ci, kd, kh, kw)
        ci = offs_k // KDHW
        rem = offs_k % KDHW
        kd = rem // KHW
        rem2 = rem % KHW
        kh = rem2 // KW
        kw = rem2 % KW

        # For ConvTranspose3d: input position from output position
        # out_d = in_d * stride_d - pad_d + kd
        # => in_d = (out_d + pad_d - kd) / stride_d (must be integer and in range)
        # Compute numerator for each (out_d, kd) pair
        num_d = out_d[:, None] + pad_d - kd[None, :]
        num_h = out_h[:, None] + pad_h - kh[None, :]
        num_w = out_w[:, None] + pad_w - kw[None, :]

        valid_d = (num_d % stride_d == 0)
        valid_h = (num_h % stride_h == 0)
        valid_w = (num_w % stride_w == 0)

        in_d = num_d // stride_d
        in_h = num_h // stride_h
        in_w = num_w // stride_w

        valid_d = valid_d & (in_d >= 0) & (in_d < ID)
        valid_h = valid_h & (in_h >= 0) & (in_h < IH)
        valid_w = valid_w & (in_w >= 0) & (in_w < IW)
        valid_k = offs_k[None, :] < K
        valid_batch = batch[:, None] < N
        valid_ci = ci[None, :] < Ci

        valid_x = valid_d & valid_h & valid_w & valid_k & valid_batch & valid_ci

        x_ptrs = (
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: W[ci, co, kd, kh, kw]
        valid_w_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N_OUT)
        w_ptrs = (
        )
        w_tile = tl.load(w_ptrs, mask=valid_w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add conv bias: B_ptr[co]
    co_offs = offs_n
    b_vals = tl.load(B_ptr + co_offs, mask=co_offs < N_OUT, other=0.0)
    acc = acc + b_vals[None, :]

    # Epilogue: original_x = acc (before bias addition)
    # Wait - we need original_x = conv output (before bias)
    # Let's store original_x separately
    # Actually: original_x = conv_out (no bias)
    # Then: result = (conv_out + bias + conv_out) * conv_out + conv_out
    #              = (2*conv_out + bias) * conv_out + conv_out

    # conv_out is acc - b_vals (we added bias above, let's redo)
    # Let conv_out = acc_no_bias
    # acc currently = conv_out + conv_bias
    # So conv_out = acc - b_vals
    conv_out = acc - b_vals[None, :]  # original_x

    # bias parameter: BIAS_ptr[co, 1, 1, 1]
    bias_vals = tl.load(BIAS_ptr + co_offs, mask=co_offs < N_OUT, other=0.0)

    # x = conv_out + bias_param
    # x = x + original_x = conv_out + bias_param + conv_out = 2*conv_out + bias_param
    # x = x * original_x = (2*conv_out + bias_param) * conv_out
    # x = x + original_x = (2*conv_out + bias_param) * conv_out + conv_out

    z = (2.0 * conv_out + bias_vals[None, :]) * conv_out + conv_out

    # Store output
    # Output layout: (N, Co, OD, OH, OW)
    # Linear index: batch * Co*OD*OH*OW + co * OD*OH*OW + out_d * OH*OW + out_h * OW + out_w
    out_idx = (
        batch[:, None] * (Co * ODHW)
    )

    valid_out = (offs_m[:, None] < M) & (offs_n[None, :] < N_OUT)
    tl.store(OUT_ptr + out_idx, z.to(tl.float32), mask=valid_out)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x):
    # x: (N, Ci, ID, IH, IW)
    # weight: (Ci, Co, KD, KH, KW)
    # bias: (Co,)
    # p_bias: (Co, 1, 1, 1)

    N, Ci, ID, IH, IW = x.shape
    Ci_w, Co, KD, KH, KW = p_conv_transpose_weight.shape

    stride_d = stride_h = stride_w = 2
    pad_d = pad_h = pad_w = 1
    output_padding = 1

    OD = (ID - 1) * stride_d - 2 * pad_d + KD + output_padding
    OH = (IH - 1) * stride_h - 2 * pad_h + KH + output_padding
    OW = (IW - 1) * stride_w - 2 * pad_w + KW + output_padding

    M = N * OD * OH * OW
    N_OUT = Co
    K = Ci * KD * KH * KW

    out = torch.empty(N, Co, OD, OH, OW, device=x.device, dtype=x.dtype)

    # Strides for x
    x_sn, x_sc, x_sd, x_sh, x_sw = x.stride()
    # Strides for weight (Ci, Co, KD, KH, KW)
    w_sci, w_sco, w_skd, w_skh, w_skw = p_conv_transpose_weight.stride()

    # p_bias shape: (Co, 1, 1, 1) -> flatten to (Co,)
    bias_flat = p_bias.view(Co)

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(N_OUT, meta['BLOCK_N']))

    conv_transpose3d_epilogue_kernel[grid](
        x, p_conv_transpose_weight, p_conv_transpose_bias, bias_flat, out,
        N, Ci, Co,
        ID, IH, IW,
        OD, OH, OW,
        KD, KH, KW,
        stride_d, stride_h, stride_w,
        pad_d, pad_h, pad_w,
        x_sn, x_sc, x_sd, x_sh, x_sw,
        w_sci, w_sco, w_skd, w_skh, w_skw,
        M, N_OUT, K,
    )

    return out


def get_inputs():
    p_conv_transpose_weight = torch.randn(32, 64, 3, 3, 3, device='cuda')
    p_conv_transpose_bias = torch.randn(64, device='cuda')
    p_bias = torch.randn(64, 1, 1, 1, device='cuda')
    x = torch.rand(16, 32, 16, 32, 32, device='cuda')
    return [p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x]


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

        p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x = inputs

        conv = nn.ConvTranspose3d(32, 64, 3, stride=2, padding=1, output_padding=1).cuda()
        with torch.no_grad():
            conv.weight.copy_(p_conv_transpose_weight)
            conv.bias.copy_(p_conv_transpose_bias)

        with torch.no_grad():
            ref = conv(x)
            original_x = ref.clone().detach()
            ref = ref + p_bias
            ref = ref + original_x
            ref = ref * original_x
            ref = ref + original_x

        print(f"Output shape: {output.shape}")
        print(f"Max diff: {(output - ref).abs().max().item()}")
        print(f"Match: {torch.allclose(output, ref, atol=1e-3)}")
        print("Kernel executed successfully.")