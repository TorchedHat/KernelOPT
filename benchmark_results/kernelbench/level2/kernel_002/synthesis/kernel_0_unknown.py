import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N_OUT', 'K_SIZE'],
)
@triton.jit
def conv_transpose2d_epilogue_kernel(
    X_ptr, W_ptr, BIAS_CONV_ptr, BIAS_ADD_ptr, OUT_ptr,
    N, CI, IH, IW,
    CO, OH, OW,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides for X: (N, CI, IH, IW)
    s_xn, s_xci, s_xih, s_xiw,
    # strides for W: (CI, CO, KH, KW)
    s_wci, s_wco, s_wkh, s_wkw,
    # strides for OUT: (N, CO, OH, OW)
    s_on, s_oco, s_ooh, s_oow,
    M, N_OUT, K_SIZE,
    scaling_factor,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, oh, ow)
    ow = offs_m % OW
    oh = (offs_m // OW) % OH
    batch = offs_m // (OW * OH)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, K_SIZE, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K -> (ci, kh, kw)
        kw_idx = offs_k % KW
        kh_idx = (offs_k // KW) % KH
        ci_idx = offs_k // (KW * KH)

        # For ConvTranspose2d:
        # oh = ih * STRIDE_H - PAD_H + kh  =>  ih = (oh + PAD_H - kh) / STRIDE_H
        # Need (oh + PAD_H - kh) % STRIDE_H == 0 and ih in [0, IH)
        ih_num = oh[:, None] + PAD_H - kh_idx[None, :]  # [BLOCK_M, BLOCK_K]
        iw_num = ow[:, None] + PAD_W - kw_idx[None, :]  # [BLOCK_M, BLOCK_K]

        ih_valid_stride = (ih_num % STRIDE_H) == 0
        iw_valid_stride = (iw_num % STRIDE_W) == 0
        ih_idx = ih_num // STRIDE_H
        iw_idx = iw_num // STRIDE_W

        valid_m = (offs_m[:, None] < M)
        valid_k = (offs_k[None, :] < K_SIZE)
        valid_ih = (ih_idx >= 0) & (ih_idx < IH)
        valid_iw = (iw_idx >= 0) & (iw_idx < IW)
        valid_ci = ci_idx[None, :] < CI
        mask_x = valid_m & valid_k & ih_valid_stride & iw_valid_stride & valid_ih & valid_iw & valid_ci

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: [CI, CO, KH, KW], we need [BLOCK_K, BLOCK_N]
        # offs_k -> (ci, kh, kw), offs_n -> co
        valid_n = offs_n[None, :] < N_OUT
        mask_w = valid_k[:, None] & valid_n & valid_ci[:, None]
        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add conv bias
    conv_bias = tl.load(BIAS_CONV_ptr + offs_n, mask=offs_n < N_OUT, other=0.0)
    acc = acc + conv_bias[None, :]

    # Add extra bias (shape [CO, 1, 1] -> per channel)
    add_bias = tl.load(BIAS_ADD_ptr + offs_n, mask=offs_n < N_OUT, other=0.0)
    acc = acc + add_bias[None, :]

    # clamp(x, 0, 1)
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 1.0)

    # * scaling_factor
    acc = acc * scaling_factor

    # clamp(x, 0, 1)
    acc = tl.maximum(acc, 0.0)
    acc = tl.minimum(acc, 1.0)

    # / scaling_factor
    acc = acc / scaling_factor

    # Store
    valid_m_out = offs_m[:, None] < M
    valid_n_out = offs_n[None, :] < N_OUT
    mask_out = valid_m_out & valid_n_out

    out_ptrs = (OUT_ptr
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x):
    # x: [N, CI, IH, IW]
    # weight: [CI, CO, KH, KW]  (ConvTranspose2d stores weight as [in_ch, out_ch, kH, kW])
    N, CI, IH, IW = x.shape
    CI_w, CO, KH, KW = p_conv_transpose_weight.shape

    STRIDE_H = 2
    STRIDE_W = 2
    PAD_H = 1
    PAD_W = 1
    OUTPUT_PAD_H = 1
    OUTPUT_PAD_W = 1

    OH = (IH - 1) * STRIDE_H - 2 * PAD_H + KH + OUTPUT_PAD_H  # = 256
    OW = (IW - 1) * STRIDE_W - 2 * PAD_W + KW + OUTPUT_PAD_W  # = 256

    M = N * OH * OW
    N_OUT = CO
    K_SIZE = CI * KH * KW

    scaling_factor = 2.0

    out = torch.empty((N, CO, OH, OW), device=x.device, dtype=x.dtype)

    # p_bias shape: [64, 1, 1] -> flatten to [64]
    bias_add = p_bias.reshape(CO)

    grid = lambda meta: (
        triton.cdiv(M, meta['BLOCK_M']),
        triton.cdiv(N_OUT, meta['BLOCK_N']),
    )

    conv_transpose2d_epilogue_kernel[grid](
        x, p_conv_transpose_weight, p_conv_transpose_bias, bias_add, out,
        N, CI, IH, IW,
        CO, OH, OW,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        x.stride(0), x.stride(1), x.stride(2), x.stride(3),
        p_conv_transpose_weight.stride(0), p_conv_transpose_weight.stride(1),
        p_conv_transpose_weight.stride(2), p_conv_transpose_weight.stride(3),
        out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        M, N_OUT, K_SIZE,
        scaling_factor,
    )

    return out


def get_inputs():
    p_conv_transpose_weight = torch.randn(64, 64, 3, 3)
    p_conv_transpose_bias = torch.randn(64)
    p_bias = torch.randn(64, 1, 1)
    x = torch.rand(128, 64, 128, 128)
    return [p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]

    output = kernel_function(*inputs_cuda)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify correctness
        import torch.nn as nn

        p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x = inputs
        x_cuda = inputs_cuda[3]
        w_cuda = inputs_cuda[0]
        b_cuda = inputs_cuda[1]
        pb_cuda = inputs_cuda[2]

        # Reference
        conv_t = nn.ConvTranspose2d(64, 64, 3, stride=2, padding=1, output_padding=1).cuda()
        with torch.no_grad():
            conv_t.weight.copy_(w_cuda)
            conv_t.bias.copy_(b_cuda)
        ref = conv_t(x_cuda)
        ref = ref + pb_cuda
        ref = torch.clamp(ref, 0.0, 1.0)
        ref = ref * 2.0
        ref = torch.clamp(ref, 0.0, 1.0)
        ref = ref / 2.0

        max_diff = (output - ref).abs().max().item()
        print(f"Max difference: {max_diff}")
        if max_diff < 1e-3:
            print("PASSED")
        else:
            print("FAILED")