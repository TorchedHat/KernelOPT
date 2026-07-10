import torch
import triton
import triton.language as tl
import triton.language.extra.libdevice as libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 64}, num_warps=8, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose2d_fused_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N_out, K,
    s_xn, s_xc, s_xh, s_xw,
    s_wic, s_woc, s_wkh, s_wkw,
    s_yn, s_yc, s_yh, s_yw,
    add_value, scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: M = N * OUT_H * OUT_W
    batch = offs_m // (OUT_H * OUT_W)
    rem = offs_m % (OUT_H * OUT_W)
    out_h = rem // OUT_W
    out_w = rem % OUT_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K = IN_C * KH * KW
    # For ConvTranspose2d: output[n, oc, oh, ow] += input[n, ic, ih, iw] * weight[ic, oc, kh, kw]
    # where ih = (oh + PAD_H - kh) and must be divisible by STRIDE_H, and ih/STRIDE_H in [0, IN_H)
    # Similarly for iw

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K: ic, kh, kw
        ic = offs_k // (KH * KW)
        kh = (offs_k % (KH * KW)) // KW
        kw = offs_k % KW

        # For each (oh, kh): ih_num = oh + PAD_H - kh, must be divisible by STRIDE_H
        # ih = ih_num / STRIDE_H, must be in [0, IN_H)
        ih_num = out_h[:, None] + PAD_H - kh[None, :]
        iw_num = out_w[:, None] + PAD_W - kw[None, :]

        ih_valid_stride = (ih_num % STRIDE_H) == 0
        iw_valid_stride = (iw_num % STRIDE_W) == 0

        ih = ih_num // STRIDE_H
        iw = iw_num // STRIDE_W

        ih_valid = (ih >= 0) & (ih < IN_H) & ih_valid_stride
        iw_valid = (iw >= 0) & (iw < IN_W) & iw_valid_stride
        k_valid = offs_k < K
        m_valid = offs_m < M

        valid_x = ih_valid & iw_valid & k_valid & m_valid[:, None] & (batch[:, None] < N)

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: shape [IN_C, OUT_C, KH, KW], strides s_wic, s_woc, s_wkh, s_wkw
        # We need w[ic, oc, kh, kw] -> tile [BLOCK_K, BLOCK_N]
        n_valid = offs_n < N_out
        valid_w = k_valid[:, None] & n_valid[None, :]

        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < N_out, other=0.0)
    acc = acc + bias[None, :]

    # Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    # softplus(x) = log(1 + exp(x))
    sp = libdevice.log1p(tl.exp(tl.minimum(acc, 20.0)))
    # For large x, softplus(x) ≈ x, so tanh(sp) ≈ 1
    mish_out = acc * libdevice.tanh(sp)

    # Add value
    mish_out = mish_out + add_value

    # Hardtanh: clamp to [-1, 1]
    mish_out = tl.minimum(tl.maximum(mish_out, -1.0), 1.0)

    # Scale
    mish_out = mish_out * scale

    # Store
    m_valid = offs_m[:, None] < M
    n_valid = offs_n[None, :] < N_out

    y_ptrs = (Y_ptr
    tl.store(y_ptrs, mish_out.to(Y_ptr.dtype.element_ty), mask=m_valid & n_valid)


def kernel_function(weight, bias, x):
    # x: (N, IN_C, IN_H, IN_W)
    # weight: (IN_C, OUT_C, KH, KW) for ConvTranspose2d
    N, IN_C, IN_H, IN_W = x.shape
    IN_C_w, OUT_C, KH, KW = weight.shape

    STRIDE_H = 2
    STRIDE_W = 2
    PAD_H = 1
    PAD_W = 1
    OUTPUT_PADDING_H = 1
    OUTPUT_PADDING_W = 1

    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + KH + OUTPUT_PADDING_H
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + KW + OUTPUT_PADDING_W

    add_value = 0.5
    scale = 2.0

    M = N * OUT_H * OUT_W
    N_out = OUT_C
    K = IN_C * KH * KW

    y = torch.empty((N, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    s_xn, s_xc, s_xh, s_xw = x.stride()
    s_wic, s_woc, s_wkh, s_wkw = weight.stride()
    s_yn, s_yc, s_yh, s_yw = y.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(N_out, meta['BLOCK_N']))

    conv_transpose2d_fused_kernel[grid](
        x, weight, bias, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M, N_out, K,
        s_xn, s_xc, s_xh, s_xw,
        s_wic, s_woc, s_wkh, s_wkw,
        s_yn, s_yc, s_yh, s_yw,
        add_value, scale,
    )

    return y


def get_inputs():
    weight = torch.randn(64, 64, 3, 3)
    bias = torch.randn(64)
    x = torch.rand(128, 64, 128, 128)
    return [weight, bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    inputs_cuda = [t.cuda() for t in inputs]

    # Verify correctness
    if not args.profile_only:
        weight_cuda, bias_cuda, x_cuda = inputs_cuda

        # Reference
        import torch.nn.functional as F
        ref_out = F.conv_transpose2d(x_cuda, weight_cuda, bias_cuda, stride=2, padding=1, output_padding=1)
        sp = torch.log1p(torch.exp(ref_out))
        ref_out = ref_out * torch.tanh(sp)
        ref_out = ref_out + 0.5
        ref_out = torch.clamp(ref_out, -1.0, 1.0)
        ref_out = ref_out * 2.0

        triton_out = kernel_function(weight_cuda, bias_cuda, x_cuda)

        max_diff = (triton_out - ref_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs_cuda)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")