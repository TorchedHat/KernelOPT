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
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)


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
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv_transpose3d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IN_C, IN_D, IN_H, IN_W,
    OUT_C, OUT_D, OUT_H, OUT_W,
    KD, KH, KW,
    STRIDE_D, STRIDE_H, STRIDE_W,
    PAD_D, PAD_H, PAD_W,
    M_TOTAL, TOTAL_K,
    s_xn, s_xc, s_xd, s_xh, s_xw,
    s_wic, s_woc, s_wkd, s_wkh, s_wkw,
    s_yn, s_yc, s_yd, s_yh, s_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: n, od, oh, ow
    ow = offs_m % OUT_W
    oh = (offs_m // OUT_W) % OUT_H
    od = (offs_m // (OUT_W * OUT_H)) % OUT_D
    n  = offs_m // (OUT_W * OUT_H * OUT_D)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)
        # Decode K: ic, kd, kh, kw
        # K ordering: ic varies fastest, then kw, kh, kd
        ic  = offs_k % IN_C
        kw_ = (offs_k // IN_C) % KW
        kh_ = (offs_k // (IN_C * KW)) % KH
        kd_ = offs_k // (IN_C * KW * KH)

        # For ConvTranspose3d: od = id*stride - pad + kd
        # => id = (od + pad - kd) / stride  (must be integer and in range)
        # id_num = od + pad - kd
        id_num = od[:, None] + PAD_D - kd_[None, :]
        ih_num = oh[:, None] + PAD_H - kh_[None, :]
        iw_num = ow[:, None] + PAD_W - kw_[None, :]

        id_valid = (id_num % STRIDE_D == 0)
        ih_valid = (ih_num % STRIDE_H == 0)
        iw_valid = (iw_num % STRIDE_W == 0)

        id_ = id_num // STRIDE_D
        ih_ = ih_num // STRIDE_H
        iw_ = iw_num // STRIDE_W

        in_bounds = (id_ >= 0) & (id_ < IN_D) & (ih_ >= 0) & (ih_ < IN_H) & (iw_ >= 0) & (iw_ < IN_W)
        k_mask = offs_k[None, :] < TOTAL_K
        m_mask = offs_m[:, None] < M_TOTAL
        n_mask = offs_n[None, :] < OUT_C

        valid_x = id_valid & ih_valid & iw_valid & in_bounds & k_mask & m_mask

        x_ptrs = X_ptr + n[:, None] * s_xn + ic[None, :] * s_xc + id_ * s_xd + ih_ * s_xh + iw_ * s_xw
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: shape [IN_C, OUT_C, KD, KH, KW]
        # index: [ic, oc, kd, kh, kw]
        w_ptrs = W_ptr + ic[:, None] * s_wic + offs_n[None, :] * s_woc + kd_[:, None] * s_wkd + kh_[:, None] * s_wkh + kw_[:, None] * s_wkw
        w_mask = (offs_k[:, None] < TOTAL_K) & n_mask
        w_tile = tl.load(w_ptrs, mask=w_mask, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Store
    m_mask = offs_m[:, None] < M_TOTAL
    n_mask = offs_n[None, :] < OUT_C
    y_ptrs = Y_ptr + n[:, None] * s_yn + offs_n[None, :] * s_yc + od[:, None] * s_yd + oh[:, None] * s_yh + ow[:, None] * s_yw
    tl.store(y_ptrs, acc.to(tl.float32), mask=m_mask & n_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['total_elements'],
)
@triton.jit
def epilogue_kernel(
    Y_ptr, OUT_ptr,
    scale1, scale2,
    bias_ptr,
    N, C, OD, OH, OW,
    PD, PH, PW,
    total_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < total_elements

    # Decode output index: n, c, pd, ph, pw
    pw = offs % PW
    ph = (offs // PW) % PH
    pd = (offs // (PW * PH)) % PD
    c  = (offs // (PW * PH * PD)) % C
    n  = offs // (PW * PH * PD * C)

    # AvgPool3d kernel_size=2: each output pool element averages 8 input elements
    # pool over od in [pd*2, pd*2+1], oh in [ph*2, ph*2+1], ow in [pw*2, pw*2+1]
    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for kd in range(2):
        for kh in range(2):
            for kw in range(2):
                id_ = pd * 2 + kd
                ih_ = ph * 2 + kh
                iw_ = pw * 2 + kw
                in_valid = (id_ < OD) & (ih_ < OH) & (iw_ < OW)
                src_idx = n * (C * OD * OH * OW) + c * (OD * OH * OW) + id_ * (OH * OW) + ih_ * OW + iw_
                val = tl.load(Y_ptr + src_idx, mask=mask & in_valid, other=0.0)
                acc = acc + val

    acc = acc / 8.0
    acc = acc * scale1

    # Add bias: shape [C, 1, 1, 1]
    b = tl.load(bias_ptr + c, mask=mask, other=0.0)
    acc = acc + b

    acc = acc * scale2

    tl.store(OUT_ptr + offs, acc, mask=mask)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias, p_scale1, p_bias, p_scale2, x):
    # x: (N, IN_C, IN_D, IN_H, IN_W)
    # weight: (IN_C, OUT_C, KD, KH, KW) -- PyTorch ConvTranspose3d weight shape is (in_channels, out_channels, kD, kH, kW)
    N, IN_C, IN_D, IN_H, IN_W = x.shape
    IN_C_w, OUT_C, KD, KH, KW = p_conv_transpose_weight.shape
    assert IN_C == IN_C_w

    STRIDE_D = STRIDE_H = STRIDE_W = 2
    PAD_D = PAD_H = PAD_W = 1

    OUT_D = (IN_D - 1) * STRIDE_D - 2 * PAD_D + KD  # = 31
    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + KH  # = 63
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + KW  # = 63

    M_TOTAL = N * OUT_D * OUT_H * OUT_W
    TOTAL_K = IN_C * KD * KH * KW

    # Allocate conv output
    y_conv = torch.empty((N, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=torch.float32)

    # Strides for x
    s_xn, s_xc, s_xd, s_xh, s_xw = x.stride()
    # Strides for weight (IN_C, OUT_C, KD, KH, KW)
    s_wic, s_woc, s_wkd, s_wkh, s_wkw = p_conv_transpose_weight.stride()
    # Strides for y
    s_yn, s_yc, s_yd, s_yh, s_yw = y_conv.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))

    conv_transpose3d_kernel[grid](
        x, p_conv_transpose_weight, p_conv_transpose_bias, y_conv,
        N, IN_C, IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KD, KH, KW,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        M_TOTAL, TOTAL_K,
        s_xn, s_xc, s_xd, s_xh, s_xw,
        s_wic, s_woc, s_wkd, s_wkh, s_wkw,
        s_yn, s_yc, s_yd, s_yh, s_yw,
    )

    # Epilogue: scale1 * avg_pool(y_conv) + bias + scale2
    PD = OUT_D // 2  # 15
    PH = OUT_H // 2  # 31
    PW = OUT_W // 2  # 31

    total_out = N * OUT_C * PD * PH * PW
    out = torch.empty((N, OUT_C, PD, PH, PW), device=x.device, dtype=torch.float32)

    scale1_val = p_scale1.item()
    scale2_val = p_scale2.item()

    grid2 = lambda meta: (cdiv(total_out, meta['BLOCK_SIZE']),)

    epilogue_kernel[grid2](
        y_conv, out,
        scale1_val, scale2_val,
        p_bias.reshape(-1),  # (OUT_C,)
        N, OUT_C, OUT_D, OUT_H, OUT_W,
        PD, PH, PW,
        total_out,
    )

    return out


def get_inputs():
    torch.manual_seed(42)
    p_conv_transpose_weight = torch.randn(3, 16, 3, 3, 3)
    p_conv_transpose_bias = torch.randn(16)
    p_scale1 = torch.tensor(0.5)
    p_bias = torch.randn(16, 1, 1, 1)
    p_scale2 = torch.tensor(1.0)
    x = torch.rand(128, 3, 16, 32, 32)
    return [p_conv_transpose_weight, p_conv_transpose_bias, p_scale1, p_bias, p_scale2, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs_cpu = get_inputs()
    inputs = [t.cuda() for t in inputs_cpu]

    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn as nn

        p_conv_transpose_weight, p_conv_transpose_bias, p_scale1, p_bias, p_scale2, x = inputs

        # Reference computation
        conv_t = nn.ConvTranspose3d(3, 16, 3, stride=2, padding=1).cuda()
        with torch.no_grad():
            conv_t.weight.copy_(p_conv_transpose_weight)
            conv_t.bias.copy_(p_conv_transpose_bias)

        with torch.no_grad():
            ref = conv_t(x)
            ref = ref * p_scale1
            ref = nn.AvgPool3d(kernel_size=2)(ref)
            ref = ref + p_bias
            ref = ref * p_scale2

        print(f"Output shape: {output.shape}, ref shape: {ref.shape}")
        print(f"Max diff: {(output - ref).abs().max().item():.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-3), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")