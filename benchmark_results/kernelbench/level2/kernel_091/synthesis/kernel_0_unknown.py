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
    key=['M', 'N', 'K'],
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
    key=['M', 'N', 'K'],
)
@triton.jit
def conv_transpose2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    N, IC, IH, IW,
    OC, OH, OW,
    KH, KW,
    stride_h, stride_w,
    pad_h, pad_w,
    M, K,
    s_xn, s_xc, s_xh, s_xw,
    s_wic, s_woc, s_wkh, s_wkw,
    s_yn, s_yc, s_yh, s_yw,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index: m = n * OH * OW + oh * OW + ow
    m_idx = offs_m
    ow = m_idx % OW
    oh = (m_idx // OW) % OH
    batch = m_idx // (OW * OH)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K index: k = ic * KH * KW + kh * KW + kw
        kw = offs_k % KW
        kh = (offs_k // KW) % KH
        ic = offs_k // (KW * KH)

        # For ConvTranspose2d: output[n, oc, oh, ow] += input[n, ic, ih, iw] * weight[ic, oc, kh, kw]
        # where oh = ih * stride_h - pad_h + kh => ih = (oh + pad_h - kh) / stride_h
        # ih must be integer and in [0, IH)
        oh_plus_pad_h = oh[:, None] + pad_h - kh[None, :]  # [BLOCK_M, BLOCK_K]
        ow_plus_pad_w = ow[:, None] + pad_w - kw[None, :]  # [BLOCK_M, BLOCK_K]

        ih = oh_plus_pad_h // stride_h
        iw = ow_plus_pad_w // stride_w

        valid_h = (oh_plus_pad_h >= 0) & (oh_plus_pad_h % stride_h == 0) & (ih >= 0) & (ih < IH)
        valid_w = (ow_plus_pad_w >= 0) & (ow_plus_pad_w % stride_w == 0) & (iw >= 0) & (iw < IW)
        valid_k = offs_k[None, :] < K
        valid_m = (offs_m[:, None] < M) & (batch[:, None] < N)
        valid_x = valid_h & valid_w & valid_k & valid_m

        x_ptrs = X_ptr + batch[:, None] * s_xn + ic[None, :] * s_xc + ih * s_xh + iw * s_xw
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: [IC, OC, KH, KW], index [ic, oc, kh, kw]
        # For tile: [BLOCK_K, BLOCK_N]
        w_ptrs = W_ptr + ic[:, None] * s_wic + offs_n[None, :] * s_woc + kh[:, None] * s_wkh + kw[:, None] * s_wkw
        valid_w_tile = (offs_k[:, None] < K) & (offs_n[None, :] < OC)
        w_tile = tl.load(w_ptrs, mask=valid_w_tile, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OC, other=0.0)
    acc = acc + bias[None, :]

    # Store
    out_m_mask = offs_m[:, None] < M
    out_n_mask = offs_n[None, :] < OC
    out_mask = out_m_mask & out_n_mask & (batch[:, None] < N)

    y_ptrs = Y_ptr + batch[:, None] * s_yn + offs_n[None, :] * s_yc + oh[:, None] * s_yh + ow[:, None] * s_yw
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8),
    ],
    key=['C'],
)
@triton.jit
def softmax_add_mul_sigmoid_kernel(
    Y_ptr, Out_ptr, Bias_ptr,
    N, C, HW,
    scaling_factor,
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles one (n, hw) pair, reducing over C
    pid = tl.program_id(0)
    n = pid // HW
    hw = pid % HW

    # Softmax over C dimension
    # First pass: find max
    row_max = tl.full((), float('-inf'), dtype=tl.float32)
    for c_start in range(0, C, BLOCK_SIZE):
        offs_c = c_start + tl.arange(0, BLOCK_SIZE)
        mask = offs_c < C
        # Y layout: [N, C, H, W] -> linear index: n*C*HW + c*HW + hw
        ptrs = Y_ptr + n * C * HW + offs_c * HW + hw
        vals = tl.load(ptrs, mask=mask, other=float('-inf'))
        row_max = tl.maximum(row_max, tl.max(vals, axis=0))

    # Second pass: compute sum of exp
    row_sum = tl.zeros((), dtype=tl.float32)
    for c_start in range(0, C, BLOCK_SIZE):
        offs_c = c_start + tl.arange(0, BLOCK_SIZE)
        mask = offs_c < C
        ptrs = Y_ptr + n * C * HW + offs_c * HW + hw
        vals = tl.load(ptrs, mask=mask, other=float('-inf'))
        row_sum += tl.sum(tl.exp(vals - row_max), axis=0)

    # Third pass: normalize, add bias, mul, sigmoid, store
    for c_start in range(0, C, BLOCK_SIZE):
        offs_c = c_start + tl.arange(0, BLOCK_SIZE)
        mask = offs_c < C
        ptrs = Y_ptr + n * C * HW + offs_c * HW + hw
        vals = tl.load(ptrs, mask=mask, other=float('-inf'))
        softmax_val = tl.exp(vals - row_max) / row_sum
        # Add bias: bias shape [C, 1, 1] -> bias[c]
        bias = tl.load(Bias_ptr + offs_c, mask=mask, other=0.0)
        result = (softmax_val + bias) * scaling_factor
        # Sigmoid
        result = 1.0 / (1.0 + tl.exp(-result))
        out_ptrs = Out_ptr + n * C * HW + offs_c * HW + hw
        tl.store(out_ptrs, result.to(tl.float32), mask=mask)


def kernel_function(p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x):
    # p_conv_transpose_weight: [IC, OC, KH, KW] = [64, 128, 4, 4]
    # p_conv_transpose_bias: [OC] = [128]
    # p_bias: [OC, 1, 1] = [128, 1, 1]
    # x: [N, IC, IH, IW] = [128, 64, 64, 64]

    IC, OC, KH, KW = p_conv_transpose_weight.shape
    N, IC_x, IH, IW = x.shape
    assert IC == IC_x

    stride_h = 2
    stride_w = 2
    pad_h = 1
    pad_w = 1
    output_padding_h = 1
    output_padding_w = 1

    OH = (IH - 1) * stride_h - 2 * pad_h + KH + output_padding_h
    OW = (IW - 1) * stride_w - 2 * pad_w + KW + output_padding_w

    # Allocate output for conv_transpose
    conv_out = torch.empty((N, OC, OH, OW), device=x.device, dtype=x.dtype)

    M = N * OH * OW
    K = IC * KH * KW

    # Strides for x: [N, IC, IH, IW]
    s_xn, s_xc, s_xh, s_xw = x.stride()
    # Strides for weight: [IC, OC, KH, KW]
    s_wic, s_woc, s_wkh, s_wkw = p_conv_transpose_weight.stride()
    # Strides for conv_out: [N, OC, OH, OW]
    s_yn, s_yc, s_yh, s_yw = conv_out.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(OC, meta['BLOCK_N']))

    conv_transpose2d_kernel[grid](
        x, p_conv_transpose_weight, p_conv_transpose_bias, conv_out,
        N, IC, IH, IW,
        OC, OH, OW,
        KH, KW,
        stride_h, stride_w,
        pad_h, pad_w,
        M, K,
        s_xn, s_xc, s_xh, s_xw,
        s_wic, s_woc, s_wkh, s_wkw,
        s_yn, s_yc, s_yh, s_yw,
    )

    # Allocate output for fused softmax+add+mul+sigmoid
    final_out = torch.empty_like(conv_out)

    # p_bias: [OC, 1, 1] -> flatten to [OC]
    bias_flat = p_bias.view(OC)

    HW = OH * OW
    total_rows = N * HW
    scaling_factor = 2.0

    grid2 = (total_rows,)
    softmax_add_mul_sigmoid_kernel[grid2](
        conv_out, final_out, bias_flat,
        N, OC, HW,
        scaling_factor,
    )

    return final_out


def get_inputs():
    torch.manual_seed(42)
    p_conv_transpose_weight = torch.randn(64, 128, 4, 4, device='cuda', dtype=torch.float32)
    p_conv_transpose_bias = torch.randn(128, device='cuda', dtype=torch.float32)
    p_bias = torch.randn(128, 1, 1, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 64, 64, 64, device='cuda', dtype=torch.float32)
    return [p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_conv_transpose_weight, p_conv_transpose_bias, p_bias, x = inputs

    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F
        # Reference computation
        ref_conv = F.conv_transpose2d(
            x, p_conv_transpose_weight, p_conv_transpose_bias,
            stride=2, padding=1, output_padding=1
        )
        ref_softmax = torch.softmax(ref_conv, dim=1)
        ref_add = ref_softmax + p_bias
        ref_mul = ref_add * 2.0
        ref_sigmoid = torch.sigmoid(ref_mul)

        print(f"Output shape: {output.shape}")
        print(f"Reference shape: {ref_sigmoid.shape}")
        max_diff = (output - ref_sigmoid).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref_sigmoid, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")