import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_H: tl.constexpr, stride_W: tl.constexpr,
    pad_H: tl.constexpr, pad_W: tl.constexpr,
    # Input strides (NCHW)
    s_xn: tl.constexpr, s_xc: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr,
    # Weight strides (OIHW)
    s_wo: tl.constexpr, s_wi: tl.constexpr, s_wkh: tl.constexpr, s_wkw: tl.constexpr,
    # Output strides (NCHW)
    s_yn: tl.constexpr, s_yc: tl.constexpr, s_yh: tl.constexpr, s_yw: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kh, kw)
        kw_idx = offs_k % KW
        tmp_k = offs_k // KW
        kh_idx = tmp_k % KH
        ic_idx = tmp_k // KH

        # Input coordinates
        ih = out_h_idx[:, None] * stride_H + kh_idx[None, :] - pad_H
        iw = out_w_idx[:, None] * stride_W + kw_idx[None, :] - pad_W

        valid_x = (
            (offs_m[:, None] < M_TOTAL) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < TOTAL_K)
        )

        x_ptrs = (X_ptr +
                  batch_idx[:, None] * s_xn +
                  ic_idx[None, :] * s_xc +
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = (offs_k[:, None] < TOTAL_K) & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * s_wo +
                  ic_idx[:, None] * s_wi +
                  kh_idx[:, None] * s_wkh +
                  kw_idx[:, None] * s_wkw)
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=offs_n < OUT_C, other=0.0)
    acc = acc + bias[None, :]

    # Store output
    mask_out = (
        (offs_m[:, None] < M_TOTAL) &
        (offs_n[None, :] < OUT_C)
    )

    y_ptrs = (Y_ptr +
              batch_idx[:, None] * s_yn +
              offs_n[None, :] * s_yc +
              out_h_idx[:, None] * s_yh +
              out_w_idx[:, None] * s_yw)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8),
    ],
    key=['N_TOTAL'],
)
@triton.jit
def fused_mish_bn_kernel(
    X_ptr, BN_MEAN_ptr, BN_VAR_ptr, BN_WEIGHT_ptr, BN_BIAS_ptr,
    OUT_ptr,
    N_TOTAL: tl.constexpr,
    C: tl.constexpr,
    HW: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N_TOTAL

    # Decode channel index from flat index
    # Layout is NCHW: index = n*C*HW + c*HW + hw
    # c = (offs // HW) % C
    c_idx = (offs // HW) % C

    x = tl.load(X_ptr + offs, mask=mask, other=0.0)

    # Mish activation: x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)) with threshold 20
    sp = tl.where(x > 20.0, x, libdevice.log1p(libdevice.exp(x)))
    mish = x * libdevice.tanh(sp)

    # BatchNorm (inference): (x - mean) / sqrt(var + eps) * weight + bias
    mean = tl.load(BN_MEAN_ptr + c_idx, mask=mask, other=0.0)
    var = tl.load(BN_VAR_ptr + c_idx, mask=mask, other=0.0)
    weight = tl.load(BN_WEIGHT_ptr + c_idx, mask=mask, other=0.0)
    bias = tl.load(BN_BIAS_ptr + c_idx, mask=mask, other=0.0)

    inv_std = 1.0 / tl.sqrt(var + eps)
    out = (mish - mean) * inv_std * weight + bias

    tl.store(OUT_ptr + offs, out, mask=mask)


def kernel_function(
    p_conv_weight, p_conv_bias, p_bn_weight, p_bn_bias,
):
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = p_conv_weight.shape
    stride_H, stride_W = 1, 1
    pad_H, pad_W = 0, 0
    OUT_H = (IN_H + 2 * pad_H - KH) // stride_H + 1
    OUT_W = (IN_W + 2 * pad_W - KW) // stride_W + 1

    # Allocate conv output
    conv_out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Input strides (NCHW)
    s_xn, s_xc, s_xh, s_xw = x.stride()
    # Weight strides (OIHW)
    s_wo, s_wi, s_wkh, s_wkw = p_conv_weight.stride()
    # Output strides (NCHW)
    s_yn, s_yc, s_yh, s_yw = conv_out.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    grid_conv = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))

    conv2d_kernel[grid_conv](
        x, p_conv_weight, p_conv_bias, conv_out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_H, stride_W,
        pad_H, pad_W,
        s_xn, s_xc, s_xh, s_xw,
        s_wo, s_wi, s_wkh, s_wkw,
        s_yn, s_yc, s_yh, s_yw,
    )

    # Fused mish + batchnorm
    N_TOTAL = BATCH * OUT_C * OUT_H * OUT_W
    HW = OUT_H * OUT_W
    eps = 1e-5

    out = torch.empty_like(conv_out)

    grid_pw = lambda meta: (cdiv(N_TOTAL, meta['BLOCK_SIZE']),)

    fused_mish_bn_kernel[grid_pw](
        conv_out, b_bn_running_mean, b_bn_running_var, p_bn_weight, p_bn_bias,
        out,
        N_TOTAL, OUT_C, HW, eps,
    )

    return out


def get_inputs():
    batch_size = 64
    in_channels = 64
    out_channels = 128
    height, width = 128, 128
    kernel_size = 3

    p_conv_weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda')
    p_conv_bias = torch.randn(out_channels, device='cuda')
    p_bn_weight = torch.randn(out_channels, device='cuda')
    p_bn_bias = torch.randn(out_channels, device='cuda')
    b_bn_running_mean = torch.randn(out_channels, device='cuda')
    b_bn_running_var = torch.abs(torch.randn(out_channels, device='cuda')) + 0.1
    x = torch.randn(batch_size, in_channels, height, width, device='cuda')

    return [p_conv_weight, p_conv_bias, p_bn_weight, p_bn_bias,


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
        import torch.nn as nn
        p_conv_weight, p_conv_bias, p_bn_weight, p_bn_bias, b_bn_running_mean, b_bn_running_var, x = inputs

        # Reference computation
        conv_ref = torch.nn.functional.conv2d(x, p_conv_weight, p_conv_bias)
        sp_ref = torch.nn.functional.softplus(conv_ref)
        tanh_ref = torch.tanh(sp_ref)
        mish_ref = tanh_ref * conv_ref
        # BN inference
        bn_ref = (mish_ref - b_bn_running_mean[None, :, None, None]) / torch.sqrt(b_bn_running_var[None, :, None, None] + 1e-5)
        bn_ref = bn_ref * p_bn_weight[None, :, None, None] + p_bn_bias[None, :, None, None]

        max_diff = (output - bn_ref).abs().max().item()
        print(f"Max difference: {max_diff:.6f}")
        if max_diff < 1e-3:
            print("PASSED: Results match within tolerance.")
        else:
            print("FAILED: Results do not match!")