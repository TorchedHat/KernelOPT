import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_H', 'OUT_W', 'KH', 'KW'],
)
@triton.jit
def conv2d_mul_leakyrelu_gelu_kernel(
    X_ptr, W_ptr, bias_ptr, mult_ptr, out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn: tl.constexpr, stride_xc: tl.constexpr, stride_xh: tl.constexpr, stride_xw: tl.constexpr,
    stride_wco: tl.constexpr, stride_wci: tl.constexpr, stride_wkh: tl.constexpr, stride_wkw: tl.constexpr,
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
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_tile in range(BLOCK_K_COUNT):
        k_start = k_tile * BLOCK_K
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kh, kw)
        # K ordering: kh * KW * IN_C + kw * IN_C + in_c  (matches NHWC-style weight layout)
        # Actually let's use: in_c + KW*IN_C*kh + IN_C*kw
        in_c = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KW
        kh_idx = offs_k // (IN_C * KW)

        # Input coordinates
        ih = out_h_idx[:, None] + kh_idx[None, :]  # no padding, stride=1
        iw = out_w_idx[:, None] + kw_idx[None, :]

        valid_k = offs_k < TOTAL_K
        valid_m = offs_m < M_TOTAL
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        mask_x = valid_m[:, None] & valid_spatial & valid_k[None, :]

        # X in NHWC layout: ptr = batch*stride_xn + h*stride_xh + w*stride_xw + c*stride_xc
        x_ptrs = X_ptr + (
            batch_idx[:, None] * stride_xn +
            in_c[None, :] * stride_xc
        )
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # W in layout: [OUT_C, IN_C, KH, KW] -> NHWC-reordered
        # W_ptr indexed as: out_c * stride_wco + in_c * stride_wci + kh * stride_wkh + kw * stride_wkw
        valid_n = offs_n < OUT_C
        mask_w = valid_k[:, None] & valid_n[None, :]
        w_ptrs = W_ptr + (
            offs_n[None, :] * stride_wco +
            in_c[:, None] * stride_wci +
            kh_idx[:, None] * stride_wkh +
            kw_idx[:, None] * stride_wkw
        )
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Epilogue: add bias, multiply, leaky_relu, gelu
    valid_m = offs_m < M_TOTAL
    valid_n = offs_n < OUT_C

    # Load bias [OUT_C]
    bias = tl.load(bias_ptr + offs_n, mask=valid_n, other=0.0)
    acc = acc + bias[None, :]

    # Load multiplier [OUT_C, 1, 1] -> indexed by out_c
    mult = tl.load(mult_ptr + offs_n, mask=valid_n, other=1.0)
    acc = acc * mult[None, :]

    # LeakyReLU (negative_slope=0.01)
    leaky = tl.where(acc > 0.0, acc, acc * 0.01)

    # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
    gelu_out = 0.5 * leaky * (1.0 + libdevice.erf(leaky * 0.7071067811865476))

    # Store output in NCHW layout: [BATCH, OUT_C, OUT_H, OUT_W]
    # out_ptr stride: [OUT_C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1]
    out_stride_n = OUT_C * OUT_H * OUT_W
    out_stride_c = OUT_H * OUT_W
    out_stride_h = OUT_W
    out_stride_w = 1

    out_ptrs = out_ptr + (
        batch_idx[:, None] * out_stride_n +
        offs_n[None, :] * out_stride_c +
        out_h_idx[:, None] * out_stride_h +
        out_w_idx[:, None] * out_stride_w
    )
    mask_out = valid_m[:, None] & valid_n[None, :]
    tl.store(out_ptrs, gelu_out.to(tl.float32), mask=mask_out)


def kernel_function(p_conv_weight, p_conv_bias, p_multiplier, x):
    # x: [64, 64, 256, 256] NCHW
    # p_conv_weight: [64, 64, 3, 3] (OUT_C, IN_C, KH, KW)
    # p_conv_bias: [64]
    # p_multiplier: [64, 1, 1]

    BATCH = x.shape[0]
    IN_C = x.shape[1]
    IN_H = x.shape[2]
    IN_W = x.shape[3]

    OUT_C = p_conv_weight.shape[0]
    KH = p_conv_weight.shape[2]
    KW = p_conv_weight.shape[3]

    OUT_H = IN_H - KH + 1  # no padding, stride=1
    OUT_W = IN_W - KW + 1

    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C]

    # stride_xn, stride_xc, stride_xh, stride_xw in NHWC
    stride_xn = IN_H * IN_W * IN_C
    stride_xh = IN_W * IN_C
    stride_xw = IN_C
    stride_xc = 1

    # Weight strides (original NCHW layout: [OUT_C, IN_C, KH, KW])
    stride_wco = p_conv_weight.stride(0)
    stride_wci = p_conv_weight.stride(1)
    stride_wkh = p_conv_weight.stride(2)
    stride_wkw = p_conv_weight.stride(3)

    # Multiplier: [64, 1, 1] -> flatten to [64]
    mult_flat = p_multiplier.reshape(OUT_C)

    M_TOTAL = BATCH * OUT_H * OUT_W
    out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))

    conv2d_mul_leakyrelu_gelu_kernel[grid](
        x_nhwc, p_conv_weight, p_conv_bias, mult_flat, out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_wco, stride_wci, stride_wkh, stride_wkw,
    )

    return out


def get_inputs():
    p_conv_weight = torch.randn(64, 64, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias = torch.randn(64, device='cuda', dtype=torch.float32)
    p_multiplier = torch.randn(64, 1, 1, device='cuda', dtype=torch.float32)
    x = torch.rand(64, 64, 256, 256, device='cuda', dtype=torch.float32)
    return [p_conv_weight, p_conv_bias, p_multiplier, x]


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
        p_conv_weight, p_conv_bias, p_multiplier, x = inputs
        import torch.nn.functional as F
        ref = F.conv2d(x, p_conv_weight, p_conv_bias)
        ref = ref * p_multiplier
        ref = F.leaky_relu(ref, negative_slope=0.01)
        ref = F.gelu(ref)
        torch.cuda.synchronize()
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match!")