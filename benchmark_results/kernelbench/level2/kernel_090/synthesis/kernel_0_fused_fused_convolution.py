import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_D', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv3d_fused_kernel(
    X_ptr, W_ptr, Bias_ptr, SumTensor_ptr, Out_ptr,
    BATCH: tl.constexpr,
    IN_C: tl.constexpr,
    IN_D: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_C: tl.constexpr,
    OUT_D: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    KD: tl.constexpr,
    KH: tl.constexpr,
    KW: tl.constexpr,
    stride_xn: tl.constexpr,
    stride_xc: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_xw: tl.constexpr,
    stride_wco: tl.constexpr,
    stride_wci: tl.constexpr,
    stride_wd: tl.constexpr,
    stride_wh: tl.constexpr,
    stride_ww: tl.constexpr,
    M_TOTAL: tl.constexpr,
    TOTAL_K: tl.constexpr,
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

    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_block in range(BLOCK_K_COUNT):
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

        # Decode K index -> (kd, kh, kw, in_c)
        in_c = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KW
        tmp_k = tmp_k // KW
        kh = tmp_k % KH
        kd = tmp_k // KH

        # Input spatial coordinates
        id_ = out_d[:, None] * 1 + kd[None, :] - 0  # stride=1, pad=0
        ih = out_h[:, None] * 1 + kh[None, :] - 0
        iw = out_w[:, None] * 1 + kw[None, :] - 0

        valid_m = (offs_m < M_TOTAL)
        valid_k = (offs_k < TOTAL_K)
        valid_spatial = (
            (id_ >= 0) & (id_ < IN_D) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W)
        )
        mask_x = valid_m[:, None] & valid_k[None, :] & valid_spatial

        x_ptrs = (
        )
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: shape [OUT_C, IN_C, KD, KH, KW]
        # w_ptr[oc, ic, kd, kh, kw] = W_ptr + oc*stride_wco + ic*stride_wci + kd*stride_wd + kh*stride_wh + kw*stride_ww
        mask_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = (
        )
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(Bias_ptr + offs_n, mask=(offs_n < OUT_C), other=0.0)
    acc = acc + bias[None, :]

    # LeakyReLU with negative_slope=0.2
    leaky = tl.where(acc > 0.0, acc, acc * 0.2)

    # Add sum_tensor (shape [64, 1, 1, 1] -> broadcast over spatial dims)
    sum_t = tl.load(SumTensor_ptr + offs_n, mask=(offs_n < OUT_C), other=0.0)
    leaky = leaky + sum_t[None, :]

    # Clamp to [-1, 1]
    clamped = tl.minimum(tl.maximum(leaky, -1.0), 1.0)

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    gelu_out = clamped * 0.5 * (1.0 + libdevice.erf(clamped * 0.7071067811865476))

    # Store output
    # Output layout: [BATCH, OUT_C, OUT_D, OUT_H, OUT_W]
    # stride: OUT_C*OUT_D*OUT_H*OUT_W, OUT_D*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1
    out_stride_n = OUT_C * OUT_D * OUT_H * OUT_W
    out_stride_c = OUT_D * OUT_H * OUT_W
    out_stride_d = OUT_H * OUT_W
    out_stride_h = OUT_W
    out_stride_w = 1

    out_idx = (
        batch[:, None] * out_stride_n
    )

    valid_out = (
        (offs_m < M_TOTAL)[:, None]
    )

    tl.store(Out_ptr + out_idx, gelu_out.to(tl.float32), mask=valid_out)


def kernel_function(p_conv_weight, p_conv_bias, p_sum_tensor, x):
    # x: [128, 8, 16, 64, 64]
    # weight: [64, 8, 3, 3, 3]
    # bias: [64]
    # sum_tensor: [64, 1, 1, 1]

    BATCH = x.shape[0]
    IN_C = x.shape[1]
    IN_D = x.shape[2]
    IN_H = x.shape[3]
    IN_W = x.shape[4]

    OUT_C = p_conv_weight.shape[0]
    KD = p_conv_weight.shape[2]
    KH = p_conv_weight.shape[3]
    KW = p_conv_weight.shape[4]

    OUT_D = IN_D - KD + 1  # no padding, stride=1
    OUT_H = IN_H - KH + 1
    OUT_W = IN_W - KW + 1

    M_TOTAL = BATCH * OUT_D * OUT_H * OUT_W
    TOTAL_K = IN_C * KD * KH * KW

    output = torch.empty((BATCH, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    # Make sure tensors are contiguous
    x_c = x.contiguous()
    w_c = p_conv_weight.contiguous()
    b_c = p_conv_bias.contiguous()
    s_c = p_sum_tensor.contiguous().view(OUT_C)

    stride_xn = x_c.stride(0)
    stride_xc = x_c.stride(1)
    stride_xd = x_c.stride(2)
    stride_xh = x_c.stride(3)
    stride_xw = x_c.stride(4)

    stride_wco = w_c.stride(0)
    stride_wci = w_c.stride(1)
    stride_wd = w_c.stride(2)
    stride_wh = w_c.stride(3)
    stride_ww = w_c.stride(4)

    BLOCK_M = 256
    BLOCK_N = 64

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(OUT_C, meta['BLOCK_N']),
    )

    conv3d_fused_kernel[grid](
        x_c, w_c, b_c, s_c, output,
        BATCH=BATCH,
        IN_C=IN_C,
        IN_D=IN_D,
        IN_H=IN_H,
        IN_W=IN_W,
        OUT_C=OUT_C,
        OUT_D=OUT_D,
        OUT_H=OUT_H,
        OUT_W=OUT_W,
        KD=KD,
        KH=KH,
        KW=KW,
        stride_xn=stride_xn,
        stride_xc=stride_xc,
        stride_xd=stride_xd,
        stride_xh=stride_xh,
        stride_xw=stride_xw,
        stride_wco=stride_wco,
        stride_wci=stride_wci,
        stride_wd=stride_wd,
        stride_wh=stride_wh,
        stride_ww=stride_ww,
        M_TOTAL=M_TOTAL,
        TOTAL_K=TOTAL_K,
    )

    return output


def get_inputs():
    p_conv_weight = torch.randn(64, 8, 3, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias = torch.randn(64, device='cuda', dtype=torch.float32)
    p_sum_tensor = torch.randn(64, 1, 1, 1, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 8, 16, 64, 64, device='cuda', dtype=torch.float32)
    return [p_conv_weight, p_conv_bias, p_sum_tensor, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    p_conv_weight, p_conv_bias, p_sum_tensor, x = inputs

    output = kernel_function(p_conv_weight, p_conv_bias, p_sum_tensor, x)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F

        # Reference computation
        ref_conv = F.conv3d(x, p_conv_weight, p_conv_bias)
        ref_leaky = F.leaky_relu(ref_conv, negative_slope=0.2)
        ref_add = ref_leaky + p_sum_tensor
        ref_clamp = torch.clamp(ref_add, min=-1.0, max=1.0)
        ref_gelu = F.gelu(ref_clamp)

        max_diff = (output - ref_gelu).abs().max().item()
        print(f"Max difference: {max_diff:.6e}")
        assert torch.allclose(output, ref_gelu, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")