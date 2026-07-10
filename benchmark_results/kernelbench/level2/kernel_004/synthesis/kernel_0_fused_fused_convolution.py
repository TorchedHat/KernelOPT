import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W'],
)
@triton.jit
def conv2d_double_mish_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # X strides (NHWC)
    s_xn: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr, s_xc: tl.constexpr,
    # W strides (NHWC-like: OUT_C, KH, KW, IN_C)
    s_wn: tl.constexpr, s_wh: tl.constexpr, s_ww: tl.constexpr, s_wc: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = IN_C * KH * KW
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    valid_m = (batch < BATCH) & (out_h < OUT_H) & (out_w < OUT_W)
    valid_n = offs_n < OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KH * KW

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        valid_k = offs_k < TOTAL_K

        # Decode K -> (kh, kw, in_c)
        in_c = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw = tmp_k % KW
        kh = tmp_k // KW

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_x = (
            valid_m[:, None]
            & (ih >= 0) & (ih < IN_H)
            & (iw >= 0) & (iw < IN_W)
        )

        x_ptrs = X_ptr + (
            batch[:, None] * s_xn
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: shape (OUT_C, KH, KW, IN_C) -> strides s_wn, s_wh, s_ww, s_wc
        w_ptrs = W_ptr + (
            offs_n[None, :] * s_wn
        )
        valid_w = valid_k[:, None] & valid_n[None, :]
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Add bias
    bias = tl.load(B_ptr + offs_n, mask=valid_n, other=0.0)
    acc = acc + bias[None, :]

    # Apply Mish twice: mish(x) = x * tanh(softplus(x))
    # softplus(x) = log(1 + exp(x)), numerically: if x > 20, softplus(x) ≈ x
    # First Mish
    x1 = acc
    sp1_gt = x1 > 20.0
    sp1_exp = libdevice.exp(x1)
    sp1_log = libdevice.log1p(sp1_exp)
    sp1 = tl.where(sp1_gt, x1, sp1_log)
    tanh1 = libdevice.tanh(sp1)
    x2 = x1 * tanh1

    # Second Mish
    sp2_gt = x2 > 20.0
    sp2_exp = libdevice.exp(x2)
    sp2_log = libdevice.log1p(sp2_exp)
    sp2 = tl.where(sp2_gt, x2, sp2_log)
    tanh2 = libdevice.tanh(sp2)
    out = x2 * tanh2

    # Store: output shape (BATCH, OUT_C, OUT_H, OUT_W) in NCHW
    # stride: (OUT_C*OUT_H*OUT_W, OUT_H*OUT_W, OUT_W, 1)
    out_stride_n = OUT_C * OUT_H * OUT_W
    out_stride_c = OUT_H * OUT_W
    out_stride_h = OUT_W
    out_stride_w = 1

    out_ptrs = Y_ptr + (
        batch[:, None] * out_stride_n
    )
    valid_out = valid_m[:, None] & valid_n[None, :]
    tl.store(out_ptrs, out.to(tl.float32), mask=valid_out)


def kernel_function(weight, bias, x):
    # x: (N, C_in, H, W) = (64, 64, 256, 256)
    # weight: (C_out, C_in, KH, KW) = (128, 64, 3, 3)
    # bias: (C_out,) = (128,)

    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1

    # Use NHWC layout for input for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
    # Weight layout: (OUT_C, KH, KW, IN_C) for coalesced access
    w_nhwc = weight.permute(0, 2, 3, 1).contiguous()  # (OUT_C, KH, KW, IN_C)

    y = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M = BATCH * OUT_H * OUT_W
    N = OUT_C

    # Strides for NHWC input
    s_xn = x_nhwc.stride(0)
    s_xh = x_nhwc.stride(1)
    s_xw = x_nhwc.stride(2)
    s_xc = x_nhwc.stride(3)

    # Strides for weight (OUT_C, KH, KW, IN_C)
    s_wn = w_nhwc.stride(0)
    s_wh = w_nhwc.stride(1)
    s_ww = w_nhwc.stride(2)
    s_wc = w_nhwc.stride(3)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    conv2d_double_mish_kernel[grid](
        x_nhwc, w_nhwc, bias, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xh, s_xw, s_xc,
        s_wn, s_wh, s_ww, s_wc,
    )

    return y


def get_inputs():
    weight = torch.rand(128, 64, 3, 3, device='cuda', dtype=torch.float32)
    bias = torch.rand(128, device='cuda', dtype=torch.float32)
    x = torch.rand(64, 64, 256, 256, device='cuda', dtype=torch.float32)
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
        weight, bias, x = inputs
        import torch.nn.functional as F
        ref = F.conv2d(x, weight, bias)
        ref = F.mish(ref)
        ref = F.mish(ref)
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff vs reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Correctness check failed: max_diff={max_diff}"
        print("Kernel executed successfully.")