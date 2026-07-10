import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 8},  num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 8},  num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W',
         'KERNEL_H', 'KERNEL_W', 'STRIDE_H', 'STRIDE_W', 'PAD_H', 'PAD_W'],
)
@triton.jit
def conv2d_relu_hardswish_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # strides for X (NHWC layout)
    s_xn: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr, s_xc: tl.constexpr,
    # strides for W (NHWC-like: [OUT_C, KH, KW, IN_C])
    s_wn: tl.constexpr, s_wh: tl.constexpr, s_ww: tl.constexpr, s_wc: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program_id(0) -> M tile (batch * out_h * out_w)
    # program_id(1) -> N tile (out_c)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W
    BLOCK_K_COUNT = (TOTAL_K + BLOCK_K - 1) // BLOCK_K

    for k_tile in range(BLOCK_K_COUNT):
        offs_k = k_tile * BLOCK_K + tl.arange(0, BLOCK_K)

        # Decode K -> (in_c, kh, kw)
        in_c_idx = offs_k % IN_C
        kw_idx = (offs_k // IN_C) % KERNEL_W
        kh_idx = offs_k // (IN_C * KERNEL_W)

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_spatial = (ih >= 0) & (ih < IN_H) & (iw >= 0) & (iw < IN_W)
        mask_x = valid_m & valid_k & valid_spatial

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: shape [OUT_C, KH, KW, IN_C] -> index [offs_n, kh, kw, in_c]
        valid_n = offs_n[None, :] < OUT_C
        valid_k2 = offs_k[:, None] < TOTAL_K
        mask_w = valid_k2 & valid_n

        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias_mask = offs_n < OUT_C
    bias = tl.load(B_ptr + offs_n, mask=bias_mask, other=0.0)
    acc = acc + bias[None, :]

    # ReLU
    acc = tl.maximum(acc, 0.0)

    # HardSwish: x * clamp((x + 3) / 6, 0, 1)
    hs_inner = (acc + 3.0) * (1.0 / 6.0)
    hs_inner = tl.minimum(tl.maximum(hs_inner, 0.0), 1.0)
    result = acc * hs_inner

    # Store output in NCHW layout: [batch, out_c, out_h, out_w]
    # stride: out_c dim = OUT_H*OUT_W, out_h dim = OUT_W, out_w dim = 1
    # flat index = batch * OUT_C*OUT_H*OUT_W + out_c * OUT_H*OUT_W + out_h * OUT_W + out_w
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)

    out_ptrs = (Y_ptr
    tl.store(out_ptrs, result.to(tl.float32), mask=mask_out)


def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: [128, 8, 128, 128] NCHW
    # weight: [64, 8, 3, 3] (OUT_C, IN_C, KH, KW)
    # bias: [64]

    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = p_conv_weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0

    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    # Convert x to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # s_xn, s_xh, s_xw, s_xc
    s_xn = IN_H * IN_W * IN_C
    s_xh = IN_W * IN_C
    s_xw = IN_C
    s_xc = 1

    # Convert weight to [OUT_C, KH, KW, IN_C] for coalesced access
    w_okhwc = p_conv_weight.permute(0, 2, 3, 1).contiguous()
    # s_wn (out_c stride), s_wh, s_ww, s_wc
    s_wn = KERNEL_H * KERNEL_W * IN_C
    s_wh = KERNEL_W * IN_C
    s_ww = IN_C
    s_wc = 1

    y = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))

    conv2d_relu_hardswish_kernel[grid](
        x_nhwc, w_okhwc, p_conv_bias, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xh, s_xw, s_xc,
        s_wn, s_wh, s_ww, s_wc,
    )

    return y


def get_inputs():
    p_conv_weight = torch.randn(64, 8, 3, 3, device='cuda', dtype=torch.float32)
    p_conv_bias = torch.randn(64, device='cuda', dtype=torch.float32)
    x = torch.rand(128, 8, 128, 128, device='cuda', dtype=torch.float32)
    return [p_conv_weight, p_conv_bias, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        import torch.nn.functional as F
        p_conv_weight, p_conv_bias, x = inputs
        conv_out = F.conv2d(x, p_conv_weight, p_conv_bias, stride=1, padding=0)
        relu_out = torch.relu(conv_out)
        ref = relu_out * torch.clamp((relu_out + 3) / 6, 0, 1)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-4), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")