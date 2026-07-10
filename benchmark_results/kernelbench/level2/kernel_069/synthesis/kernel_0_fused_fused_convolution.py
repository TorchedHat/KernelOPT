import torch
import triton
import triton.language as tl

# Epilogue kernel: fuse bias add + hardswish + relu
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2),
    ],
    key=['N'],
)
@triton.jit
def epilogue_kernel(
    out_ptr,
    bias_ptr,
    N,
    C: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < N
    # Each element: index into (N*H*W, C) layout
    # out is stored as (batch, C, H*W) = (batch*H*W, C) flattened
    # We need to figure out which channel each element belongs to
    # out_ptr layout: (128, 64, 126, 126) with strides (1016064, 15876, 126, 1)
    # Flat index: element i -> channel = (i // (126*126)) % 64
    # But we store in a different layout from the conv output
    # Let's just use flat indexing with channel = offs % C
    c_idx = offs % C
    bias = tl.load(bias_ptr + c_idx, mask=mask, other=0.0)
    x = tl.load(out_ptr + offs, mask=mask, other=0.0)
    x = x + bias
    # hardswish: x * clamp(x+3, 0, 6) / 6
    x3 = x + 3.0
    x3 = tl.where(x3 < 0.0, 0.0, x3)
    x3 = tl.where(x3 > 6.0, 6.0, x3)
    hs = x * x3 * (1.0 / 6.0)
    # relu
    result = tl.where(hs > 0.0, hs, 0.0)
    tl.store(out_ptr + offs, result, mask=mask)


# Conv2d implicit GEMM kernel with fused hardswish+relu
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def conv2d_hardswish_relu_kernel(
    X_ptr, W_ptr, bias_ptr, Y_ptr,
    BATCH, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KERNEL_H, KERNEL_W,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    M, N, K,
    # strides for X (NHWC layout)
    s_xn, s_xh, s_xw, s_xc,
    # strides for W (out_c, in_c, kh, kw) -> stored as (out_c, kh, kw, in_c) for NHWC
    s_wo, s_wkh, s_wkw, s_wc,
    # strides for Y (NHWC layout: batch, out_h, out_w, out_c)
    s_yn, s_yh, s_yw, s_yc,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K = IN_C * KERNEL_H * KERNEL_W
    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        # Decode K -> (kh, kw, in_c)
        in_c_idx = offs_k % IN_C
        tmp_k = offs_k // IN_C
        kw_idx = tmp_k % KERNEL_W
        kh_idx = tmp_k // KERNEL_W

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            (offs_m[:, None] < M) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W) &
            (offs_k[None, :] < K)
        )

        x_ptrs = (
            batch_idx[:, None] * s_xn +
            in_c_idx[None, :] * s_xc
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight: stored as (OUT_C, KERNEL_H, KERNEL_W, IN_C) for NHWC
        valid_w = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w_ptrs = (
            offs_n[None, :] * s_wo +
            kh_idx[:, None] * s_wkh +
            kw_idx[:, None] * s_wkw +
            in_c_idx[:, None] * s_wc
        )
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Add bias
    bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
    acc = acc + bias[None, :]

    # HardSwish: x * clamp(x+3, 0, 6) / 6
    x3 = acc + 3.0
    x3 = tl.where(x3 < 0.0, 0.0, x3)
    x3 = tl.where(x3 > 6.0, 6.0, x3)
    hs = acc * x3 * (1.0 / 6.0)

    # ReLU
    result = tl.where(hs > 0.0, hs, 0.0)

    # Store to Y (NHWC layout)
    mask_out = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    y_ptrs = (
        batch_idx[:, None] * s_yn +
        out_h_idx[:, None] * s_yh +
        out_w_idx[:, None] * s_yw +
        offs_n[None, :] * s_yc
    )
    tl.store(y_ptrs, result.to(tl.float32), mask=mask_out)


def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: (128, 8, 128, 128) NCHW
    # weight: (64, 8, 3, 3)
    # bias: (64,)

    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = p_conv_weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0

    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    # Convert input to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    # Convert weight to (OUT_C, KERNEL_H, KERNEL_W, IN_C) for NHWC
    w_nhwc = p_conv_weight.permute(0, 2, 3, 1).contiguous()

    # Output in NHWC layout
    y_nhwc = torch.empty((BATCH, OUT_H, OUT_W, OUT_C), device=x.device, dtype=x.dtype)

    M = BATCH * OUT_H * OUT_W
    N = OUT_C
    K = IN_C * KERNEL_H * KERNEL_W

    # Strides for x_nhwc: (BATCH, IN_H, IN_W, IN_C)
    s_xn = x_nhwc.stride(0)
    s_xh = x_nhwc.stride(1)
    s_xw = x_nhwc.stride(2)
    s_xc = x_nhwc.stride(3)

    # Strides for w_nhwc: (OUT_C, KERNEL_H, KERNEL_W, IN_C)
    s_wo = w_nhwc.stride(0)
    s_wkh = w_nhwc.stride(1)
    s_wkw = w_nhwc.stride(2)
    s_wc = w_nhwc.stride(3)

    # Strides for y_nhwc: (BATCH, OUT_H, OUT_W, OUT_C)
    s_yn = y_nhwc.stride(0)
    s_yh = y_nhwc.stride(1)
    s_yw = y_nhwc.stride(2)
    s_yc = y_nhwc.stride(3)

    def grid(meta):
        return (
            triton.cdiv(M, meta['BLOCK_M']),
            triton.cdiv(N, meta['BLOCK_N']),
        )

    conv2d_hardswish_relu_kernel[grid](
        x_nhwc, w_nhwc, p_conv_bias, y_nhwc,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        M, N, K,
        s_xn, s_xh, s_xw, s_xc,
        s_wo, s_wkh, s_wkw, s_wc,
        s_yn, s_yh, s_yw, s_yc,
    )

    # Convert back to NCHW
    y_nchw = y_nhwc.permute(0, 3, 1, 2).contiguous()
    return y_nchw


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
        p_conv_weight, p_conv_bias, x = inputs
        import torch.nn.functional as F
        conv_out = F.conv2d(x, p_conv_weight, p_conv_bias, stride=1, padding=0)
        hs = F.hardswish(conv_out)
        ref = F.relu(hs)

        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={((output - ref).abs().max().item()):.4e}"
        print("Kernel executed successfully.")