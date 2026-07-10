import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32,  'BLOCK_K': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32,  'BLOCK_K': 8}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64,  'BLOCK_K': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64,  'BLOCK_K': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 8}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64,  'BLOCK_K': 8}, num_stages=3, num_warps=8),
    ],
    key=['BATCH', 'OUT_C', 'OUT_H', 'OUT_W', 'IN_C'],
)
@triton.jit
def conv2d_mish_kernel(
    X_ptr, W_ptr, B_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    SUB_VAL: tl.constexpr,
    # strides for X (NHWC layout)
    s_xn: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr, s_xc: tl.constexpr,
    # strides for W (OUT_C, KH, KW, IN_C) layout
    s_wn: tl.constexpr, s_wh: tl.constexpr, s_ww: tl.constexpr, s_wc: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_H * OUT_W, N = OUT_C, K = KH * KW * IN_C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    M_TOTAL = BATCH * OUT_H * OUT_W

    # Decode M index -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    valid_m = offs_m < M_TOTAL
    valid_n = offs_n < OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Precompute base spatial coords once, outside all loops
    ih_base = out_h * STRIDE_H - PAD_H   # shape [BLOCK_M]
    iw_base = out_w * STRIDE_W - PAD_W   # shape [BLOCK_M]

    for kh_i in tl.static_range(KH):           # scalar, unrolled at compile time
        ih_i = ih_base + kh_i                   # shape [BLOCK_M], scalar addition
        valid_h = (ih_i >= 0) & (ih_i < IN_H)  # shape [BLOCK_M]
        for kw_i in tl.static_range(KW):       # scalar, unrolled at compile time
            iw_i = iw_base + kw_i              # shape [BLOCK_M], scalar addition
            valid_w = (iw_i >= 0) & (iw_i < IN_W)  # shape [BLOCK_M]
            valid_spatial = valid_h & valid_w  # shape [BLOCK_M]
            for ic_start in range(0, IN_C, BLOCK_K):
                offs_ic = ic_start + tl.arange(0, BLOCK_K)  # shape [BLOCK_K]
                valid_ic = offs_ic < IN_C
                mask_x = valid_m[:, None] & valid_spatial[:, None] & valid_ic[None, :]
                x_ptrs = (X_ptr
                          + batch[:, None]  * s_xn
                          + ih_i[:, None]   * s_xh
                          + iw_i[:, None]   * s_xw
                          + offs_ic[None,:] * s_xc)
                x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)
                mask_w = valid_ic[:, None] & valid_n[None, :]
                w_ptrs = (W_ptr
                          + offs_n[None,:] * s_wn
                          + kh_i * s_wh
                          + kw_i * s_ww
                          + offs_ic[:, None] * s_wc)
                w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)
                acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=True)

    # Load bias and apply epilogue: bias + sub(0.5) + sub(0.2) + mish
    bias = tl.load(B_ptr + offs_n, mask=valid_n, other=0.0)
    acc = acc + bias[None, :]
    acc = acc - SUB_VAL  # combined subtract: 0.5 + 0.2 = 0.7

    # Mish: x * tanh(softplus(x)) = x * tanh(log(1 + exp(x)))
    # Numerically stable: if x > 20, softplus(x) ≈ x
    softplus = tl.where(acc > 20.0, acc, libdevice.log(1.0 + libdevice.exp(acc)))
    result = acc * libdevice.tanh(softplus)

    # Store output in NCHW layout: (BATCH, OUT_C, OUT_H, OUT_W)
    # Linear index: batch * OUT_C * OUT_H * OUT_W + c * OUT_H * OUT_W + h * OUT_W + w
    out_idx = batch[:, None] * (OUT_C * OUT_H * OUT_W) + offs_n[None, :] * (OUT_H * OUT_W) + out_h[:, None] * OUT_W + out_w[:, None]
    mask_out = valid_m[:, None] & valid_n[None, :]
    tl.store(Y_ptr + out_idx, result.to(tl.float32), mask=mask_out)

def kernel_function(p_conv_weight, p_conv_bias, x):
    # x: (128, 8, 256, 256) NCHW
    # weight: (64, 8, 3, 3)
    # bias: (64,)
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KH, KW = p_conv_weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2 * PAD_H - KH) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KW) // STRIDE_W + 1

    # Convert x to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (N, H, W, C)
    # Convert weight to (OUT_C, KH, KW, IN_C) for easier indexing
    w_okhwc = p_conv_weight.permute(0, 2, 3, 1).contiguous()  # (OUT_C, KH, KW, IN_C)

    y = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W
    SUB_VAL = 0.5 + 0.2  # combined subtract value

    def grid(meta):
        return (
            triton.cdiv(M_TOTAL, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )

    # Strides for NHWC input
    s_xn, s_xh, s_xw, s_xc = x_nhwc.stride()
    # Strides for (OUT_C, KH, KW, IN_C) weight
    s_wn, s_wh, s_ww, s_wc = w_okhwc.stride()

    conv2d_mish_kernel[grid](
        x_nhwc, w_okhwc, p_conv_bias, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        SUB_VAL,
        s_xn, s_xh, s_xw, s_xc,
        s_wn, s_wh, s_ww, s_wc,
    )

    return y


def get_inputs():
    import torch
    p_conv_weight = torch.randn(64, 8, 3, 3, dtype=torch.float32, device='cuda')
    p_conv_bias = torch.randn(64, dtype=torch.float32, device='cuda')
    x = torch.rand(128, 8, 256, 256, dtype=torch.float32, device='cuda')
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
        ref = F.conv2d(x, p_conv_weight, p_conv_bias)
        ref = ref - 0.5
        ref = ref - 0.2
        ref = F.mish(ref)
        print(f"Output shape: {output.shape}")
        print(f"Ref shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max diff: {max_diff:.4e}")
        assert torch.allclose(output, ref, rtol=1e-3, atol=1e-5), \
            f"Results don't match! max_diff={max_diff:.4e}"
        print("Kernel executed successfully.")