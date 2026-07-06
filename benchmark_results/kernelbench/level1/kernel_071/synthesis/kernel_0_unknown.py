import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
    ],
    key=['M_TOTAL', 'OUT_C', 'TOTAL_K'],
)
@triton.jit
def conv_transpose2d_kernel(
    X_ptr, W_ptr, Y_ptr,
    N, IN_C, IN_H, IN_W,
    OUT_C, OUT_H, OUT_W,
    KH, KW,
    STRIDE_H, STRIDE_W,
    PAD_H, PAD_W,
    # strides for X: (N, IN_C, IN_H, IN_W)
    s_xn, s_xc, s_xh, s_xw,
    # strides for W: (IN_C, OUT_C, KH, KW)
    s_wi, s_wo, s_wkh, s_wkw,
    # strides for Y: (N, OUT_C, OUT_H, OUT_W)
    s_yn, s_yc, s_yh, s_yw,
    M_TOTAL, OUT_C_: tl.constexpr,
    TOTAL_K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    out_h = (offs_m // OUT_W) % OUT_H
    batch = offs_m // (OUT_W * OUT_H)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kh, kw)
        # K = in_c * KH * KW + kh * KW + kw
        in_c = offs_k // (KH * KW)
        rem = offs_k % (KH * KW)
        kh = rem // KW
        kw = rem % KW

        # For transposed conv: input position ih = oh - kh (no stride, no padding here)
        # With padding P: ih = oh + PAD_H - kh
        # With stride S: input ih = (oh + PAD_H - kh) / STRIDE_H (only integer)
        # For stride=1, pad=0: ih = oh - kh
        ih = out_h[:, None] - kh[None, :]  # [BLOCK_M, BLOCK_K]
        iw = out_w[:, None] - kw[None, :]  # [BLOCK_M, BLOCK_K]

        valid_m = offs_m[:, None] < M_TOTAL
        valid_k = offs_k[None, :] < TOTAL_K
        valid_ih = (ih >= 0) & (ih < IN_H)
        valid_iw = (iw >= 0) & (iw < IN_W)
        valid_batch = batch[:, None] < N
        valid_ic = in_c[None, :] < IN_C

        mask_x = valid_m & valid_k & valid_ih & valid_iw & valid_batch & valid_ic

        x_ptrs = (X_ptr
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Weight: W[in_c, out_c, kh, kw] -> shape [BLOCK_K, BLOCK_N]
        valid_n = offs_n[None, :] < OUT_C_
        mask_w = valid_k[:, None] & valid_n & valid_ic[:, None]

        w_ptrs = (W_ptr
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc=acc)

    # Store output
    valid_m_out = offs_m[:, None] < M_TOTAL
    valid_n_out = offs_n[None, :] < OUT_C_

    y_ptrs = (Y_ptr

    tl.store(y_ptrs, acc.to(tl.float32), mask=valid_m_out & valid_n_out)


def kernel_function(weight, x):
    # weight: (IN_C, OUT_C, KH, KW) - PyTorch ConvTranspose2d weight layout
    # x: (N, IN_C, IN_H, IN_W)
    N, IN_C, IN_H, IN_W = x.shape
    # PyTorch ConvTranspose2d weight shape: (in_channels, out_channels, kH, kW)
    IN_C_w, OUT_C, KH, KW = weight.shape

    STRIDE_H = 1
    STRIDE_W = 1
    PAD_H = 0
    PAD_W = 0

    OUT_H = (IN_H - 1) * STRIDE_H - 2 * PAD_H + KH
    OUT_W = (IN_W - 1) * STRIDE_W - 2 * PAD_W + KW

    M_TOTAL = N * OUT_H * OUT_W
    TOTAL_K = IN_C * KH * KW

    y = torch.zeros(N, OUT_C, OUT_H, OUT_W, device=x.device, dtype=x.dtype)

    # Strides for x: (N, IN_C, IN_H, IN_W)
    s_xn, s_xc, s_xh, s_xw = x.stride()
    # Strides for weight: (IN_C, OUT_C, KH, KW)
    s_wi, s_wo, s_wkh, s_wkw = weight.stride()
    # Strides for y: (N, OUT_C, OUT_H, OUT_W)
    s_yn, s_yc, s_yh, s_yw = y.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))

    conv_transpose2d_kernel[grid](
        x, weight, y,
        N, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xc, s_xh, s_xw,
        s_wi, s_wo, s_wkh, s_wkw,
        s_yn, s_yc, s_yh, s_yw,
        M_TOTAL, OUT_C,
        TOTAL_K,
    )

    return y


def get_inputs():
    weight = torch.randn(32, 32, 3, 3, dtype=torch.float32, device='cuda')
    x = torch.randn(8, 32, 512, 1024, dtype=torch.float32, device='cuda')
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, x = inputs

    # Verify correctness
    if not args.profile_only:
        import torch.nn as nn
        conv_t = nn.ConvTranspose2d(32, 32, 3, stride=1, padding=0, bias=False).cuda()
        conv_t.weight.data.copy_(weight)
        ref_out = conv_t(x)
        our_out = kernel_function(weight, x)
        max_diff = (ref_out - our_out).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Correctness check passed!")

    output = kernel_function(*inputs)
    torch.cuda.synchronize()
    if not args.profile_only:
        print("Kernel executed successfully.")