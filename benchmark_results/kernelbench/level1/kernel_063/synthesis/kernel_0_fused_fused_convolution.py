import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv2d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # strides for X: (BATCH, IN_C, IN_H, IN_W) in NCHW
    s_xn: tl.constexpr, s_xc: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr,
    # strides for W: (OUT_C, IN_C, KH, KW)
    s_wo: tl.constexpr, s_wi: tl.constexpr, s_wkh: tl.constexpr, s_wkw: tl.constexpr,
    # strides for Y: (BATCH, OUT_C, OUT_H, OUT_W)
    s_yn: tl.constexpr, s_yc: tl.constexpr, s_yh: tl.constexpr, s_yw: tl.constexpr,
    M_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index -> (batch, out_h, out_w)
    out_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h = tmp % OUT_H
    batch = tmp // OUT_H

    mask_m = offs_m < M_TOTAL
    mask_n = offs_n < OUT_C

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    for k_off in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_off + tl.arange(0, BLOCK_K)

        # Decode K index -> (in_c, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        in_c = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w[:, None] * STRIDE_W + kw[None, :] - PAD_W

        valid_k = offs_k < TOTAL_K
        valid_x = (
            mask_m[:, None]
            & (ih >= 0)
            & (iw >= 0)
        )

        x_ptrs = (
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        valid_w = valid_k[:, None] & mask_n[None, :]
        w_ptrs = (
        )
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output: Y is (BATCH, OUT_C, OUT_H, OUT_W)
    y_ptrs = (
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(weight, x):
    # weight: (OUT_C, IN_C, KH, KW) = (128, 16, 3, 3)
    # x: (BATCH, IN_C, IN_H, IN_W) = (16, 16, 1024, 1024)
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_H, KERNEL_W = weight.shape
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0

    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    M_TOTAL = BATCH * OUT_H * OUT_W

    y = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    # Strides for x (NCHW)
    s_xn, s_xc, s_xh, s_xw = x.stride()
    # Strides for weight (OUT_C, IN_C, KH, KW)
    s_wo, s_wi, s_wkh, s_wkw = weight.stride()
    # Strides for y (BATCH, OUT_C, OUT_H, OUT_W)
    s_yn, s_yc, s_yh, s_yw = y.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))

    conv2d_implicit_gemm_kernel[grid](
        x, weight, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xc, s_xh, s_xw,
        s_wo, s_wi, s_wkh, s_wkw,
        s_yn, s_yc, s_yh, s_yw,
        M_TOTAL,
    )

    return y


def get_inputs():
    weight = torch.randn(128, 16, 3, 3, device='cuda', dtype=torch.float32)
    x = torch.randn(16, 16, 1024, 1024, device='cuda', dtype=torch.float32)
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
        import torch.nn.functional as F
        ref = F.conv2d(x, weight, bias=None, stride=1, padding=0)
        out = kernel_function(weight, x)
        torch.cuda.synchronize()
        max_diff = (out - ref).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")