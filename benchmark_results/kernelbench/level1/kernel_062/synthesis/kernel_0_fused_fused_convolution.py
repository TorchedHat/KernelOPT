import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
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
    # Strides for X: (BATCH, IN_C, IN_H, IN_W) in NCHW
    s_xn: tl.constexpr, s_xc: tl.constexpr, s_xh: tl.constexpr, s_xw: tl.constexpr,
    # Strides for W: (OUT_C, IN_C, KERNEL_H, KERNEL_W)
    s_woc: tl.constexpr, s_wic: tl.constexpr, s_wkh: tl.constexpr, s_wkw: tl.constexpr,
    # Strides for Y: (BATCH, OUT_C, OUT_H, OUT_W) in NCHW
    s_yn: tl.constexpr, s_yc: tl.constexpr, s_yh: tl.constexpr, s_yw: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # M dimension: BATCH * OUT_H * OUT_W
    # N dimension: OUT_C
    # K dimension: IN_C * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Output spatial positions for this block
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M -> (batch, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    batch_idx = tmp // OUT_H

    mask_m = (batch_idx < BATCH) & (out_h_idx < OUT_H) & (out_w_idx < OUT_W)
    mask_n = offs_n < OUT_C

    # Total K = IN_C * KERNEL_H * KERNEL_W
    TOTAL_K = IN_C * KERNEL_H * KERNEL_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        mask_k = offs_k < TOTAL_K

        # Decode K -> (ic, kh, kw)
        kw_idx = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh_idx = tmp_k % KERNEL_H
        ic_idx = tmp_k // KERNEL_H

        # Input coordinates
        ih = out_h_idx[:, None] * STRIDE_H + kh_idx[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw_idx[None, :] - PAD_W

        valid_x = (
            mask_m[:, None]
            & (ih >= 0)
            & (iw >= 0)
        )

        # Compute X pointers
        x_ptrs = (
        )
        x_tile = tl.load(x_ptrs, mask=valid_x, other=0.0)

        # Weight pointers: W[oc, ic, kh, kw]
        # offs_n: [BLOCK_N], offs_k: [BLOCK_K]
        # w_ptrs shape: [BLOCK_K, BLOCK_N]
        w_ptrs = (
        )
        valid_w = mask_k[:, None] & mask_n[None, :]
        w_tile = tl.load(w_ptrs, mask=valid_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output Y[batch, oc, out_h, out_w]
    y_ptrs = (
    )
    out_mask = mask_m[:, None] & mask_n[None, :]
    tl.store(y_ptrs, acc.to(tl.float32), mask=out_mask)


def kernel_function(weight, x):
    # weight: (OUT_C, IN_C, KERNEL_H, KERNEL_W)
    # x: (BATCH, IN_C, IN_H, IN_W)
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, IN_C_w, KERNEL_H, KERNEL_W = weight.shape
    assert IN_C == IN_C_w

    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0

    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    # Output in NCHW layout
    y = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M = BATCH * OUT_H * OUT_W
    N = OUT_C

    # Make sure inputs are contiguous
    x_c = x.contiguous()
    w_c = weight.contiguous()

    # Strides for X (NCHW)
    s_xn, s_xc, s_xh, s_xw = x_c.stride()
    # Strides for W (OUT_C, IN_C, KH, KW)
    s_woc, s_wic, s_wkh, s_wkw = w_c.stride()
    # Strides for Y (NCHW)
    s_yn, s_yc, s_yh, s_yw = y.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M, meta['BLOCK_M']), cdiv(N, meta['BLOCK_N']))

    conv2d_implicit_gemm_kernel[grid](
        x_c, w_c, y,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        s_xn, s_xc, s_xh, s_xw,
        s_woc, s_wic, s_wkh, s_wkw,
        s_yn, s_yc, s_yh, s_yw,
    )

    return y


def get_inputs():
    weight = torch.randn(64, 32, 5, 9, device='cuda', dtype=torch.float32)
    x = torch.randn(8, 32, 512, 512, device='cuda', dtype=torch.float32)
    return [weight, x]


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
        weight, x = inputs
        import torch.nn.functional as F
        ref = F.conv2d(x, weight, bias=None, stride=1, padding=0)
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Correctness check failed: max_diff={max_diff}"
        print("Kernel executed successfully.")
        print(f"Output shape: {output.shape}")