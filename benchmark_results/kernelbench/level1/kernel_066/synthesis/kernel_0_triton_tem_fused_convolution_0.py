import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128,'BLOCK_K': 16}, num_stages=2, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'OUT_C', 'OUT_D', 'OUT_H', 'OUT_W',
         'KERNEL_D', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv3d_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr,
    IN_D: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_D: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_D: tl.constexpr, KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_D: tl.constexpr, STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr, PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    # Input strides
    s_xn: tl.constexpr, s_xc: tl.constexpr, s_xd: tl.constexpr,
    s_xh: tl.constexpr, s_xw: tl.constexpr,
    # Weight strides
    s_wco: tl.constexpr, s_wci: tl.constexpr, s_wkd: tl.constexpr,
    s_wkh: tl.constexpr, s_wkw: tl.constexpr,
    # Output strides
    s_yn: tl.constexpr, s_yc: tl.constexpr, s_yd: tl.constexpr,
    s_yh: tl.constexpr, s_yw: tl.constexpr,
    M_TOTAL: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Each program handles a BLOCK_M x BLOCK_N tile of the output
    # M dimension = BATCH * OUT_D * OUT_H * OUT_W
    # N dimension = OUT_C
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M index into (batch, out_d, out_h, out_w)
    tmp = offs_m
    out_w = tmp % OUT_W
    tmp = tmp // OUT_W
    out_h = tmp % OUT_H
    tmp = tmp // OUT_H
    out_d = tmp % OUT_D
    batch = tmp // OUT_D

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Number of BLOCK_K chunks over IN_C
    BLOCK_K_COUNT = (IN_C + BLOCK_K - 1) // BLOCK_K

    # Loop over kernel positions and input channels
    for dijk in range(KERNEL_D * KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k_block = dijk % BLOCK_K_COUNT
        dij = dijk // BLOCK_K_COUNT
        kw = dij % KERNEL_W
        di = dij // KERNEL_W
        kh = di % KERNEL_H
        kd = di // KERNEL_H

        # Input spatial coordinates
        id_ = out_d * STRIDE_D + kd - PAD_D
        ih = out_h * STRIDE_H + kh - PAD_H
        iw = out_w * STRIDE_W + kw - PAD_W

        # Channel offsets for this block
        offs_k = k_block * BLOCK_K + tl.arange(0, BLOCK_K)

        # Validity masks
        valid_m = offs_m < M_TOTAL
        valid_k = offs_k < IN_C
        valid_spatial = (
            (id_ >= 0) & (id_ < IN_D) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W)
        )
        mask_x = valid_spatial[:, None] & valid_m[:, None] & valid_k[None, :]

        # Compute input pointers: shape [BLOCK_M, BLOCK_K]
        x_ptrs = (
        )
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Compute weight pointers: shape [BLOCK_K, BLOCK_N]
        # Weight layout: [OUT_C, IN_C, KD, KH, KW]
        mask_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = (
        )
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output
    mask_out = (
        (offs_m < M_TOTAL)[:, None] &
        (offs_n < OUT_C)[None, :]
    )
    y_ptrs = (
    )
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def kernel_function(weight, x):
    # weight: [OUT_C, IN_C, KD, KH, KW]
    # x:      [BATCH, IN_C, IN_D, IN_H, IN_W]
    BATCH, IN_C, IN_D, IN_H, IN_W = x.shape
    OUT_C, _, KERNEL_D, KERNEL_H, KERNEL_W = weight.shape

    STRIDE_D, STRIDE_H, STRIDE_W = 1, 1, 1
    PAD_D, PAD_H, PAD_W = 0, 0, 0

    OUT_D = (IN_D + 2 * PAD_D - KERNEL_D) // STRIDE_D + 1
    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    M_TOTAL = BATCH * OUT_D * OUT_H * OUT_W

    y = torch.empty((BATCH, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    # Strides
    s_xn, s_xc, s_xd, s_xh, s_xw = x.stride()
    s_wco, s_wci, s_wkd, s_wkh, s_wkw = weight.stride()
    s_yn, s_yc, s_yd, s_yh, s_yw = y.stride()

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))

    conv3d_kernel[grid](
        x, weight, y,
        BATCH, IN_C,
        IN_D, IN_H, IN_W,
        OUT_C, OUT_D, OUT_H, OUT_W,
        KERNEL_D, KERNEL_H, KERNEL_W,
        STRIDE_D, STRIDE_H, STRIDE_W,
        PAD_D, PAD_H, PAD_W,
        s_xn, s_xc, s_xd, s_xh, s_xw,
        s_wco, s_wci, s_wkd, s_wkh, s_wkw,
        s_yn, s_yc, s_yd, s_yh, s_yw,
        M_TOTAL,
    )

    return y


def get_inputs():
    weight = torch.randn(64, 3, 3, 5, 7, device='cuda', dtype=torch.float32)
    x = torch.randn(8, 3, 16, 128, 128, device='cuda', dtype=torch.float32)
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
        ref = F.conv3d(x, weight, bias=None, stride=1, padding=0)
        out = kernel_function(weight, x)
        max_diff = (out - ref).abs().max().item()
        print(f"Max difference vs reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Correctness check failed: max_diff={max_diff}"
        print("Correctness check passed!")

    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        print(f"Output shape: {output.shape}")
        print("Kernel executed successfully.")