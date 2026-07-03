import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_D', 'IN_H', 'IN_W', 'OUT_C', 'OUT_D', 'OUT_H', 'OUT_W',
         'KERNEL_D', 'KERNEL_H', 'KERNEL_W'],
)
@triton.jit
def conv3d_implicit_gemm_kernel(
    X_ptr, W_ptr, Y_ptr,
    BATCH: tl.constexpr,
    IN_C: tl.constexpr,
    IN_D: tl.constexpr,
    IN_H: tl.constexpr,
    IN_W: tl.constexpr,
    OUT_C: tl.constexpr,
    OUT_D: tl.constexpr,
    OUT_H: tl.constexpr,
    OUT_W: tl.constexpr,
    KERNEL_D: tl.constexpr,
    KERNEL_H: tl.constexpr,
    KERNEL_W: tl.constexpr,
    STRIDE_D: tl.constexpr,
    STRIDE_H: tl.constexpr,
    STRIDE_W: tl.constexpr,
    PAD_D: tl.constexpr,
    PAD_H: tl.constexpr,
    PAD_W: tl.constexpr,
    # Input strides
    stride_xn: tl.constexpr,
    stride_xc: tl.constexpr,
    stride_xd: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_xw: tl.constexpr,
    # Weight strides
    stride_wco: tl.constexpr,
    stride_wci: tl.constexpr,
    stride_wkd: tl.constexpr,
    stride_wkh: tl.constexpr,
    stride_wkw: tl.constexpr,
    # Output strides
    stride_yn: tl.constexpr,
    stride_yc: tl.constexpr,
    stride_yd: tl.constexpr,
    stride_yh: tl.constexpr,
    stride_yw: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # M = BATCH * OUT_D * OUT_H * OUT_W
    # N = OUT_C
    # K = IN_C * KERNEL_D * KERNEL_H * KERNEL_W
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Offsets for M dimension (output spatial positions)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # Offsets for N dimension (output channels)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M offset -> (batch, out_d, out_h, out_w)
    out_w_idx = offs_m % OUT_W
    tmp = offs_m // OUT_W
    out_h_idx = tmp % OUT_H
    tmp2 = tmp // OUT_H
    out_d_idx = tmp2 % OUT_D
    batch_idx = tmp2 // OUT_D

    # Total K = IN_C * KERNEL_D * KERNEL_H * KERNEL_W
    TOTAL_K = IN_C * KERNEL_D * KERNEL_H * KERNEL_W

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over K dimension in tiles of BLOCK_K
    for k_start in range(0, TOTAL_K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        # Decode K offset -> (in_c, kd, kh, kw)
        kw = offs_k % KERNEL_W
        tmp_k = offs_k // KERNEL_W
        kh = tmp_k % KERNEL_H
        tmp_k2 = tmp_k // KERNEL_H
        kd = tmp_k2 % KERNEL_D
        in_c = tmp_k2 // KERNEL_D

        # Compute input coordinates
        # Shape: [BLOCK_M, BLOCK_K]
        id_ = out_d_idx[:, None] * STRIDE_D + kd[None, :] - PAD_D
        ih = out_h_idx[:, None] * STRIDE_H + kh[None, :] - PAD_H
        iw = out_w_idx[:, None] * STRIDE_W + kw[None, :] - PAD_W

        # Validity mask for input
        valid_m = (offs_m < BATCH * OUT_D * OUT_H * OUT_W)
        valid_k = (offs_k < TOTAL_K)
        valid_spatial = (
            (id_ >= 0) & (id_ < IN_D) &
            (ih >= 0) & (ih < IN_H) &
            (iw >= 0) & (iw < IN_W)
        )
        mask_x = valid_m[:, None] & valid_k[None, :] & valid_spatial

        # Load input tile [BLOCK_M, BLOCK_K]
        x_ptrs = (X_ptr +
                  batch_idx[:, None] * stride_xn +
                  in_c[None, :] * stride_xc +
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Load weight tile [BLOCK_K, BLOCK_N]
        # Weight shape: [OUT_C, IN_C, KD, KH, KW]
        # offs_k encodes (in_c, kd, kh, kw)
        mask_w = valid_k[:, None] & (offs_n[None, :] < OUT_C)
        w_ptrs = (W_ptr +
                  offs_n[None, :] * stride_wco +
                  in_c[:, None] * stride_wci +
                  kd[:, None] * stride_wkd +
                  kh[:, None] * stride_wkh +
                  kw[:, None] * stride_wkw)
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output [BLOCK_M, BLOCK_N]
    # Output shape: [BATCH, OUT_C, OUT_D, OUT_H, OUT_W]
    M_TOTAL = BATCH * OUT_D * OUT_H * OUT_W
    mask_out = (offs_m[:, None] < M_TOTAL) & (offs_n[None, :] < OUT_C)

    y_ptrs = (Y_ptr +
              batch_idx[:, None] * stride_yn +
              offs_n[None, :] * stride_yc +
              out_d_idx[:, None] * stride_yd +
              out_h_idx[:, None] * stride_yh +
              out_w_idx[:, None] * stride_yw)
    tl.store(y_ptrs, acc.to(tl.float32), mask=mask_out)


def kernel_function(weight, x):
    # weight: [OUT_C, IN_C, KD, KH, KW] = [64, 3, 3, 3, 3]
    # x: [BATCH, IN_C, IN_D, IN_H, IN_W] = [16, 3, 64, 64, 64]
    BATCH, IN_C, IN_D, IN_H, IN_W = x.shape
    OUT_C, IN_C_w, KERNEL_D, KERNEL_H, KERNEL_W = weight.shape
    assert IN_C == IN_C_w

    STRIDE_D, STRIDE_H, STRIDE_W = 1, 1, 1
    PAD_D, PAD_H, PAD_W = 0, 0, 0

    OUT_D = (IN_D + 2 * PAD_D - KERNEL_D) // STRIDE_D + 1
    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    y = torch.empty((BATCH, OUT_C, OUT_D, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_D * OUT_H * OUT_W

    def cdiv(a, b):
        return (a + b - 1) // b

    grid = lambda meta: (cdiv(M_TOTAL, meta['BLOCK_M']), cdiv(OUT_C, meta['BLOCK_N']))

    conv3d_implicit_gemm_kernel[grid](
        x, weight, y,
        BATCH=BATCH,
        IN_C=IN_C,
        IN_D=IN_D,
        IN_H=IN_H,
        IN_W=IN_W,
        OUT_C=OUT_C,
        OUT_D=OUT_D,
        OUT_H=OUT_H,
        OUT_W=OUT_W,
        KERNEL_D=KERNEL_D,
        KERNEL_H=KERNEL_H,
        KERNEL_W=KERNEL_W,
        STRIDE_D=STRIDE_D,
        STRIDE_H=STRIDE_H,
        STRIDE_W=STRIDE_W,
        PAD_D=PAD_D,
        PAD_H=PAD_H,
        PAD_W=PAD_W,
        stride_xn=x.stride(0),
        stride_xc=x.stride(1),
        stride_xd=x.stride(2),
        stride_xh=x.stride(3),
        stride_xw=x.stride(4),
        stride_wco=weight.stride(0),
        stride_wci=weight.stride(1),
        stride_wkd=weight.stride(2),
        stride_wkh=weight.stride(3),
        stride_wkw=weight.stride(4),
        stride_yn=y.stride(0),
        stride_yc=y.stride(1),
        stride_yd=y.stride(2),
        stride_yh=y.stride(3),
        stride_yw=y.stride(4),
    )

    return y


def get_inputs():
    weight = torch.randn(64, 3, 3, 3, 3, device='cuda', dtype=torch.float32)
    x = torch.randn(16, 3, 64, 64, 64, device='cuda', dtype=torch.float32)
    return [weight, x]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    weight, x = inputs

    # Verify correctness
    output = kernel_function(weight, x)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Compare with PyTorch reference
        import torch.nn.functional as F
        ref = F.conv3d(x, weight, bias=None, stride=1, padding=0)
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match PyTorch reference.")
        print(f"Output shape: {output.shape}")