import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 8, 'BLOCK_K': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_N': 8, 'BLOCK_K': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 32, 'BLOCK_K': 16}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 16, 'BLOCK_K': 16}, num_warps=2, num_stages=2),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W',
         'KERNEL_H', 'KERNEL_W', 'STRIDE_H', 'STRIDE_W', 'GROUPS'],
)
@triton.jit
def depthwise_conv2d_kernel(
    X_ptr, W_ptr, Out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KERNEL_H: tl.constexpr, KERNEL_W: tl.constexpr,
    STRIDE_H: tl.constexpr, STRIDE_W: tl.constexpr,
    PAD_H: tl.constexpr, PAD_W: tl.constexpr,
    GROUPS: tl.constexpr,
    # Strides for X in NHWC layout (after permute)
    stride_xn: tl.constexpr, stride_xh: tl.constexpr, stride_xw: tl.constexpr, stride_xc: tl.constexpr,
    # Strides for W: shape [OUT_C, GROUP_IN_C, KH, KW]
    stride_woc: tl.constexpr, stride_wic: tl.constexpr, stride_wkh: tl.constexpr, stride_wkw: tl.constexpr,
    # Strides for Out in NHWC layout
    stride_on: tl.constexpr, stride_oh: tl.constexpr, stride_ow: tl.constexpr, stride_oc: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program_id(0): tile over M = BATCH * OUT_H * OUT_W
    # program_id(1): tile over output channels within a group
    # program_id(2): group index
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    group = tl.program_id(2)

    GROUP_IN_C = IN_C // GROUPS
    GROUP_OUT_C = OUT_C // GROUPS

    # M dimension: BATCH * OUT_H * OUT_W
    M_TOTAL = BATCH * OUT_H * OUT_W

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    # Decode M offsets
    idx_w = offs_m % OUT_W
    tmp = offs_m // OUT_W
    idx_h = tmp % OUT_H
    idx_n = tmp // OUT_H

    # Output channel indices within group
    idx_oc = offs_n  # local channel index within group

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Loop over kernel height, kernel width, and input channels within group
    BLOCK_K_COUNT = (GROUP_IN_C + BLOCK_K - 1) // BLOCK_K

    for ijk in range(KERNEL_H * KERNEL_W * BLOCK_K_COUNT):
        k_blk = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        kh = ij // KERNEL_W
        kw = ij % KERNEL_W

        # Input spatial coordinates
        ih = idx_h * STRIDE_H + kh - PAD_H
        iw = idx_w * STRIDE_W + kw - PAD_W

        # Input channel indices within group
        idx_ic = tl.arange(0, BLOCK_K) + k_blk

        # Load X tile [BLOCK_M, BLOCK_K]
        # X is in NHWC layout: ptr = X_ptr + n*stride_xn + h*stride_xh + w*stride_xw + c*stride_xc
        # c = group * GROUP_IN_C + idx_ic
        x_ptrs = (X_ptr

        mask_x = (
            (offs_m[:, None] < M_TOTAL)
            & (ih[:, None] >= 0)
            & (iw[:, None] >= 0)
        )
        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Load W tile [BLOCK_K, BLOCK_N]
        # W shape: [OUT_C, GROUP_IN_C, KH, KW]
        # global out channel = group * GROUP_OUT_C + idx_oc
        w_ptrs = (W_ptr

        mask_w = (
            (idx_ic[:, None] < GROUP_IN_C)
        )
        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        acc = tl.dot(x_tile, w_tile, acc, allow_tf32=False)

    # Store output in NHWC layout
    # Out shape: [BATCH, OUT_H, OUT_W, OUT_C] in NHWC
    # global out channel = group * GROUP_OUT_C + idx_oc
    out_ptrs = (Out_ptr

    mask_out = (
        (offs_m[:, None] < M_TOTAL)
    )
    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def kernel_function(weight, x):
    # weight: [8, 1, 3, 1] = [OUT_C, GROUP_IN_C, KH, KW]
    # x: [64, 8, 512, 512] = [N, C, H, W]
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, GROUP_IN_C, KERNEL_H, KERNEL_W = weight.shape
    GROUPS = IN_C // GROUP_IN_C
    STRIDE_H, STRIDE_W = 1, 1
    PAD_H, PAD_W = 0, 0
    OUT_H = (IN_H + 2 * PAD_H - KERNEL_H) // STRIDE_H + 1
    OUT_W = (IN_W + 2 * PAD_W - KERNEL_W) // STRIDE_W + 1

    # Convert x to NHWC for coalesced access
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()

    # Output in NHWC layout, then convert back
    out_nhwc = torch.empty((BATCH, OUT_H, OUT_W, OUT_C), device=x.device, dtype=x.dtype)

    M_TOTAL = BATCH * OUT_H * OUT_W
    GROUP_OUT_C = OUT_C // GROUPS

    BLOCK_M = 256
    BLOCK_N = 16
    BLOCK_K = 16

    grid = lambda meta: (
        triton.cdiv(M_TOTAL, meta['BLOCK_M']),
        triton.cdiv(GROUP_OUT_C, meta['BLOCK_N']),
        GROUPS,
    )

    depthwise_conv2d_kernel[grid](
        x_nhwc, weight, out_nhwc,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KERNEL_H, KERNEL_W,
        STRIDE_H, STRIDE_W,
        PAD_H, PAD_W,
        GROUPS,
        # X strides in NHWC: [N, H, W, C]
        x_nhwc.stride(0), x_nhwc.stride(1), x_nhwc.stride(2), x_nhwc.stride(3),
        # W strides: [OUT_C, GROUP_IN_C, KH, KW]
        weight.stride(0), weight.stride(1), weight.stride(2), weight.stride(3),
        # Out strides in NHWC: [N, H, W, C]
        out_nhwc.stride(0), out_nhwc.stride(1), out_nhwc.stride(2), out_nhwc.stride(3),
    )

    # Convert back to NCHW
    out = out_nhwc.permute(0, 3, 1, 2).contiguous()
    return out


def get_inputs():
    weight = torch.rand(8, 1, 3, 1, device='cuda', dtype=torch.float32)
    x = torch.rand(64, 8, 512, 512, device='cuda', dtype=torch.float32)
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
        weight_ref = inputs[0]
        x_ref = inputs[1]
        import torch.nn.functional as F
        ref_out = F.conv2d(x_ref, weight_ref, bias=None, stride=1, padding=0, groups=8)
        max_diff = (output - ref_out).abs().max().item()
        print(f"Max difference from reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match.")
        print(f"Output shape: {output.shape}")