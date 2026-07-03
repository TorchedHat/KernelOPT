import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=2, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 16}, num_stages=2, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'BLOCK_K': 16}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['BATCH', 'IN_C', 'IN_H', 'IN_W', 'OUT_C', 'OUT_H', 'OUT_W', 'KH', 'KW'],
)
@triton.jit
def conv2d_kernel(
    x_ptr, w_ptr, out_ptr,
    BATCH: tl.constexpr, IN_C: tl.constexpr, IN_H: tl.constexpr, IN_W: tl.constexpr,
    OUT_C: tl.constexpr, OUT_H: tl.constexpr, OUT_W: tl.constexpr,
    KH: tl.constexpr, KW: tl.constexpr,
    stride_xn: tl.constexpr, stride_xc: tl.constexpr, stride_xh: tl.constexpr, stride_xw: tl.constexpr,
    stride_woc: tl.constexpr, stride_wic: tl.constexpr, stride_wkh: tl.constexpr, stride_wkw: tl.constexpr,
    stride_on: tl.constexpr, stride_oc: tl.constexpr, stride_oh: tl.constexpr, stride_ow: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # program_id(0) -> tile over output spatial (N*OH*OW)
    # program_id(1) -> tile over output channels (OUT_C)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Each program handles BLOCK_M output pixels and BLOCK_N output channels
    nhw_start = pid_m * BLOCK_M
    oc_start = pid_n * BLOCK_N

    # Compute indices for the BLOCK_M output pixels
    nhw_offsets = nhw_start + tl.arange(0, BLOCK_M)
    oc_offsets = oc_start + tl.arange(0, BLOCK_N)

    # Decompose nhw into n, oh, ow
    ow_idx = nhw_offsets % OUT_W
    nh_idx = nhw_offsets // OUT_W
    oh_idx = nh_idx % OUT_H
    n_idx = nh_idx // OUT_H

    # Accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over kernel height, kernel width, and input channels in blocks
    BLOCK_K_COUNT = (IN_C + BLOCK_K - 1) // BLOCK_K
    total_iters = KH * KW * BLOCK_K_COUNT

    for ijk in range(total_iters):
        k_block = (ijk % BLOCK_K_COUNT) * BLOCK_K
        ij = ijk // BLOCK_K_COUNT
        kh_idx = ij // KW
        kw_idx = ij % KW

        # Input spatial positions
        ih_idx = oh_idx * 1 + kh_idx  # stride=1, padding=0
        iw_idx = ow_idx * 1 + kw_idx  # stride=1, padding=0

        # Input channel offsets for this block
        ic_offsets = k_block + tl.arange(0, BLOCK_K)

        # Load input tile: shape [BLOCK_M, BLOCK_K]
        x_ptrs = (x_ptr

        mask_x = ((n_idx[:, None] < BATCH)
                  & (ih_idx[:, None] >= 0)
                  & (iw_idx[:, None] >= 0)

        x_tile = tl.load(x_ptrs, mask=mask_x, other=0.0)

        # Load weight tile: shape [BLOCK_K, BLOCK_N]
        w_ptrs = (w_ptr

        mask_w = ((ic_offsets[:, None] < IN_C)

        w_tile = tl.load(w_ptrs, mask=mask_w, other=0.0)

        # Accumulate
        acc = tl.dot(x_tile, w_tile, acc=acc, allow_tf32=False)

    # Store output: shape [BATCH, OUT_C, OUT_H, OUT_W]
    out_ptrs = (out_ptr

    mask_out = ((n_idx[:, None] < BATCH)

    tl.store(out_ptrs, acc.to(tl.float32), mask=mask_out)


def kernel_function(weight, x):
    # weight: [OUT_C, IN_C, KH, KW]
    # x: [BATCH, IN_C, IN_H, IN_W]
    BATCH, IN_C, IN_H, IN_W = x.shape
    OUT_C, IN_C_w, KH, KW = weight.shape

    assert IN_C == IN_C_w, "Input channels must match"

    # Output dimensions (no padding, stride=1, dilation=1)
    OUT_H = IN_H - KH + 1  # 512 - 5 + 1 = 508
    OUT_W = IN_W - KW + 1  # 256 - 7 + 1 = 250

    # Allocate output
    out = torch.empty((BATCH, OUT_C, OUT_H, OUT_W), device=x.device, dtype=x.dtype)

    # Strides for x: [BATCH, IN_C, IN_H, IN_W] with standard layout
    stride_xn = x.stride(0)
    stride_xc = x.stride(1)
    stride_xh = x.stride(2)
    stride_xw = x.stride(3)

    # Strides for weight: [OUT_C, IN_C, KH, KW]
    stride_woc = weight.stride(0)
    stride_wic = weight.stride(1)
    stride_wkh = weight.stride(2)
    stride_wkw = weight.stride(3)

    # Strides for output: [BATCH, OUT_C, OUT_H, OUT_W]
    stride_on = out.stride(0)
    stride_oc = out.stride(1)
    stride_oh = out.stride(2)
    stride_ow = out.stride(3)

    # Total output pixels
    total_nhw = BATCH * OUT_H * OUT_W

    # Grid
    def grid(meta):
        return (
            triton.cdiv(total_nhw, meta['BLOCK_M']),
            triton.cdiv(OUT_C, meta['BLOCK_N']),
        )

    conv2d_kernel[grid](
        x, weight, out,
        BATCH, IN_C, IN_H, IN_W,
        OUT_C, OUT_H, OUT_W,
        KH, KW,
        stride_xn, stride_xc, stride_xh, stride_xw,
        stride_woc, stride_wic, stride_wkh, stride_wkw,
        stride_on, stride_oc, stride_oh, stride_ow,
    )

    return out


def get_inputs():
    weight = torch.randn(128, 64, 5, 7, device='cuda', dtype=torch.float32)
    x = torch.randn(8, 64, 512, 256, device='cuda', dtype=torch.float32)
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
        ref = F.conv2d(x, weight, bias=None, stride=(1, 1), padding=(0, 0))
        out = kernel_function(weight, x)
        print(f"Output shape: {out.shape}")
        print(f"Reference shape: {ref.shape}")
        max_diff = (out - ref).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match!")
    else:
        output = kernel_function(*inputs)
        torch.cuda.synchronize()
        print("Kernel executed successfully.")