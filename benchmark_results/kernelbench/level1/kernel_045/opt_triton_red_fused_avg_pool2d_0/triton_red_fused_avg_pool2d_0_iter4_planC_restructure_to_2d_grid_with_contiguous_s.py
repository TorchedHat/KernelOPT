import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'OW_BLOCK': 16, 'STRIP_W': 256},  num_warps=4, num_stages=2),
        triton.Config({'OW_BLOCK': 16, 'STRIP_W': 256},  num_warps=8, num_stages=3),
        triton.Config({'OW_BLOCK': 32, 'STRIP_W': 512},  num_warps=4, num_stages=2),
        triton.Config({'OW_BLOCK': 32, 'STRIP_W': 512},  num_warps=8, num_stages=3),
        triton.Config({'OW_BLOCK': 64, 'STRIP_W': 1024}, num_warps=4, num_stages=2),
        triton.Config({'OW_BLOCK': 64, 'STRIP_W': 1024}, num_warps=8, num_stages=3),
    ],
    key=['OW', 'OH', 'B', 'C'],
)
@triton.jit
def avg_pool2d_kernel(
    in_ptr, out_ptr,
    B, C, H, W,
    OH, OW,
    KH, KW,
    SH, SW,
    OW_BLOCK: tl.constexpr,
    STRIP_W:  tl.constexpr,
):
    pid0 = tl.program_id(0).to(tl.int64)  # OW tile index
    pid1 = tl.program_id(1).to(tl.int64)  # B*C*OH row index

    # Decode pid1 -> (b, c, oh)
    oh  = pid1 % OH
    tmp = pid1 // OH
    c   = tmp % C
    b   = tmp // C

    # Output column range for this CTA
    ow_start = pid0 * OW_BLOCK
    ow_offs  = ow_start + tl.arange(0, OW_BLOCK)  # shape [OW_BLOCK]
    ow_mask  = ow_offs < OW

    acc = tl.zeros([OW_BLOCK], dtype=tl.float32)

    row_base = b * (C * H * W) + c * (H * W)
    iw_base  = ow_start * SW  # first input column for this CTA

    # Contiguous strip offsets [0, 1, ..., STRIP_W-1]
    strip_offs = tl.arange(0, STRIP_W)  # shape [STRIP_W]

    for kh in tl.static_range(11):  # KH=11, fully unrolled
        ih = oh * SH + kh
        strip_base = row_base + ih * W + iw_base
        strip_mask = (iw_base + strip_offs) < W  # right-boundary guard
        # Single coalesced load: STRIP_W contiguous floats
        strip = tl.load(in_ptr + strip_base + strip_offs,
                        mask=strip_mask, other=0.0)  # shape [STRIP_W]

        for kw in tl.static_range(11):  # KW=11, fully unrolled
            # For output column ow_offs[i], the strip offset is:
            # (ow_offs[i] - ow_start) * SW + kw = tl.arange(0,OW_BLOCK)*SW + kw
            gather_idx = tl.arange(0, OW_BLOCK) * SW + kw  # shape [OW_BLOCK]
            # gather_idx < STRIP_W by construction (verified above)
            acc += tl.where(ow_mask, strip[gather_idx], 0.0)

    inv_count = 1.0 / (KH * KW)  # = 1/121
    result = acc * inv_count

    out_idx = b * (C * OH * OW) + c * (OH * OW) + oh * OW + ow_offs
    tl.store(out_ptr + out_idx, result, mask=ow_mask)

def kernel_function(x):
    B, C, H, W = x.shape
    KH, KW = 11, 11
    SH, SW = 11, 11
    pad_h, pad_w = 0, 0

    OH = (H + 2 * pad_h - KH) // SH + 1
    OW = (W + 2 * pad_w - KW) // SW + 1

    out = torch.empty((B, C, OH, OW), dtype=x.dtype, device=x.device)

    xnumel = B * C * OH * OW
    rnumel = KH * KW

    grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)

    avg_pool2d_kernel[grid](
        x, out,
        B, C, H, W,
        OH, OW,
        KH, KW,
        SH, SW,
        xnumel, rnumel,
    )

    return out


def get_inputs():
    x = torch.rand(16, 64, 2048, 2048, device='cuda', dtype=torch.float32)
    return [x]


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
        import torch.nn as nn
        ref_pool = nn.AvgPool2d(kernel_size=11, stride=11, padding=0)
        ref_out = ref_pool(inputs[0])
        max_diff = (output - ref_out).abs().max().item()
        print(f"Max difference from reference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match.")