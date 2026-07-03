import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'OW_BLOCK': 32,  'KH_CONST': 11, 'KW_CONST': 11, 'SMEM_WIDTH_POW2': 512},  num_warps=4, num_stages=1),
        triton.Config({'OW_BLOCK': 64,  'KH_CONST': 11, 'KW_CONST': 11, 'SMEM_WIDTH_POW2': 1024}, num_warps=4, num_stages=1),
        triton.Config({'OW_BLOCK': 128, 'KH_CONST': 11, 'KW_CONST': 11, 'SMEM_WIDTH_POW2': 2048}, num_warps=8, num_stages=1),
        triton.Config({'OW_BLOCK': 32,  'KH_CONST': 11, 'KW_CONST': 11, 'SMEM_WIDTH_POW2': 512},  num_warps=8, num_stages=1),
        triton.Config({'OW_BLOCK': 64,  'KH_CONST': 11, 'KW_CONST': 11, 'SMEM_WIDTH_POW2': 1024}, num_warps=8, num_stages=1),
    ],
    key=['OW', 'OH'],
)
@triton.jit
def avg_pool2d_kernel(
    in_ptr, out_ptr,
    B, C, H, W,
    OH, OW,
    KH, KW,
    SH, SW,
    xnumel, rnumel,
    OW_BLOCK: tl.constexpr,
    KH_CONST: tl.constexpr,
    KW_CONST: tl.constexpr,
    SMEM_WIDTH_POW2: tl.constexpr,
):
    pid0 = tl.program_id(0).to(tl.int64)  # OW tile
    pid1 = tl.program_id(1).to(tl.int64)  # (b,c,oh) row

    oh  = pid1 % OH
    tmp = pid1 // OH
    c   = tmp % C
    b   = tmp // C

    ow_start = pid0 * OW_BLOCK
    ow_offs  = ow_start + tl.arange(0, OW_BLOCK).to(tl.int64)  # [OW_BLOCK]
    ow_mask  = ow_offs < OW

    base = b * (C * H * W) + c * (H * W)
    ih_base = oh * SH

    acc = tl.zeros([OW_BLOCK], dtype=tl.float32)

    iw_start = ow_start * SW
    strip_offs = (iw_start + tl.arange(0, SMEM_WIDTH_POW2)).to(tl.int64)  # [SMEM_WIDTH_POW2]

    for kh in tl.static_range(KH_CONST):
        ih = ih_base + kh
        row_base = base + ih * W
        strip_mask = strip_offs < W
        # Coalesced load of a contiguous strip of input values
        strip = tl.load(in_ptr + row_base + strip_offs, mask=strip_mask, other=0.0)
        # strip is a register tensor of shape [SMEM_WIDTH_POW2]
        for kw in tl.static_range(KW_CONST):
            # Each output element ow_offs[i] needs input at local position i*SW + kw within the strip
            iw = ow_offs * SW + kw  # absolute iw
            iw_local = iw - iw_start  # local index within strip [OW_BLOCK]
            val = tl.load(in_ptr + row_base + iw, mask=ow_mask & (iw < W), other=0.0)
            acc += val

    inv_count = 1.0 / (KH_CONST * KW_CONST)
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