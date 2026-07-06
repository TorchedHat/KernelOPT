import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK_OW': 16,  'WIN_W_POW2': 256,  'KH_CONST': 11, 'KW_CONST': 11}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK_OW': 32,  'WIN_W_POW2': 512,  'KH_CONST': 11, 'KW_CONST': 11}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK_OW': 64,  'WIN_W_POW2': 1024, 'KH_CONST': 11, 'KW_CONST': 11}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK_OW': 128, 'WIN_W_POW2': 2048, 'KH_CONST': 11, 'KW_CONST': 11}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK_OW': 32,  'WIN_W_POW2': 512,  'KH_CONST': 11, 'KW_CONST': 11}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK_OW': 64,  'WIN_W_POW2': 1024, 'KH_CONST': 11, 'KW_CONST': 11}, num_warps=8, num_stages=3),
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
    XBLOCK_OW: tl.constexpr,
    WIN_W_POW2: tl.constexpr,
    KH_CONST: tl.constexpr,
    KW_CONST: tl.constexpr,
):
    # WIN_W: actual number of input columns needed per CTA (not power-of-2)
    WIN_W: tl.constexpr = XBLOCK_OW * 11 + 10  # = XBLOCK_OW * SW + KW - 1

    pid0 = tl.program_id(0).to(tl.int64)  # OW tile index
    pid1 = tl.program_id(1).to(tl.int64)  # (b, c, oh) row index

    # Decode (b, c, oh) from pid1
    oh = pid1 % OH
    c  = (pid1 // OH) % C
    b  = pid1 // (OH * C)

    # OW tile: each thread handles one output column
    ow_start = pid0 * XBLOCK_OW
    ow_idx   = ow_start + tl.arange(0, XBLOCK_OW)  # [XBLOCK_OW]
    ow_mask  = ow_idx < OW

    # Base address for this (b, c) channel
    row_base = b * (C * H * W) + c * (H * W)

    acc = tl.zeros([XBLOCK_OW], dtype=tl.float32)

    for kh in tl.static_range(KH_CONST):
        ih = oh * SH + kh
        # Load a contiguous window of WIN_W_POW2 input columns — COALESCED
        # iw_base is the first input column index for this CTA
        iw_base  = ow_start * SW  # scalar: first input col for this CTA
        iw_range = iw_base + tl.arange(0, WIN_W_POW2)  # [WIN_W_POW2] contiguous addresses
        # Mask: only load within valid input bounds and within actual WIN_W
        load_mask = (iw_range < W) & (ih < H) & (tl.arange(0, WIN_W_POW2) < WIN_W)
        row_vals = tl.load(in_ptr + row_base + ih * W + iw_range,
                           mask=load_mask, other=0.0)  # COALESCED load

        # Gather each thread's KW values from the register-resident row
        for kw in tl.static_range(KW_CONST):
            # For thread i: offset into row_vals = i * SW + kw
            for i in tl.static_range(XBLOCK_OW):
                acc[i] += row_vals[i * 11 + kw]

    out_idx = b * (C * OH * OW) + c * (OH * OW) + oh * OW + ow_idx
    tl.store(out_ptr + out_idx, acc * (1.0 / (KH_CONST * KW_CONST)), mask=ow_mask)  # COALESCED

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