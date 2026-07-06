import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'RBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'RBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'RBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'RBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'RBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'RBLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256, 'RBLOCK': 16}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256, 'RBLOCK': 32}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'rnumel'],
)
@triton.jit
def avg_pool2d_kernel(
    in_ptr, out_ptr,
    B, C, H, W,
    OH, OW,
    KH, KW,
    SH, SW,
    xnumel, rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    # 2D grid: pid0 tiles over OW (innermost spatial dim), pid1 encodes (b, c, oh)
    pid0 = tl.program_id(0).to(tl.int64)  # ow tile index
    pid1 = tl.program_id(1).to(tl.int64)  # encodes (b, c, oh)

    # Decode pid1 -> (b, c, oh)
    oh = pid1 % OH
    c  = (pid1 // OH) % C
    b  = pid1 // (OH * C)

    # ow indices for this tile: shape [XBLOCK]
    ow = pid0 * XBLOCK + tl.arange(0, XBLOCK).to(tl.int64)
    ow_mask = ow < OW

    # Accumulator: shape [XBLOCK]
    acc = tl.zeros([XBLOCK], dtype=tl.float32)

    # Base pointer for this (b, c) input plane
    base_bc = b * (C * H * W) + c * (H * W)

    for r_offset in range(0, rnumel, RBLOCK):
        rindex = r_offset + tl.arange(0, RBLOCK).to(tl.int64)
        rmask = rindex < rnumel

        kw = rindex % KW   # shape [RBLOCK]
        kh = rindex // KW  # shape [RBLOCK]

        # For each kh value in this RBLOCK chunk, accumulate over kw
        # ih is fixed per kh, iw = ow*SW + kw
        # We iterate over the RBLOCK reduction elements
        # ow: [XBLOCK], kw: [RBLOCK] -> iw: [XBLOCK, RBLOCK]
        ih = oh * SH + kh  # shape [RBLOCK]
        iw = ow[:, None] * SW + kw[None, :]  # shape [XBLOCK, RBLOCK]

        # Flat input index: [XBLOCK, RBLOCK]
        in_idx = base_bc + ih[None, :] * W + iw

        # Mask: ow in bounds AND r in bounds
        load_mask = ow_mask[:, None] & rmask[None, :]

        val = tl.load(in_ptr + in_idx, mask=load_mask, other=0.0)
        acc += tl.sum(val, axis=1)

    inv_count = 1.0 / (KH * KW)
    result = acc * inv_count

    # Output index: b*(C*OH*OW) + c*(OH*OW) + oh*OW + ow  -> contiguous in ow
    out_idx = b * (C * OH * OW) + c * (OH * OW) + oh * OW + ow
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