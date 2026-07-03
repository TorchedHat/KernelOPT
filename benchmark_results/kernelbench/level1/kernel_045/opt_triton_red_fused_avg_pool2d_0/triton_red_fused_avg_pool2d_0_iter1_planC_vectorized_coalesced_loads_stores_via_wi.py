import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'RBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'RBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'RBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'RBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'RBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'RBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'RBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'RBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16,  'RBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,   'RBLOCK': 128}, num_warps=4, num_stages=2),
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
    # Each program handles XBLOCK output elements
    # Reduction over KH*KW elements
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = xindex < xnumel

    # Decode output indices
    ow = xindex % OW
    oh = (xindex // OW) % OH
    c  = (xindex // (OW * OH)) % C
    b  = xindex // (OW * OH * C)

    # Accumulator
    acc = tl.zeros([XBLOCK, RBLOCK], dtype=tl.float32)

    for r_offset in range(0, rnumel, RBLOCK):
        rindex = r_offset + tl.arange(0, RBLOCK)[None, :].to(tl.int64)
        rmask = rindex < rnumel

        kw = rindex % KW
        kh = rindex // KW

        ih = oh * SH + kh
        iw = ow * SW + kw

        # Compute flat input index
        in_idx = b * (C * H * W) + c * (H * W) + ih * W + iw

        val = tl.load(in_ptr + in_idx, mask=(xmask & rmask), other=0.0)
        acc += val

    total = tl.sum(acc, axis=1)[:, None]
    inv_count = 1.0 / (KH * KW)
    result = total * inv_count

    tl.store(out_ptr + xindex, result, mask=xmask)

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