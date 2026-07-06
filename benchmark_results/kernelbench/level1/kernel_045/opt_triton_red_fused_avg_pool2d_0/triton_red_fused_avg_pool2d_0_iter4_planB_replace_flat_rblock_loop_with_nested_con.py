import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 32},  num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 64},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 256}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 256}, num_warps=16, num_stages=4),
    ],
    key=['xnumel'],
)
@triton.jit
def avg_pool2d_kernel(
    in_ptr, out_ptr,
    B, C, H, W,
    OH, OW,
    KH, KW,
    SH, SW,
    xnumel,
    XBLOCK: tl.constexpr,
    KH_CONST: tl.constexpr,
    KW_CONST: tl.constexpr,
):
    # Each program handles XBLOCK output elements
    xoffset = tl.program_id(0).to(tl.int64) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None].to(tl.int64)
    xmask = xindex < xnumel

    # Decode output indices
    ow = xindex % OW
    oh = (xindex // OW) % OH
    c  = (xindex // (OW * OH)) % C
    b  = xindex // (OW * OH * C)

    # Hoist invariant terms before the loop (shapes [XBLOCK, 1])
    batch_chan_base = b * (C * H * W) + c * (H * W)  # [XBLOCK, 1]
    oh_base = oh * SH   # [XBLOCK, 1]
    ow_base = ow * SW   # [XBLOCK, 1]

    # Accumulator — [XBLOCK, 1], no RBLOCK dimension needed
    acc = tl.zeros([XBLOCK, 1], dtype=tl.float32)

    for kh in tl.static_range(KH_CONST):            # fully unrolled at compile time
        ih_stride = (oh_base + kh) * W              # hoisted from inner loop
        for kw in tl.static_range(KW_CONST):        # fully unrolled at compile time
            in_idx = batch_chan_base + ih_stride + ow_base + kw
            val = tl.load(in_ptr + in_idx, mask=xmask, other=0.0)
            acc += val                               # FMA-fusable, no integer division

    result = acc * (1.0 / (KH_CONST * KW_CONST))
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