import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'RBLOCK': 128, 'KH_CONST': 11, 'KW_CONST': 11}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256, 'RBLOCK': 128, 'KH_CONST': 11, 'KW_CONST': 11}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512, 'RBLOCK': 128, 'KH_CONST': 11, 'KW_CONST': 11}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024, 'RBLOCK': 128, 'KH_CONST': 11, 'KW_CONST': 11}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'RBLOCK': 128, 'KH_CONST': 11, 'KW_CONST': 11}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'RBLOCK': 128, 'KH_CONST': 11, 'KW_CONST': 11}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 256, 'RBLOCK': 128, 'KH_CONST': 11, 'KW_CONST': 11}, num_warps=8, num_stages=3),
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
    xnumel, rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
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

    # Hoist base address components out of reduction loops
    b_stride = b * (C * H * W)
    c_stride = c * (H * W)
    oh_base  = oh * SH   # [XBLOCK, 1], computed once
    ow_base  = ow * SW   # [XBLOCK, 1], computed once

    # 1D accumulator (no RBLOCK dimension needed)
    acc = tl.zeros([XBLOCK, 1], dtype=tl.float32)

    # Nested loops: NO integer division/modulo, fully unrolled at compile time
    for kh in tl.static_range(KH_CONST):       # 11 iters, unrolled
        ih = oh_base + kh                       # [XBLOCK, 1], kh is scalar
        ih_stride = ih * W                      # [XBLOCK, 1], hoisted from kw loop
        for kw in tl.static_range(KW_CONST):   # 11 iters, unrolled
            iw = ow_base + kw                   # [XBLOCK, 1], kw is scalar
            in_idx = b_stride + c_stride + ih_stride + iw
            val = tl.load(in_ptr + in_idx, mask=xmask, other=0.0)
            acc += val                          # FMA-fusable

    inv_count = 1.0 / (KH_CONST * KW_CONST)
    result = acc * inv_count

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