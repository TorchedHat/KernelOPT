import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK_OW': 8,  'KH_CONST': 11, 'KW_CONST': 11, 'SW_CONST': 11, 'STRIP_W_POW2': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK_OW': 16, 'KH_CONST': 11, 'KW_CONST': 11, 'SW_CONST': 11, 'STRIP_W_POW2': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK_OW': 32, 'KH_CONST': 11, 'KW_CONST': 11, 'SW_CONST': 11, 'STRIP_W_POW2': 512}, num_warps=8, num_stages=2),
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
    XBLOCK_OW: tl.constexpr,
    KH_CONST: tl.constexpr,
    KW_CONST: tl.constexpr,
    SW_CONST: tl.constexpr,
    STRIP_W_POW2: tl.constexpr,
):
    # STRIP_W = XBLOCK_OW * SW_CONST + KW_CONST - 1 (actual width needed)
    # STRIP_W_POW2 >= STRIP_W, rounded up to power of 2 for tl.arange
    STRIP_W: tl.constexpr = XBLOCK_OW * SW_CONST + KW_CONST - 1

    pid_ow  = tl.program_id(0).to(tl.int64)
    pid_row = tl.program_id(1).to(tl.int64)

    # Decode (b, c, oh) from pid_row
    oh = pid_row % OH
    c  = (pid_row // OH) % C
    b  = pid_row // (OH * C)

    ow_base = pid_ow * XBLOCK_OW
    iw_base = ow_base * SW_CONST

    b_stride = b * (C * H * W)
    c_stride = c * (H * W)

    acc = tl.zeros([XBLOCK_OW], dtype=tl.float32)

    for kh in tl.static_range(KH_CONST):
        ih = oh * SH + kh
        row_base = b_stride + c_stride + ih * W + iw_base
        strip_offsets = tl.arange(0, STRIP_W_POW2)
        strip_mask = strip_offsets < STRIP_W
        strip_mask = strip_mask & ((iw_base + strip_offsets) < W)
        strip = tl.load(in_ptr + row_base + strip_offsets, mask=strip_mask, other=0.0)
        for kw in tl.static_range(KW_CONST):
            ow_local = tl.arange(0, XBLOCK_OW)
            gather_idx = ow_local * SW_CONST + kw
            acc += strip[gather_idx]

    inv_count = 1.0 / (KH_CONST * KW_CONST)
    result = acc * inv_count

    ow_out = ow_base + tl.arange(0, XBLOCK_OW)
    out_idx = b * (C * OH * OW) + c * (OH * OW) + oh * OW + ow_out
    out_mask = ow_out < OW
    tl.store(out_ptr + out_idx, result, mask=out_mask)

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