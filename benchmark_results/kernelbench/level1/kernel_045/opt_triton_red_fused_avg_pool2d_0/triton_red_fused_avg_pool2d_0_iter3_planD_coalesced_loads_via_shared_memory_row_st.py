import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'OW_TILE': 8},  num_warps=4, num_stages=2),
        triton.Config({'OW_TILE': 8},  num_warps=8, num_stages=2),
        triton.Config({'OW_TILE': 16}, num_warps=4, num_stages=2),
        triton.Config({'OW_TILE': 16}, num_warps=8, num_stages=2),
        triton.Config({'OW_TILE': 32}, num_warps=4, num_stages=2),
        triton.Config({'OW_TILE': 32}, num_warps=8, num_stages=2),
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
    xnumel, rnumel,
    OW_TILE: tl.constexpr,
):
    # 2D grid: pid0 over OW tiles, pid1 over B*C*OH
    pid_ow = tl.program_id(0)
    pid_row = tl.program_id(1)

    # Decode pid_row -> (b, c, oh)
    oh = pid_row % OH
    tmp = pid_row // OH
    c = tmp % C
    b = tmp // C

    # Output column offsets for this tile
    ow_base = pid_ow * OW_TILE
    ow_offs = ow_base + tl.arange(0, OW_TILE)
    ow_mask = ow_offs < OW

    # Input base column for this tile
    iw_base = ow_base * 11  # SW=11 always

    # SHMEM_W = OW_TILE * SW + KW - 1 = OW_TILE * 11 + 10
    # For OW_TILE=8:  SHMEM_W=98
    # For OW_TILE=16: SHMEM_W=186
    # For OW_TILE=32: SHMEM_W=362
    SHMEM_W: tl.constexpr = OW_TILE * 11 + 10

    # Accumulator for OW_TILE output elements
    acc = tl.zeros([OW_TILE], dtype=tl.float32)

    # Shared memory offsets (1D range over SHMEM_W)
    shmem_offs = tl.arange(0, SHMEM_W)

    # Loop over KH=11 rows (compile-time unrolled)
    for kh in tl.static_range(11):
        ih = oh * 11 + kh  # SH=11 always; ih always valid (no padding)

        # Pointer to start of this input row segment
        row_ptr = in_ptr + b.to(tl.int64) * (C * H * W) + c.to(tl.int64) * (H * W) + ih.to(tl.int64) * W + iw_base

        # Coalesced load of full row segment into registers (acts as shared memory)
        shmem_mask = (iw_base + shmem_offs) < W
        row_vals = tl.load(row_ptr + shmem_offs, mask=shmem_mask, other=0.0)

        # Loop over KW=11 columns (compile-time unrolled)
        for kw in tl.static_range(11):
            # Index into row_vals: for each output ow_offs[i], the input col is
            # (ow_offs[i] - ow_base)*SW + kw = (ow_offs[i] - ow_base)*11 + kw
            shmem_idx = (ow_offs - ow_base) * 11 + kw  # shape [OW_TILE], values in [0, SHMEM_W)
            # Gather from row_vals using shmem_idx
            val = tl.gather(row_vals, shmem_idx, axis=0)
            acc += tl.where(ow_mask, val, 0.0)

    # Average: divide by KH*KW = 121
    result = acc * (1.0 / 121)

    # Store results
    out_offs = b.to(tl.int64) * (C * OH * OW) + c.to(tl.int64) * (OH * OW) + oh.to(tl.int64) * OW + ow_offs
    tl.store(out_ptr + out_offs, result, mask=ow_mask)

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