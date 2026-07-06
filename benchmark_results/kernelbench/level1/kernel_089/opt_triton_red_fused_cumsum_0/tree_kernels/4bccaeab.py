import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=2,  num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 2048}, num_warps=16, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def cumsum_kernel_v2(
    in_ptr0,
    out_ptr0,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    """
    Cumulative sum along the row dimension (axis=1).
    We iterate over R0_BLOCK-sized chunks along the column dimension,
    """
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    xmask = xindex < xnumel  # [XBLOCK, 1]

    r0_base = tl.arange(0, R0_BLOCK)[None, :]  # [1, R0_BLOCK]

    # Running carry for each row: shape [XBLOCK, 1]
    carry = tl.zeros([XBLOCK, 1], dtype=tl.float32)

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base  # [1, R0_BLOCK]
        r0_mask = r0_index < r0_numel   # [1, R0_BLOCK]

        # Load input tile [XBLOCK, R0_BLOCK]
        ptrs = in_ptr0 + (r0_index + r0_numel * xindex)
        ptrs = tl.multiple_of(ptrs, [1, 4])  # 4-float (128-bit) alignment along R0
        tmp = tl.load(ptrs, mask=xmask & r0_mask, other=0.0, eviction_policy='evict_first')
        tmp = tmp.to(tl.float32)

        # Prefix scan within this tile along axis=1
        scanned, = tl.associative_scan((tmp,), 1, _add_combine)

        # Add carry from previous tiles
        result = scanned + carry

        # Update carry: last element of each row in this tile
        # We want the value at column R0_BLOCK-1 (or last valid column)
        # Broadcast: select last column
        last_mask = (r0_base == (R0_BLOCK - 1)) & r0_mask  # [1, R0_BLOCK]
        # Sum over R0_BLOCK dimension to get [XBLOCK, 1]
        new_carry = tl.sum(tl.where(last_mask, result, tl.zeros([XBLOCK, R0_BLOCK], dtype=tl.float32)), axis=1, keep_dims=True)

        # For partial last block, we need the actual last valid element
        # Compute last valid index in this chunk
        chunk_end = r0_offset + R0_BLOCK
        if chunk_end > r0_numel:
            # Partial block: find last valid
            last_valid = r0_numel - r0_offset - 1
            last_mask_partial = (r0_base == last_valid) & r0_mask
            new_carry = tl.sum(tl.where(last_mask_partial, result, tl.zeros([XBLOCK, R0_BLOCK], dtype=tl.float32)), axis=1, keep_dims=True)

        carry = new_carry

        # Store output
        out_ptrs = out_ptr0 + (r0_index + r0_numel * xindex)
        out_ptrs = tl.multiple_of(out_ptrs, [1, 4])
        tl.store(out_ptrs, result, mask=xmask & r0_mask)


@triton.jit

def _add_combine(a, b):
    return a + b


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 1024}, num_warps=16, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def cumsum_kernel_v2_v2(
    in_ptr0,
    out_ptr0,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    """
    Cumulative sum along the row dimension (axis=1).
    We iterate over R0_BLOCK-sized chunks along the column dimension,
    """
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]  # [XBLOCK, 1]
    xmask = xindex < xnumel  # [XBLOCK, 1]

    r0_base = tl.arange(0, R0_BLOCK)[None, :]  # [1, R0_BLOCK]

    # Running carry for each row: shape [XBLOCK, 1]
    carry = tl.zeros([XBLOCK, 1], dtype=tl.float32)

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base  # [1, R0_BLOCK]
        r0_mask = r0_index < r0_numel   # [1, R0_BLOCK]

        # Load input tile [XBLOCK, R0_BLOCK]
        ptrs = in_ptr0 + (r0_index + r0_numel * xindex)
        tmp = tl.load(ptrs, mask=xmask & r0_mask, other=0.0, eviction_policy='evict_first')
        tmp = tmp.to(tl.float32)

        # Prefix scan within this tile along axis=1
        scanned, = tl.associative_scan((tmp,), 1, _add_combine)

        # Add carry from previous tiles
        result = scanned + carry

        # Update carry: last element of each row in this tile
        # We want the value at column R0_BLOCK-1 (or last valid column)
        # Broadcast: select last column
        last_mask = (r0_base == (R0_BLOCK - 1)) & r0_mask  # [1, R0_BLOCK]
        # Sum over R0_BLOCK dimension to get [XBLOCK, 1]
        new_carry = tl.sum(tl.where(last_mask, result, tl.zeros([XBLOCK, R0_BLOCK], dtype=tl.float32)), axis=1, keep_dims=True)

        # For partial last block, we need the actual last valid element
        # Compute last valid index in this chunk
        chunk_end = r0_offset + R0_BLOCK
        if chunk_end > r0_numel:
            # Partial block: find last valid
            last_valid = r0_numel - r0_offset - 1
            last_mask_partial = (r0_base == last_valid) & r0_mask
            new_carry = tl.sum(tl.where(last_mask_partial, result, tl.zeros([XBLOCK, R0_BLOCK], dtype=tl.float32)), axis=1, keep_dims=True)

        carry = new_carry

        # Store output
        out_ptrs = out_ptr0 + (r0_index + r0_numel * xindex)
        tl.store(out_ptrs, result, mask=xmask & r0_mask)


def kernel_function(x):
    """
    Compute cumulative sum along dim=1 for a 2D tensor.
    """
    assert x.ndim == 2, "Only 2D tensors supported"
    assert x.is_cuda, "Input must be on CUDA"

    rows, cols = x.shape
    out = torch.empty_like(x)

    xnumel = rows
    r0_numel = cols

    def grid(meta):
        return (triton.cdiv(xnumel, meta['XBLOCK']),)

    cumsum_kernel_v2_v2[grid](
        x,
        out,
        xnumel,
        r0_numel,
    )

    return out


def get_inputs():
    return [torch.rand(32768, 32768, device='cuda', dtype=torch.float32)]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    x = inputs[0]

    # Verify correctness on a small example first
    x_small = torch.rand(4, 8, device='cuda', dtype=torch.float32)
    ref_small = torch.cumsum(x_small, dim=1)
    out_small = kernel_function(x_small)
    max_err = (ref_small - out_small).abs().max().item()
    print(f"Small test max error: {max_err:.6f}")
    assert max_err < 1e-3, f"Correctness check failed: max_err={max_err}"

    # Run on full input
    output = kernel_function(x)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify against PyTorch
        ref = torch.cumsum(x, dim=1)
        max_err = (ref - output).abs().max().item()
        print(f"Full test max error: {max_err:.6f}")
        assert max_err < 1e-3, f"Correctness check failed: max_err={max_err}"
        print("Kernel executed successfully.")