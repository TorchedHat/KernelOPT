import torch
import triton
import triton.language as tl


@triton.jit
def _add_combine(a, b):
    return a + b


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 8192}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 16384}, num_warps=32, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def masked_cumsum_kernel(
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    # Each program handles XBLOCK rows
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel

    r0_base = tl.arange(0, R0_BLOCK)[None, :]

    # carry accumulator for multi-block scan
    carry = tl.zeros([XBLOCK, 1], dtype=tl.float32)
    first_block = True

    for r0_offset in range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel

        # Load x and mask
        x_val = tl.load(
            in_ptr0 + (r0_index + r0_numel * xindex),
            mask=xmask & r0_mask,
            other=0.0,
            eviction_policy='evict_first',
        )
        mask_val = tl.load(
            in_ptr1 + (r0_index + r0_numel * xindex),
            mask=xmask & r0_mask,
            other=0,
            eviction_policy='evict_first',
        ).to(tl.int1)

        # Apply mask: zero out elements where mask is False
        masked_x = tl.where(mask_val, x_val, tl.zeros_like(x_val))
        masked_x = masked_x.to(tl.float32)

        # Broadcast to [XBLOCK, R0_BLOCK]
        tile = tl.broadcast_to(masked_x, [XBLOCK, R0_BLOCK])

        # Associative scan (prefix sum) within this block
        scanned = tl.associative_scan(tile, 1, _add_combine)

        # Add carry from previous blocks
        result = scanned + carry

        # Update carry: last valid element in each row
        # Get the last element of the scan for this block
        last_col = tl.minimum(r0_numel - r0_offset, R0_BLOCK) - 1
        # We need the carry to be the sum up to the end of this block
        # Use the scanned value at the last valid position
        r0_last = tl.arange(0, R0_BLOCK)[None, :]
        is_last = (r0_last == last_col)
        # Extract last element per row
        last_val = tl.sum(tl.where(is_last, result, tl.zeros_like(result)), axis=1, keep_dims=True)
        carry = last_val

        # Store results
        tl.store(
            out_ptr0 + (r0_index + r0_numel * xindex),
            result.to(tl.float32),
            mask=xmask & r0_mask,
        )


def kernel_function(x, mask):
    assert x.is_cuda and mask.is_cuda
    assert x.dtype == torch.float32
    assert mask.dtype == torch.bool

    rows, cols = x.shape
    output = torch.empty_like(x)

    # Grid: one program per XBLOCK rows
    def grid(meta):
        return (triton.cdiv(rows, meta['XBLOCK']),)

    masked_cumsum_kernel[grid](
        x, mask, output,
        rows, cols,
    )

    return output


def get_inputs():
    x = torch.rand(32768, 32768, device='cuda', dtype=torch.float32)
    mask = torch.randint(0, 2, x.shape, device='cuda').bool()
    return [x, mask]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    x, mask = inputs

    # Verify correctness on a small example first
    x_small = torch.rand(4, 8, device='cuda', dtype=torch.float32)
    mask_small = torch.randint(0, 2, (4, 8), device='cuda').bool()
    ref = torch.cumsum(x_small * mask_small, dim=1)
    out = kernel_function(x_small, mask_small)
    max_diff = (ref - out).abs().max().item()
    print(f"Small test max diff: {max_diff}")
    assert max_diff < 1e-3, f"Correctness check failed: max_diff={max_diff}"

    # Full size test
    output = kernel_function(*inputs)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify against PyTorch
        ref_output = torch.cumsum(x * mask, dim=1)
        max_diff = (ref_output - output).abs().max().item()
        print(f"Full test max diff: {max_diff}")
        assert max_diff < 1e-3, f"Correctness check failed: max_diff={max_diff}"
        print("Kernel executed successfully.")