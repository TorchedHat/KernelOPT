import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 32}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 64}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 32}, num_warps=8, num_stages=3),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def max_dim1_kernel(
    in_ptr0,
    out_ptr0,
    xnumel,
    r0_numel,
    XBLOCK: tl.constexpr,
    R0_BLOCK: tl.constexpr,
):
    # 2D grid: program_id(0) covers dim2 tiles, program_id(1) covers batch
    # xnumel = dim2 = 4095, batch = 128 (derived as xnumel_total/xnumel in kernel_function)
    x0_tile = tl.program_id(0)
    x1 = tl.program_id(1)  # batch index

    # x0_offsets: contiguous dim2 indices (stride=1 in memory)
    x0_offsets = x0_tile * XBLOCK + tl.arange(0, XBLOCK)
    x0_mask = x0_offsets < xnumel  # xnumel = dim2 = 4095

    # Initialize accumulator with -inf for each x0 element
    _max_val = tl.full([XBLOCK], float("-inf"), tl.float32)

    for r0_off in tl.range(0, r0_numel, R0_BLOCK):
        r0_idx = r0_off + tl.arange(0, R0_BLOCK)
        r0_mask = r0_idx < r0_numel

        # Load shape [XBLOCK, R0_BLOCK]:
        # ptrs = in_ptr0 + x1*16773120 + r0_idx[None,:]*4095 + x0_offsets[:,None]
        # Adjacent threads differ in x0_offsets (stride=1) -> fully coalesced
        ptrs = in_ptr0 + (x1 * 16773120 + r0_idx[None, :] * 4095 + x0_offsets[:, None])
        combined_mask = x0_mask[:, None] & r0_mask[None, :]
        tmp = tl.load(
            ptrs,
            combined_mask,
            eviction_policy='evict_first',
            other=float("-inf"),
        )
        # Reduce over R0_BLOCK dimension to get per-x0 max
        tile_max = tl.max(tmp, axis=1)
        _max_val = tl.maximum(_max_val, tile_max)

    # Store result: output shape is [batch, xnumel] = [128, 4095]
    # xnumel here is dim2=4095, x1 is batch index
    out_offsets = x1 * xnumel + x0_offsets
    tl.store(out_ptr0 + out_offsets, _max_val, x0_mask)

def kernel_function(x):
    # x: [128, 4096, 4095], reduce along dim=1
    assert x.is_cuda and x.dtype == torch.float32
    assert x.shape == (128, 4096, 4095)

    batch_size = x.shape[0]
    dim1 = x.shape[1]  # reduction dimension = 4096
    dim2 = x.shape[2]  # 4095

    xnumel = batch_size * dim2  # 128 * 4095 = 524160
    r0_numel = dim1              # 4096

    out = torch.empty((batch_size, dim2), dtype=torch.float32, device=x.device)

    def grid(meta):
        return (triton.cdiv(xnumel, meta['XBLOCK']),)

    max_dim1_kernel[grid](
        x,
        out,
        xnumel,
        r0_numel,
    )

    return out


def get_inputs():
    x = torch.rand(128, 4096, 4095, device='cuda')
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
        ref_output = torch.max(inputs[0], dim=1)[0]
        max_diff = (output - ref_output).abs().max().item()
        print(f"Max difference from reference: {max_diff}")
        assert max_diff < 1e-3, f"Correctness check failed! Max diff: {max_diff}"
        print("Kernel executed successfully.")
        print(f"Output shape: {output.shape}")
        print(f"Output dtype: {output.dtype}")