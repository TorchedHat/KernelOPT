import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128,  'R0_BLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256,  'R0_BLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512,  'R0_BLOCK': 16},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024, 'R0_BLOCK': 8},   num_warps=8, num_stages=2),
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
    # NEW layout: each program covers a contiguous chunk of x0 (dim2, stride=1)
    # for a fixed x1 (batch) index, and loops over r0 in R0_BLOCK chunks.
    # This makes adjacent warp threads access adjacent x0 values (stride=1),
    # enabling fully coalesced LDG.128 loads.
    pid = tl.program_id(0)
    num_x0_programs = tl.cdiv(4095, XBLOCK)
    x1_index = pid // num_x0_programs   # batch index (scalar per program)
    x0_offset = (pid % num_x0_programs) * XBLOCK

    # 1-D range over x0 (dim2, stride=1 in memory)
    x0_range = x0_offset + tl.arange(0, XBLOCK)  # [XBLOCK]
    x0_mask = x0_range < 4095

    # Apply contiguity hints to 1-D x0_range BEFORE broadcasting to 2-D
    # This tells Triton the offsets are contiguous and 64-aligned -> LDG.128
    # 64 divides all XBLOCK values (min XBLOCK=64), enables full LDG.128 vectorization
    x0_hints = tl.max_contiguous(tl.multiple_of(x0_range, 64), 64)  # [XBLOCK]

    # Initialize accumulator with -inf, shape [XBLOCK]
    _max_val = tl.full([XBLOCK], float('-inf'), tl.float32)

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_range = r0_offset + tl.arange(0, R0_BLOCK)  # [R0_BLOCK]
        r0_mask = r0_range < r0_numel

        # 2-D tile: [R0_BLOCK, XBLOCK], x0 is fast (stride-1 in memory)
        # ptr = x1_index * 16773120 + r0_range * 4095 + x0_range
        ptrs = (in_ptr0
                + x1_index * 16773120
                + r0_range[:, None] * 4095
                + x0_hints[None, :])  # broadcast 1-D hints to [R0_BLOCK, XBLOCK]
        tile = tl.load(ptrs,
                       mask=r0_mask[:, None] & x0_mask[None, :],
                       other=float('-inf'),
                       eviction_policy='evict_first')  # [R0_BLOCK, XBLOCK]
        # Reduce over r0 dimension (axis=0), accumulate into [XBLOCK]
        _max_val = tl.maximum(_max_val, tl.max(tile, axis=0))

    # Store result: output shape [128, 4095], stride [4095, 1]
    out_ptrs = out_ptr0 + x1_index * 4095 + x0_range
    tl.store(out_ptrs, _max_val, mask=x0_mask)

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