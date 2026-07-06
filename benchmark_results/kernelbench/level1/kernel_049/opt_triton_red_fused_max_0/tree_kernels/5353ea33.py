import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=2,  num_stages=2),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 256}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 64},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 32},  num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 64},  num_warps=8,  num_stages=2),
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
    # x dimension: batch * dim2 = 128 * 4095 = 524160
    # r0 dimension: dim1 = 4096 (reduction dimension)
    pid = tl.program_id(0)
    num_x0_progs = tl.cdiv(4095, XBLOCK)
    x1_index = pid // num_x0_progs          # batch index (scalar)
    x0_offset = (pid % num_x0_progs) * XBLOCK
    x0_range = x0_offset + tl.arange(0, XBLOCK)   # [XBLOCK], stride-1
    x0_hints = tl.max_contiguous(tl.multiple_of(x0_range, 4), 4)  # vectorization hint
    x0_mask = x0_range < 4095

    # Initialize 1D accumulator with -inf
    acc = tl.full([XBLOCK], float('-inf'), tl.float32)

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_range = r0_offset + tl.arange(0, R0_BLOCK)   # [R0_BLOCK]
        r0_mask = r0_range < r0_numel

        # Input layout: [128, 4096, 4095] with strides [16773120, 4095, 1]
        # ptr = x1_index * 16773120 + r0_range * 4095 + x0_hints
        ptrs = in_ptr0 + x1_index * 16773120 + r0_range[:, None] * 4095 + x0_hints[None, :]
        tile = tl.load(ptrs, mask=r0_mask[:, None] & x0_mask[None, :],
                       eviction_policy='evict_first', other=float('-inf'))  # [R0_BLOCK, XBLOCK]
        tile_max = tl.max(tile, axis=0)   # [XBLOCK], exact fp32 max
        acc = tl.maximum(acc, tile_max)   # [XBLOCK]

    # Store result
    tl.store(out_ptr0 + x1_index * 4095 + x0_range, acc, mask=x0_mask)

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