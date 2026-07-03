import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 64},   num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128},  num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},   num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128},  num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 32},   num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 64},   num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 32},   num_warps=16, num_stages=2),
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
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel

    r0_base = tl.arange(0, R0_BLOCK)[None, :]

    # Decode x index: x0 = col index (dim2), x1 = batch index
    x0 = xindex % 4095   # dim2 index
    x1 = xindex // 4095  # batch index

    # Initialize accumulator with -inf
    _max_val = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel

        # Input layout: [128, 4096, 4095] with strides [16773120, 4095, 1]
        # ptr = x1 * 16773120 + r0_index * 4095 + x0
        tmp0 = tl.load(in_ptr0 + (x0 + 4095 * r0_index + 16773120 * x1), mask=xmask, eviction_policy='evict_first', other=float("-inf"))
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
        _max_val = tl.where(r0_mask & xmask, tl.maximum(_max_val, tmp1), _max_val)

    # Reduce across R0_BLOCK dimension
    result = tl.max(_max_val, axis=1)[:, None]
    tl.store(out_ptr0 + xindex, result, mask=xmask)

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