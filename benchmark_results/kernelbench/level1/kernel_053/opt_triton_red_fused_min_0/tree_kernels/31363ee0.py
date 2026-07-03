import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 4096}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 128},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 512},  num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def min_dim1_kernel(
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
    xindex = xoffset + tl.arange(0, XBLOCK)   # [XBLOCK] 1D
    xmask = xindex < xnumel

    # x0 = column index (dim2), x1 = batch index
    x0 = xindex % 4095   # [XBLOCK]
    x1 = xindex // 4095  # [XBLOCK]

    # Initialize accumulator with +inf, shape [XBLOCK]
    _min_val = tl.full([XBLOCK], float("inf"), tl.float32)

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + tl.arange(0, R0_BLOCK)  # [R0_BLOCK] contiguous
        r0_mask = r0_index < r0_numel

        # 2D pointer array: [XBLOCK, R0_BLOCK]
        # innermost dim = r0_index -> stride-1 in memory -> COALESCED
        # Input layout: [128, 4096, 4095] with strides [16773120, 4095, 1]
        # index = x1 * 16773120 + r0_index * 4095 + x0
        ptrs = (in_ptr0
                + x0[:, None]                # [XBLOCK, 1]
                + 4095 * r0_index[None, :]   # [1, R0_BLOCK] stride-1!
                + 16773120 * x1[:, None])    # [XBLOCK, 1]
        mask2d = xmask[:, None] & r0_mask[None, :]
        tmp0 = tl.load(ptrs, mask=mask2d, other=float("inf"),
                       eviction_policy='evict_first')  # [XBLOCK, R0_BLOCK]
        row_min = tl.min(tmp0, axis=1)   # reduce R0 immediately -> [XBLOCK]
        _min_val = tl.minimum(_min_val, row_min)

    tl.store(out_ptr0 + xindex, _min_val, mask=xmask)

def kernel_function(*inputs):
    x, = inputs
    assert x.ndim == 3
    batch, dim1, dim2 = x.shape

    # Output shape: [batch, dim2] = [128, 4095]
    out = torch.empty((batch, dim2), dtype=x.dtype, device=x.device)

    xnumel = batch * dim2  # 128 * 4095 = 524160
    r0_numel = dim1  # 4096

    # Grid: number of x-blocks
    def grid(meta):
        return (triton.cdiv(xnumel, meta['XBLOCK']),)

    min_dim1_kernel[grid](
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
        x = inputs[0]
        ref = torch.min(x, dim=1)[0]
        print(f"Output shape: {output.shape}")
        print(f"Reference shape: {ref.shape}")
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference: {max_diff}")
        if max_diff < 1e-3:
            print("Correctness PASSED!")
        else:
            print("Correctness FAILED!")