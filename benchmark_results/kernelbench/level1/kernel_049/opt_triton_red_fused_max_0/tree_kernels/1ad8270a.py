import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 32},  num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 16},  num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 32},  num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 16},  num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 32},  num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 16},  num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 64},  num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 64},  num_warps=16, num_stages=3),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},  num_warps=16, num_stages=4),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 32},  num_warps=16, num_stages=4),
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
    # New coalesced layout:
    # XBLOCK = TILE_DIM2: threads vary in dim2 (stride=1, coalesced)
    # R0_BLOCK = TILE_DIM1: loop tile over dim1 (reduction dimension, stride=4095)
    # pid covers (batch_idx, dim2_tile)
    pid = tl.program_id(0)
    num_tiles_dim2 = tl.cdiv(4095, XBLOCK)
    pid_b  = pid // num_tiles_dim2
    pid_d2 = pid  % num_tiles_dim2

    tid_d2 = tl.arange(0, XBLOCK)
    tid_d1 = tl.arange(0, R0_BLOCK)

    # Contiguity+alignment hint -> enables LDG.128 vectorized loads and conflict-free smem layout
    dim2_offsets = tl.max_contiguous(tl.multiple_of(pid_d2 * XBLOCK + tid_d2, 16), 16)
    dim2_mask    = dim2_offsets < 4095

    acc = tl.full([XBLOCK], float('-inf'), tl.float32)

    for d1_base in tl.range(0, r0_numel, R0_BLOCK):
        d1_offsets = d1_base + tid_d1
        d1_mask    = d1_offsets < r0_numel

        # Coalesced: consecutive threads differ in dim2_offsets (stride=1)
        ptrs = (in_ptr0
                + pid_b * 16773120
                + d1_offsets[:, None] * 4095
                + dim2_offsets[None, :])

        mask2d = d1_mask[:, None] & dim2_mask[None, :]
        tile = tl.load(ptrs, mask=mask2d, other=float('-inf'))

        tile_max = tl.max(tile, axis=0)
        acc = tl.maximum(acc, tile_max)

    out_ptrs = out_ptr0 + pid_b * 4095 + dim2_offsets
    tl.store(out_ptrs, acc, mask=dim2_mask)

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