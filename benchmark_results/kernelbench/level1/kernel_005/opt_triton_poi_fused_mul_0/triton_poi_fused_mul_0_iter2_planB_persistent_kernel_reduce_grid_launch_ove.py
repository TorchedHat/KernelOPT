import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def mul_scalar_kernel(
    in_ptr,
    out_ptr,
    scalar,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)
    num_tiles = tl.cdiv(n_elements, BLOCK_SIZE)
    tile_id = sm_id
    while tile_id < num_tiles:
        block_start = tile_id * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr + offsets, mask=mask, eviction_policy='evict_first')
        result = tl.math.fma(x, scalar, 0.0)
        tl.store(out_ptr + offsets, result, mask=mask, eviction_policy='evict_first')
        tile_id += num_sms

def kernel_function(a, s):
    output = torch.empty_like(a)
    n_elements = a.numel()
    scalar = float(s)
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    mul_scalar_kernel[grid](
        a,
        output,
        scalar,
        n_elements,
    )
    return output


def get_inputs():
    M = 16384 * 4
    N = 4096 * 4
    A = torch.rand(M, N, device='cuda', dtype=torch.float32)
    s = 3.14
    return [A, s]


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
        a, s = inputs
        expected = a * s
        max_diff = (output - expected).abs().max().item()
        print(f"Max difference from PyTorch: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")