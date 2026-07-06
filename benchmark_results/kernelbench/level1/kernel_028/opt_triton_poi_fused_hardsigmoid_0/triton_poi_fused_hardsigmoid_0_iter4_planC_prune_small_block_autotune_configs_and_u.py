import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=3),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def hardsigmoid_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, eviction_policy='evict_first')

    # hardsigmoid: clamp(x + 3, 0, 6) / 6
    x = x + 3.0
    x = tl.maximum(x, 0.0)
    x = tl.minimum(x, 6.0)
    x = x * (1.0 / 6.0)

    tl.store(out_ptr + offsets, x, mask=mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    x, = inputs
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    hardsigmoid_kernel[grid](x, output, n_elements)
    return output


def get_inputs():
    x = torch.rand(4096, 393216, device='cuda', dtype=torch.float32)
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
        ref = torch.nn.functional.hardsigmoid(x)
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")