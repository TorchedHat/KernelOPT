import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 2}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 4}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024, 'ELEMENTS_PER_THREAD': 8}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ELEMENTS_PER_THREAD': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ELEMENTS_PER_THREAD': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048, 'ELEMENTS_PER_THREAD': 8}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096, 'ELEMENTS_PER_THREAD': 2}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096, 'ELEMENTS_PER_THREAD': 4}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096, 'ELEMENTS_PER_THREAD': 8}, num_warps=8, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, ELEMENTS_PER_THREAD: tl.constexpr):
    pid = tl.program_id(0)
    tile_size = BLOCK_SIZE * ELEMENTS_PER_THREAD
    tile_start = pid * tile_size
    zero = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for i in tl.static_range(ELEMENTS_PER_THREAD):
        chunk_start = tile_start + i * BLOCK_SIZE
        offsets = chunk_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
        result = tl.maximum(zero, x)
        tl.store(output_ptr + offsets, result, mask=mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    x, = inputs
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](x, output, n_elements)
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
        expected = torch.relu(x)
        max_diff = (output - expected).abs().max().item()
        print(f"Max difference from PyTorch: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")