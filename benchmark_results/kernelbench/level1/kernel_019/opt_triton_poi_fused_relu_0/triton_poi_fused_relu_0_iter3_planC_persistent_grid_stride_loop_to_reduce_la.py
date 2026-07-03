import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096,  'BLOCKS_PER_SM': 4},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 4096,  'BLOCKS_PER_SM': 8},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192,  'BLOCKS_PER_SM': 4},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 8192,  'BLOCKS_PER_SM': 8},  num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384, 'BLOCKS_PER_SM': 4},  num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384, 'BLOCKS_PER_SM': 8},  num_warps=16),
        triton.Config({'BLOCK_SIZE': 32768, 'BLOCKS_PER_SM': 2},  num_warps=16),
        triton.Config({'BLOCK_SIZE': 32768, 'BLOCKS_PER_SM': 4},  num_warps=16),
        triton.Config({'BLOCK_SIZE': 65536, 'BLOCKS_PER_SM': 2},  num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, BLOCKS_PER_SM: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    # Cast to int64 to prevent overflow: n_elements can be ~1.6B, and
    # block_start accumulates up to n_elements which exceeds int32 max
    block_start = pid.to(tl.int64) * BLOCK_SIZE
    stride = num_programs.to(tl.int64) * BLOCK_SIZE
    while block_start < n_elements:
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(input_ptr + offsets, mask=mask, other=0.0, eviction_policy='evict_first')
        zero = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
        result = tl.maximum(zero, x)
        tl.store(output_ptr + offsets, result, mask=mask, eviction_policy='evict_first')
        block_start += stride

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