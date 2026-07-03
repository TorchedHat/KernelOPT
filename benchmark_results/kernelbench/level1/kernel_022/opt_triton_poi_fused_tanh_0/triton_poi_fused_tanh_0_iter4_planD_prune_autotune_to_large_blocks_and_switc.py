import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def tanh_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = tl.multiple_of(pid * BLOCK_SIZE, BLOCK_SIZE)
    offsets = tl.max_contiguous(block_start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0, cache_modifier='.ca')
    y = libdevice.tanh(x)
    tl.store(output_ptr + offsets, y, mask=mask, cache_modifier='.cs')

def kernel_function(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    tanh_kernel[grid](x, output, n_elements)
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
        expected = torch.tanh(inputs[0])
        max_diff = (output - expected).abs().max().item()
        print(f"Max difference from torch.tanh: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")