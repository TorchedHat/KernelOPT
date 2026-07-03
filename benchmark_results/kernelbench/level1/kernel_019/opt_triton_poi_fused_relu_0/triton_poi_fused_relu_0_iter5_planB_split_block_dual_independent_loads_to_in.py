import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    HALF: tl.constexpr = BLOCK_SIZE // 2
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets_a = block_start + tl.arange(0, HALF)           # first half
    offsets_b = block_start + HALF + tl.arange(0, HALF)   # second half
    mask_a = offsets_a < n_elements
    mask_b = offsets_b < n_elements
    x_a = tl.load(input_ptr + offsets_a, mask=mask_a, other=0.0)  # load A issued first
    x_b = tl.load(input_ptr + offsets_b, mask=mask_b, other=0.0)  # load B issued immediately — both in-flight simultaneously
    r_a = tl.where(x_a > 0.0, x_a, 0.0)  # compute after both loads issued
    r_b = tl.where(x_b > 0.0, x_b, 0.0)
    tl.store(output_ptr + offsets_a, r_a, mask=mask_a, eviction_policy='evict_first')
    tl.store(output_ptr + offsets_b, r_b, mask=mask_b, eviction_policy='evict_first')

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