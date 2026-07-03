import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024, 'UNROLL_FACTOR': 1}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'UNROLL_FACTOR': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 1024, 'UNROLL_FACTOR': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 2048, 'UNROLL_FACTOR': 1}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 2048, 'UNROLL_FACTOR': 2}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 512, 'UNROLL_FACTOR': 2}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512, 'UNROLL_FACTOR': 4}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 4096, 'UNROLL_FACTOR': 1}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 4096, 'UNROLL_FACTOR': 2}, num_warps=16),
        triton.Config({'BLOCK_SIZE': 8192, 'UNROLL_FACTOR': 1}, num_warps=16),
    ],
    key=['n_elements'],
)
@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, UNROLL_FACTOR: tl.constexpr):
    pid = tl.program_id(0)
    cta_start = pid * BLOCK_SIZE * UNROLL_FACTOR

    # Issue ALL loads first (independent, no data dependency) to maximize MLP
    offsets_0 = cta_start + tl.arange(0, BLOCK_SIZE)
    offsets_0 = tl.max_contiguous(tl.multiple_of(offsets_0, BLOCK_SIZE), BLOCK_SIZE)
    mask_0 = offsets_0 < n_elements
    x0 = tl.load(input_ptr + offsets_0, mask=mask_0, other=0.0)

    if UNROLL_FACTOR >= 2:
        offsets_1 = cta_start + BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_1 = offsets_1 < n_elements
        x1 = tl.load(input_ptr + offsets_1, mask=mask_1, other=0.0)

    if UNROLL_FACTOR >= 4:
        offsets_2 = cta_start + 2 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_2 = offsets_2 < n_elements
        x2 = tl.load(input_ptr + offsets_2, mask=mask_2, other=0.0)

        offsets_3 = cta_start + 3 * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask_3 = offsets_3 < n_elements
        x3 = tl.load(input_ptr + offsets_3, mask=mask_3, other=0.0)

    # Compute and store all results
    zero = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    tl.store(output_ptr + offsets_0, tl.maximum(zero, x0), mask=mask_0)

    if UNROLL_FACTOR >= 2:
        tl.store(output_ptr + offsets_1, tl.maximum(zero, x1), mask=mask_1)

    if UNROLL_FACTOR >= 4:
        tl.store(output_ptr + offsets_2, tl.maximum(zero, x2), mask=mask_2)
        tl.store(output_ptr + offsets_3, tl.maximum(zero, x3), mask=mask_3)

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