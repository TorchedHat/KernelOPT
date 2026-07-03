import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE':  128}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE':  256}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE':  512}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=3),
    ],
    key=['n_elements'],
)
@triton.jit
def leaky_relu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    negative_slope,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.max_contiguous(tl.multiple_of(block_start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE), BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask)

    result = tl.where(x > 0, x, x * negative_slope)

    tl.store(output_ptr + offsets, result, mask=mask)

def kernel_function(*inputs):
    x = inputs[0]
    negative_slope = 0.01

    output = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)

    leaky_relu_kernel[grid](
        x,
        output,
        n_elements,
        negative_slope,
    )

    return output


def get_inputs():
    x = torch.rand(4096, 393216, dtype=torch.float32, device='cuda')
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
        expected = torch.nn.functional.leaky_relu(x, negative_slope=0.01)
        max_diff = (output - expected).abs().max().item()
        print(f"Max difference from PyTorch reference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match PyTorch reference.")