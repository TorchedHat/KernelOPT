"""Simple element-wise addition kernel for testing."""

import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr, y_ptr, output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Triton kernel for element-wise addition."""
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y

    tl.store(output_ptr + offsets, output, mask=mask)


def add_triton(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Run Triton addition kernel."""
    output = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)

    return output


if __name__ == "__main__":
    # Test the kernel
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')

    output_triton = add_triton(x, y)
    output_torch = x + y

    print(f"Triton output: {output_triton[:5]}")
    print(f"Torch output: {output_torch[:5]}")
    print(f"Allclose: {torch.allclose(output_triton, output_torch)}")
