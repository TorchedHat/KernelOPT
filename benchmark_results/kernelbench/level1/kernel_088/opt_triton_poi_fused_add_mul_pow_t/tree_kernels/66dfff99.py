import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2,  num_stages=4),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)

    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x2 = x * x
    x3 = x2 * x
    inner = 0.7978845608028654 * x + 0.03567627438067918 * x3  # explicit FMA form
    tanh_val = libdevice.tanh(inner)
    result = 0.5 * x * (1.0 + tanh_val)

    tl.store(out_ptr + offsets, result, mask=mask, eviction_policy='evict_first')

def kernel_function(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, output, n_elements)
    return output


def get_inputs():
    return [torch.rand(8192, 8192, device='cuda', dtype=torch.float32)]


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
        import math
        x = inputs[0]
        expected = 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))
        max_diff = (output - expected).abs().max().item()
        print(f"Max difference from reference: {max_diff:.6e}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match.")