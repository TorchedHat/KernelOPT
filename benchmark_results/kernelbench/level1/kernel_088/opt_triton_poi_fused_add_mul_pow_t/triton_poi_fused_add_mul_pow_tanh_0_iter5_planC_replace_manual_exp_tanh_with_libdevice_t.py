import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=2,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=2),
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
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0, cache_modifier='.ca')

    # GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    x_half = x * 0.5
    inner = (x + 0.044715 * x * x * x) * 0.7978845608028654
    result = x_half * (1.0 + libdevice.tanh(inner))

    tl.store(out_ptr + offsets, result, mask=mask, cache_modifier='.cs')

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