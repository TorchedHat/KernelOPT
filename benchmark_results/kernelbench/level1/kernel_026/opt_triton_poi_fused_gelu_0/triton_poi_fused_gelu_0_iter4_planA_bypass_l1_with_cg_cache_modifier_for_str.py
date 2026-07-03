import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=1, num_stages=1),
        triton.Config({'BLOCK_SIZE': 128}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=1, num_stages=2),
        triton.Config({'BLOCK_SIZE': 512}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, cache_modifier='.cg')

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    # = 0.5 * x * (1 + erf(x * 0.7071067811865476))
    x_half = x * 0.5
    x_scaled = tl.math.fma(x, 0.7071067811865476, 0.0)
    erf_val = libdevice.erf(x_scaled)
    result = tl.math.fma(x_half, erf_val, x_half)

    tl.store(output_ptr + offsets, result, mask=mask, cache_modifier='.cs')

def kernel_function(*inputs):
    x = inputs[0]
    assert x.is_cuda, "Input must be on CUDA"
    assert x.dtype == torch.float32, "Input must be float32"

    output = torch.empty_like(x)
    n_elements = x.numel()

    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    gelu_kernel[grid](x, output, n_elements)

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
        import torch.nn.functional as F
        ref = F.gelu(inputs[0])
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff:.6f}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully and results match.")