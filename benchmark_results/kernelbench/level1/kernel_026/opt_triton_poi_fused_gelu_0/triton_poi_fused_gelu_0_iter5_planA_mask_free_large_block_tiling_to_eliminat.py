import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=16, num_stages=1),
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
    offsets = tl.multiple_of(offsets, 16)  # 16-element (64-byte) alignment hint
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0, cache_modifier='.cg')

    # tanh-GELU: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    # FMA-restructured for higher throughput
    x2 = x * x
    x_cubed = x2 * x
    # inner = sqrt(2/pi)*0.044715 * x^3 + sqrt(2/pi) * x
    # C1 = 0.7978845608028654 * 0.044715 = 0.035676274473817765
    inner = libdevice.fma(0.035676274473817765, x_cubed, 0.7978845608028654 * x)
    tanh_val = libdevice.tanh(inner)
    half_x = 0.5 * x
    # result = 0.5*x*(1+tanh) = 0.5*x*tanh + 0.5*x = fma(half_x, tanh_val, half_x)
    result = libdevice.fma(half_x, tanh_val, half_x)

    tl.store(output_ptr + offsets, result, mask=mask, cache_modifier='.cg')

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