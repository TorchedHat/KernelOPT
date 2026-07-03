import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=2,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 32768}, num_warps=32, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def gelu_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    pid_col = tl.program_id(0)
    pid_row = tl.program_id(1)
    block_start = pid_row * n_cols + pid_col * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, eviction_policy='evict_first')

    # GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
    # = x * (0.5 * erf(x * 0.7071067811865476) + 0.5)
    # FMA-friendly: inner expression is one FFMA, outer is one FMUL
    x_scaled = x * 0.7071067811865476
    erf_val = libdevice.erf(x_scaled)
    result = x * tl.fma(0.5, erf_val, 0.5)

    tl.store(output_ptr + offsets, result, mask=mask, eviction_policy='evict_first')

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