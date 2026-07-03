import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=32, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 512},   num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},   num_warps=4,  num_stages=2),
        # Existing configs exploring XBLOCK (128, 256, 512, 1024) with num_warps 4/8
        triton.Config({'BLOCK_SIZE': 128},   num_warps=4),
        triton.Config({'BLOCK_SIZE': 256},   num_warps=8),
        triton.Config({'BLOCK_SIZE': 512},   num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=8),
        # New num_stages=1 variants to reduce register pressure from software pipelining
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=32, num_stages=1),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=32, num_stages=1),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=16, num_stages=1),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048},  num_warps=8,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024},  num_warps=4,  num_stages=1),
    ],
    key=['n_elements'],
)
@triton.jit
def swish_kernel(
    x_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, cache_modifier='.ca')
    # swish(x) = x * sigmoid(x)
    result = x * tl.sigmoid(x)

    tl.store(out_ptr + offsets, result, mask=mask, cache_modifier='.cs')

def kernel_function(*inputs):
    x, = inputs
    assert x.is_cuda, "Input must be on CUDA"
    out = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    swish_kernel[grid](x, out, n_elements)
    return out


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
        expected = x * torch.sigmoid(x)
        max_diff = (output - expected).abs().max().item()
        print(f"Max difference from reference: {max_diff:.6e}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")