import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2, num_stages=3),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2, num_ctas=2),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8, num_stages=2, num_ctas=2),
        triton.Config({'BLOCK_SIZE': 4096},  num_warps=8, num_stages=2, num_ctas=2),
        triton.Config({'BLOCK_SIZE': 16384}, num_warps=8, num_stages=2, num_ctas=4),
        triton.Config({'BLOCK_SIZE': 8192},  num_warps=8, num_stages=2, num_ctas=4),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=1, num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=1, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=2, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=1, num_stages=2),
    ],
    key=['n_elements'],
)
@triton.jit
def hardtanh_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    min_val: tl.constexpr,
    max_val: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask, eviction_policy='evict_first')
    # HardTanh: clamp to [min_val, max_val] using FP16 intermediates to reduce register pressure
    x16 = x.to(tl.float16)
    x16 = tl.minimum(x16, tl.full(x16.shape, max_val, tl.float16))
    x16 = tl.maximum(x16, tl.full(x16.shape, min_val, tl.float16))
    x = x16.to(tl.float32)
    tl.store(output_ptr + offsets, x, mask=mask, eviction_policy='evict_first')

def kernel_function(*inputs):
    x, = inputs
    output = torch.empty_like(x)
    n_elements = x.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    hardtanh_kernel[grid](
        x,
        output,
        n_elements,
        min_val=-1.0,
        max_val=1.0,
    )
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
        ref = F.hardtanh(inputs[0], min_val=-1., max_val=1.)
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")