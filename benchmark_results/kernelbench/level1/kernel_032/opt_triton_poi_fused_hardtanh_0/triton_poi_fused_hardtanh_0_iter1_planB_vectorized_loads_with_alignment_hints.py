import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2,  num_stages=2),
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
    # tl.multiple_of: block_start is always a multiple of BLOCK_SIZE (pid * BLOCK_SIZE)
    # tl.max_contiguous: the BLOCK_SIZE offsets are contiguous (arange)
    # Together these unlock 128-bit (LDG.128) vectorized loads
    offsets = tl.max_contiguous(
        tl.multiple_of(block_start + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE),
        BLOCK_SIZE
    )
    mask = offsets < n_elements
    x = tl.load(input_ptr + offsets, mask=mask)
    # HardTanh: clamp to [min_val, max_val]
    x = tl.where(x < min_val, min_val, x)
    x = tl.where(x > max_val, max_val, x)
    tl.store(output_ptr + offsets, x, mask=mask)

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