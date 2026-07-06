import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=8, num_stages=2),
        # num_stages=4 for deeper pipelining with 2x unroll
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=4),
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
    base0 = pid * 2 * BLOCK_SIZE
    offsets0 = tl.max_contiguous(tl.multiple_of(base0 + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE), BLOCK_SIZE)
    mask0 = offsets0 < n_elements
    base1 = base0 + BLOCK_SIZE
    offsets1 = tl.max_contiguous(tl.multiple_of(base1 + tl.arange(0, BLOCK_SIZE), BLOCK_SIZE), BLOCK_SIZE)
    mask1 = offsets1 < n_elements

    # Issue BOTH loads before any compute — critical for ILP
    x0 = tl.load(x_ptr + offsets0, mask=mask0, other=0.0)
    x1 = tl.load(x_ptr + offsets1, mask=mask1, other=0.0)

    # swish(x) = x * sigmoid(x)
    x0_f32 = x0.to(tl.float32)
    x1_f32 = x1.to(tl.float32)
    result0 = x0_f32 * tl.sigmoid(x0_f32)
    result1 = x1_f32 * tl.sigmoid(x1_f32)

    tl.store(out_ptr + offsets0, result0.to(x0.dtype), mask=mask0)
    tl.store(out_ptr + offsets1, result1.to(x1.dtype), mask=mask1)

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