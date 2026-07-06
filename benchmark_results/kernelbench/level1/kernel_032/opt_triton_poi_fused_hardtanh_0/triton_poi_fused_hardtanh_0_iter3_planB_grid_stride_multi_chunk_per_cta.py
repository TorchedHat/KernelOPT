import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        # CHUNKS>1: num_stages=1 because unrolled loop provides ILP
        triton.Config({'BLOCK_SIZE': 1024, 'CHUNKS': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512,  'CHUNKS': 8}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048, 'CHUNKS': 4}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024, 'CHUNKS': 4}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512,  'CHUNKS': 4}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 2048, 'CHUNKS': 2}, num_warps=8, num_stages=1),
        triton.Config({'BLOCK_SIZE': 1024, 'CHUNKS': 2}, num_warps=4, num_stages=1),
        triton.Config({'BLOCK_SIZE': 512,  'CHUNKS': 2}, num_warps=4, num_stages=1),
        # CHUNKS=1 fallback: num_stages>1 for software pipelining
        triton.Config({'BLOCK_SIZE': 4096, 'CHUNKS': 1}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048, 'CHUNKS': 1}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 1024, 'CHUNKS': 1}, num_warps=4,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 4096, 'CHUNKS': 1}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_SIZE': 2048, 'CHUNKS': 1}, num_warps=16, num_stages=4),
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
    CHUNKS: tl.constexpr,
):
    pid = tl.program_id(0)
    base = pid * BLOCK_SIZE * CHUNKS
    offsets = tl.arange(0, BLOCK_SIZE)
    for chunk in tl.static_range(CHUNKS):
        chunk_start = base + chunk * BLOCK_SIZE
        co = chunk_start + offsets
        mask = co < n_elements
        x = tl.load(input_ptr + co, mask=mask, other=0.0)
        x = tl.where(x < min_val, min_val, x)
        x = tl.where(x > max_val, max_val, x)
        tl.store(output_ptr + co, x, mask=mask)

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