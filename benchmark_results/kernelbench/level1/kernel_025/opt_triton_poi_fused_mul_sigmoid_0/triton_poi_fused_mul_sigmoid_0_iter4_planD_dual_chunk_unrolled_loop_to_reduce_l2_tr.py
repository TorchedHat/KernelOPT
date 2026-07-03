import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 8192}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_SIZE': 4096}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 512},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_SIZE': 256},  num_warps=4,  num_stages=1),
        triton.Config({'BLOCK_SIZE': 128},  num_warps=2,  num_stages=1),
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
    grid_size = tl.num_programs(0)
    num_chunks = tl.cdiv(n_elements, BLOCK_SIZE)

    for chunk_id in tl.range(pid, num_chunks, grid_size * 2):
        # Chunk A (always valid since chunk_id < num_chunks)
        start_a = chunk_id * BLOCK_SIZE
        offs_a  = start_a + tl.arange(0, BLOCK_SIZE)
        mask_a  = offs_a < n_elements
        x_a     = tl.load(x_ptr + offs_a, mask=mask_a, eviction_policy='evict_first')

        # Chunk B (may be out-of-range — clamp chunk_id_b to avoid OOB pointer arithmetic)
        chunk_id_b = chunk_id + grid_size
        # Clamp to num_chunks - 1 so pointer stays within allocated memory;
        # mask_b will be all-False when chunk_id_b >= num_chunks, suppressing load/store
        chunk_id_b_safe = tl.minimum(chunk_id_b, num_chunks - 1)
        start_b = chunk_id_b_safe * BLOCK_SIZE
        offs_b  = start_b + tl.arange(0, BLOCK_SIZE)
        mask_b  = (chunk_id_b < num_chunks) & (offs_b < n_elements)
        x_b     = tl.load(x_ptr + offs_b, mask=mask_b, eviction_policy='evict_first')

        # Compute A and B independently (enables ILP / FMA pairing)
        xa_f32  = x_a.to(tl.float32)
        xb_f32  = x_b.to(tl.float32)
        res_a   = xa_f32 * tl.sigmoid(xa_f32)
        res_b   = xb_f32 * tl.sigmoid(xb_f32)

        # Store both
        tl.store(out_ptr + offs_a, res_a.to(x_a.dtype), mask=mask_a, eviction_policy='evict_first')
        tl.store(out_ptr + offs_b, res_b.to(x_b.dtype), mask=mask_b, eviction_policy='evict_first')

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