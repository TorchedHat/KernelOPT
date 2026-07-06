import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 256},  num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 512},  num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 2048}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 4096}, num_warps=16, num_stages=2),
    ],
    key=['batch', 'reduce_dim', 'last_dim'],
)
@triton.jit
def sum_dim1_kernel(
    in_ptr,
    out_ptr,
    batch,
    reduce_dim,
    last_dim,
    BLOCK_M: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # 2D grid: axis 0 = output element index, axis 1 = r-chunk index
    pid_out = tl.program_id(0)   # which output element (batch_idx * last_dim + col_idx)
    pid_r   = tl.program_id(1)   # which chunk of reduce_dim

    batch_idx = pid_out // last_dim
    col_idx   = pid_out %  last_dim

    if batch_idx >= batch:
        return

    r_start   = pid_r * BLOCK_R
    r_offsets = r_start + tl.arange(0, BLOCK_R)   # contiguous in memory
    r_mask    = r_offsets < reduce_dim

    # Coalesced 1D load along r_offsets for fixed (batch_idx, col_idx)
    # linear_idx strides along r_offsets by last_dim — each CTA loads a contiguous
    # chunk of the reduction dimension for a single output element
    linear_idx = batch_idx * (reduce_dim * last_dim) + r_offsets * last_dim + col_idx
    vals = tl.load(in_ptr + linear_idx, mask=r_mask, other=0.0)
    partial = tl.sum(vals, axis=0)   # scalar

    # Accumulate partial sums from different pid_r CTAs into the pre-zeroed output
    tl.atomic_add(out_ptr + pid_out, partial)

def kernel_function(x):
    # x: [128, 4096, 4095], reduce over dim=1, keepdim=True -> [128, 1, 4095]
    batch = x.shape[0]
    reduce_dim = x.shape[1]
    last_dim = x.shape[2]

    out = torch.empty((batch, 1, last_dim), dtype=torch.float32, device=x.device)

    total_outputs = batch * last_dim  # 128 * 4095 = 524160

    # We'll use BLOCK_M to tile over output elements
    # Grid size depends on BLOCK_M (set by autotune)
    def grid(meta):
        return ((total_outputs + meta['BLOCK_M'] - 1) // meta['BLOCK_M'],)

    sum_dim1_kernel[grid](
        x,
        out.view(-1),  # flatten output for 1D indexing
        batch,
        reduce_dim,
        last_dim,
    )

    return out


def get_inputs():
    x = torch.rand(128, 4096, 4095, device='cuda', dtype=torch.float32)
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
        ref = torch.sum(x, dim=1, keepdim=True)
        max_diff = (output - ref).abs().max().item()
        print(f"Max difference from reference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")