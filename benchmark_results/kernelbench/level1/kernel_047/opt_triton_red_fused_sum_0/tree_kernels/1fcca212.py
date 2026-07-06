import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 1024}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 2048}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_R': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 2, 'BLOCK_R': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 4, 'BLOCK_R': 256}, num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 4, 'BLOCK_R': 512}, num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 1, 'BLOCK_R': 4096}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 2, 'BLOCK_R': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 4, 'BLOCK_R': 1024}, num_warps=8, num_stages=3),
        triton.Config({'BLOCK_M': 2, 'BLOCK_R': 2048}, num_warps=8, num_stages=3),
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
    # Grid: (batch * last_dim / BLOCK_M,)
    # Each program handles BLOCK_M output elements along last_dim
    # and reduces over reduce_dim
    pid = tl.program_id(0)

    # Each pid handles BLOCK_M consecutive last_dim positions for a given batch
    # We flatten: output index = batch_idx * last_dim + col_idx
    # pid covers BLOCK_M output elements
    out_start = pid * BLOCK_M

    # Compute batch index and column index
    col_offsets = out_start + tl.arange(0, BLOCK_M)
    batch_idx = col_offsets // last_dim
    col_idx = col_offsets % last_dim
    # Hint: col_idx is contiguous and aligned to BLOCK_M
    col_idx = tl.max_contiguous(tl.multiple_of(col_idx, BLOCK_M), BLOCK_M)

    mask_out = (batch_idx < batch) & (col_idx < last_dim)

    # Accumulate sum over reduce_dim
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for r_start in range(0, reduce_dim, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)
        # Hint: r_offsets are contiguous and aligned to BLOCK_R
        r_offsets = tl.max_contiguous(tl.multiple_of(r_offsets, BLOCK_R), BLOCK_R)
        r_mask = r_offsets < reduce_dim

        # Load: in_ptr[batch_idx, r_offsets, col_idx]
        # Shape: [BLOCK_R, BLOCK_M]  (transposed for coalesced access)
        # in_ptr stride: [reduce_dim * last_dim, last_dim, 1]
        # Linear index: batch_idx * reduce_dim * last_dim + r * last_dim + col_idx

        b_idx = batch_idx[None, :]  # [1, BLOCK_M] — broadcast over r
        r_idx = r_offsets[:, None]  # [BLOCK_R, 1] — r varies on axis=0
        c_idx = col_idx[None, :]    # [1, BLOCK_M] — col varies on axis=1 (contiguous)

        linear_idx = b_idx * (reduce_dim * last_dim) + r_idx * last_dim + c_idx  # [BLOCK_R, BLOCK_M]

        combined_mask = r_mask[:, None] & mask_out[None, :]  # [BLOCK_R, BLOCK_M]

        vals = tl.load(in_ptr + linear_idx, mask=combined_mask, other=0.0)
        acc += tl.sum(vals, axis=0)  # reduce over BLOCK_R (axis=0), acc shape stays [BLOCK_M]

    tl.store(out_ptr + col_offsets, acc, mask=mask_out)

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