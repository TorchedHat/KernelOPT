import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_R': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_R': 64},  num_warps=4,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_R': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_R': 128}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 256, 'BLOCK_R': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 64,  'BLOCK_R': 256}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_R': 256}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_R': 128}, num_warps=8,  num_stages=4),
        triton.Config({'BLOCK_M': 256, 'BLOCK_R': 256}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_M': 512, 'BLOCK_R': 64},  num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 512, 'BLOCK_R': 128}, num_warps=16, num_stages=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_R': 512}, num_warps=16, num_stages=4),
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
    # 2D grid: pid(0) = col chunk, pid(1) = batch index
    # BLOCK_M tiles over last_dim (output columns), BLOCK_R tiles over reduce_dim
    pid_col   = tl.program_id(0)   # which BLOCK_M chunk of last_dim
    pid_batch = tl.program_id(1)   # which batch index

    col_offsets = pid_col * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    col_mask    = col_offsets < last_dim

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for r_start in range(0, reduce_dim, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)   # [BLOCK_R]
        r_mask    = r_offsets < reduce_dim

        # Load [BLOCK_R, BLOCK_M]: col_offsets varies fastest (stride=1 in memory) — COALESCED
        r_idx2 = r_offsets[:, None]    # [BLOCK_R, 1]
        c_idx2 = col_offsets[None, :]  # [1, BLOCK_M]
        linear_idx2 = (pid_batch * reduce_dim * last_dim
                       + r_idx2 * last_dim   # each row is one r value
                       + c_idx2)             # cols are contiguous, stride=1

        combined_mask = r_mask[:, None] & col_mask[None, :]  # [BLOCK_R, BLOCK_M]

        vals = tl.load(in_ptr + linear_idx2,
                       mask=combined_mask, other=0.0)  # [BLOCK_R, BLOCK_M]
        acc += tl.sum(vals, axis=0)   # sum over r axis -> [BLOCK_M]

    tl.store(out_ptr + pid_batch * last_dim + col_offsets,
             acc, mask=col_mask)

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