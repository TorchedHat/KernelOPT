import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64,  'BLOCK_R': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_R': 256},  num_warps=4, num_stages=2),
        triton.Config({'BLOCK_M': 128, 'BLOCK_R': 512},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_R': 256},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 256, 'BLOCK_R': 512},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 512, 'BLOCK_R': 128},  num_warps=8, num_stages=2),
        triton.Config({'BLOCK_M': 64,  'BLOCK_R': 512},  num_warps=4, num_stages=3),
        triton.Config({'BLOCK_M': 128, 'BLOCK_R': 1024}, num_warps=8, num_stages=3),
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
    # Grid: (batch * cdiv(last_dim, BLOCK_M),)
    # Each program handles BLOCK_M contiguous columns for one batch element
    # and reduces over reduce_dim
    pid = tl.program_id(0)

    # Decompose pid into (batch_idx, col_tile)
    num_col_tiles = tl.cdiv(last_dim, BLOCK_M)
    pid_batch = pid // num_col_tiles
    pid_col   = pid %  num_col_tiles

    # BLOCK_M consecutive column offsets — contiguous in memory (stride-1)
    col_offsets = pid_col * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    mask_col = (col_offsets < last_dim) & (pid_batch < batch)

    # Accumulate sum over reduce_dim
    acc = tl.zeros([BLOCK_M], dtype=tl.float32)

    for r_start in range(0, reduce_dim, BLOCK_R):
        r_offsets = r_start + tl.arange(0, BLOCK_R)
        r_mask = r_offsets < reduce_dim

        # Load: in_ptr[pid_batch, r_offsets, col_offsets]
        # Shape: [BLOCK_R, BLOCK_M] — innermost dim is contiguous col_offsets
        # in_ptr stride: [reduce_dim * last_dim, last_dim, 1]
        # Linear index: pid_batch*(reduce_dim*last_dim) + r_idx*last_dim + c_idx

        r_idx = r_offsets[:, None]    # [BLOCK_R, 1]
        c_idx = col_offsets[None, :]  # [1, BLOCK_M]

        linear_idx = pid_batch * (reduce_dim * last_dim) + r_idx * last_dim + c_idx  # [BLOCK_R, BLOCK_M]

        combined_mask = r_mask[:, None] & mask_col[None, :]

        vals = tl.load(in_ptr + linear_idx, mask=combined_mask, other=0.0)  # [BLOCK_R, BLOCK_M]
        acc += tl.sum(vals, axis=0)  # reduce over BLOCK_R -> [BLOCK_M]

    out_offsets = pid_batch * last_dim + col_offsets
    tl.store(out_ptr + out_offsets, acc, mask=mask_col)

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