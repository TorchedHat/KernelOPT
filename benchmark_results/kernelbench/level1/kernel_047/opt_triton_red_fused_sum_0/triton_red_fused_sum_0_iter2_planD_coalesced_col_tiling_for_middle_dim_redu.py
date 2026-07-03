import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128,  'BLOCK_R': 1}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 256,  'BLOCK_R': 1}, num_warps=4,  num_stages=2),
        triton.Config({'BLOCK_M': 512,  'BLOCK_R': 1}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 1024, 'BLOCK_R': 1}, num_warps=8,  num_stages=2),
        triton.Config({'BLOCK_M': 2048, 'BLOCK_R': 1}, num_warps=16, num_stages=2),
        triton.Config({'BLOCK_M': 256,  'BLOCK_R': 1}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 512,  'BLOCK_R': 1}, num_warps=8,  num_stages=3),
        triton.Config({'BLOCK_M': 1024, 'BLOCK_R': 1}, num_warps=16, num_stages=3),
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
    # Grid: (cdiv(last_dim, BLOCK_M), batch)
    # BLOCK_M tiles over col_idx (stride=1, fully coalesced)
    # BLOCK_R is unused (we loop scalar over r for coalesced access)
    pid_c = tl.program_id(0)   # col chunk index
    pid_b = tl.program_id(1)   # batch index

    col_offsets = pid_c * BLOCK_M + tl.arange(0, BLOCK_M)  # stride=1 in memory
    col_mask = col_offsets < last_dim

    acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    base = pid_b * (reduce_dim * last_dim)

    for r in range(0, reduce_dim):
        # Load x[pid_b, r, col_offsets]: contiguous along last_dim (stride=1)
        # -> fully coalesced 128-bit loads
        linear_idx = base + r * last_dim + col_offsets  # [BLOCK_M]
        vals = tl.load(in_ptr + linear_idx, mask=col_mask, other=0.0)
        acc += vals

    # Store: out[pid_b * last_dim + col_offsets]
    out_offsets = pid_b * last_dim + col_offsets
    tl.store(out_ptr + out_offsets, acc, mask=col_mask)

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