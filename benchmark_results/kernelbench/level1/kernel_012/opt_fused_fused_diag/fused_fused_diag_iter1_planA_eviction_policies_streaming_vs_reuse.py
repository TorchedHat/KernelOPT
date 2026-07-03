import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 1024}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 512}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 2, 'BLOCK_COLS': 512}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 4, 'BLOCK_COLS': 256}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 256}, num_warps=2),
        triton.Config({'BLOCK_ROWS': 8, 'BLOCK_COLS': 128}, num_warps=4),
        triton.Config({'BLOCK_ROWS': 1, 'BLOCK_COLS': 2048}, num_warps=8),
        triton.Config({'BLOCK_ROWS': 2, 'BLOCK_COLS': 1024}, num_warps=8),
    ],
    key=['num_rows', 'num_cols'],
)
@triton.jit
def diag_matmul_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    num_rows,
    num_cols,
    stride_b_row,
    stride_b_col,
    stride_out_row,
    stride_out_col,
    BLOCK_ROWS: tl.constexpr,
    BLOCK_COLS: tl.constexpr,
):
    # Each program handles BLOCK_ROWS rows and BLOCK_COLS columns
    pid_row = tl.program_id(0)
    pid_col = tl.program_id(1)

    row_start = pid_row * BLOCK_ROWS
    col_start = pid_col * BLOCK_COLS

    row_offsets = row_start + tl.arange(0, BLOCK_ROWS)
    col_offsets = col_start + tl.arange(0, BLOCK_COLS)

    row_mask = row_offsets < num_rows
    col_mask = col_offsets < num_cols

    # Load diagonal elements A[row_offsets]
    # 'a' is a small diagonal vector (num_rows floats) reused across all column CTAs
    # Keep it L2-resident with evict_last (.ca cache modifier)
    a_vals = tl.load(a_ptr + row_offsets, mask=row_mask, other=0.0, cache_modifier='.ca')  # shape [BLOCK_ROWS]

    # Load B block: shape [BLOCK_ROWS, BLOCK_COLS]
    b_ptrs = b_ptr + row_offsets[:, None] * stride_b_row + col_offsets[None, :] * stride_b_col
    full_mask = row_mask[:, None] & col_mask[None, :]
    # 'b' is streaming — each element read exactly once; evict_first avoids displacing 'a'
    b_vals = tl.load(b_ptrs, mask=full_mask, other=0.0, cache_modifier='.cg')

    # Scale each row by corresponding diagonal element
    out_vals = a_vals[:, None] * b_vals

    # Store result
    out_ptrs = out_ptr + row_offsets[:, None] * stride_out_row + col_offsets[None, :] * stride_out_col
    # Output is write-once — streaming store bypasses L2 pollution
    tl.store(out_ptrs, out_vals, mask=full_mask, cache_modifier='.cs')

def kernel_function(a, b):
    assert a.ndim == 1
    assert b.ndim == 2
    assert a.shape[0] == b.shape[0]

    num_rows = b.shape[0]
    num_cols = b.shape[1]

    out = torch.empty((num_rows, num_cols), dtype=b.dtype, device=b.device)

    def grid(meta):
        return (
            triton.cdiv(num_rows, meta['BLOCK_ROWS']),
            triton.cdiv(num_cols, meta['BLOCK_COLS']),
        )

    diag_matmul_kernel[grid](
        a,
        b,
        out,
        num_rows,
        num_cols,
        b.stride(0),
        b.stride(1),
        out.stride(0),
        out.stride(1),
    )

    return out


def get_inputs():
    N = 4096
    M = 4096
    a = torch.rand(N, device='cuda', dtype=torch.float32)
    b = torch.rand(N, M, device='cuda', dtype=torch.float32)
    return [a, b]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()

    inputs = get_inputs()
    a, b = inputs

    output = kernel_function(a, b)
    torch.cuda.synchronize()

    if not args.profile_only:
        # Verify correctness
        expected = torch.diag(a) @ b
        max_diff = (output - expected).abs().max().item()
        print(f"Max difference: {max_diff}")
        assert max_diff < 1e-3, f"Results don't match! Max diff: {max_diff}"
        print("Kernel executed successfully.")