import torch
import triton
import triton.language as tl



@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 4096}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 4096}, num_warps=16, num_stages=3),
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
    XBLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    numel = num_rows * num_cols
    flat_start = pid * XBLOCK
    flat_offsets = flat_start + tl.arange(0, XBLOCK)
    mask = flat_offsets < numel
    row_offsets = flat_offsets // num_cols
    col_offsets = flat_offsets % num_cols
    a_vals = tl.load(a_ptr + row_offsets, mask=mask, other=0.0)
    b_ptrs = b_ptr + row_offsets * stride_b_row + col_offsets * stride_b_col
    b_vals = tl.load(b_ptrs, mask=mask, other=0.0)
    out_vals = a_vals * b_vals
    out_ptrs = out_ptr + row_offsets * stride_out_row + col_offsets * stride_out_col
    tl.store(out_ptrs, out_vals, mask=mask)

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