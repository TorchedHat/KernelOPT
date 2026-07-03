import torch
import triton
import triton.language as tl


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'RBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'RBLOCK': 512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 4, 'RBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4, 'RBLOCK': 512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 8, 'RBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8, 'RBLOCK': 512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 16, 'RBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'RBLOCK': 256}, num_warps=8, num_stages=1),
    ],
    key=['xnumel', 'rnumel'],
)
@triton.jit
def argmin_kernel(
    in_ptr,
    out_ptr,
    xnumel,
    rnumel,
    XBLOCK: tl.constexpr,
    RBLOCK: tl.constexpr,
):
    # xnumel = 128 * 4095 = 524160
    # rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel

    r_base = tl.arange(0, RBLOCK)[None, :]

    # Decompose xindex into batch and col indices
    x_col = xindex % dim2      # column index (0..4094)
    x_batch = xindex // dim2   # batch index (0..127)

    # Initialize running min value and index
    cur_min = tl.full([XBLOCK, RBLOCK], float('inf'), dtype=tl.float32)
    cur_idx = tl.full([XBLOCK, RBLOCK], 2147483647, dtype=tl.int32)

    for r_offset in tl.range(0, rnumel, RBLOCK):
        r_index = r_offset + r_base
        r_mask = r_index < rnumel

        # Load input: shape [128, 4096, 4095], stride [16773120, 4095, 1]
        # Element at [batch, r, col] = batch * 16773120 + r * 4095 + col
        ptr = x_batch * stride_x0 + r_index * stride_r + x_col * stride_x1
        val = tl.load(in_ptr + ptr, mask=(xmask & r_mask), other=float('inf'))

        # Update minimum: keep track of index
        # Use stable comparison: update if val < cur_min, or (val == cur_min and r_index < cur_idx)
        val_bc = tl.broadcast_to(val, [XBLOCK, RBLOCK])
        r_bc = tl.broadcast_to(r_index, [XBLOCK, RBLOCK])

        update = val_bc < cur_min
        cur_min = tl.where(update, val_bc, cur_min)
        cur_idx = tl.where(update, r_bc, cur_idx)

    # Reduce across RBLOCK dimension to get final argmin
    # We need to find the minimum across the RBLOCK dimension
    # Do a tree reduction manually
    final_min = tl.min(cur_min, axis=1)[:, None]
    is_min = (cur_min == final_min)
    # Among positions with minimum value, pick the smallest index
    masked_idx = tl.where(is_min, cur_idx, 2147483647)
    final_idx = tl.min(masked_idx, axis=1)

    tl.store(out_ptr + xindex[:, 0], final_idx.to(tl.int64), mask=xmask[:, 0])


def kernel_function(x):
    assert x.is_cuda and x.dtype == torch.float32
    batch, dim1, dim2 = x.shape  # 128, 4096, 4095
    # argmin along dim=1, output shape [128, 4095]
    out = torch.empty((batch, dim2), dtype=torch.int64, device=x.device)

    xnumel = batch * dim2  # 524160
    rnumel = dim1           # 4096

    stride_x0 = x.stride(0)  # 16773120
    stride_r  = x.stride(1)  # 4095
    stride_x1 = x.stride(2)  # 1
    out_stride0 = out.stride(0)  # 4095
    out_stride1 = out.stride(1)  # 1

    grid = lambda meta: (triton.cdiv(xnumel, meta['XBLOCK']),)

    argmin_kernel[grid](
        x, out,
        xnumel, rnumel,
        stride_x0, stride_x1, stride_r,
        out_stride0, out_stride1,
        dim2,
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
        ref = torch.argmin(inputs[0], dim=1)
        if torch.allclose(output.float(), ref.float(), atol=1e-3):
            print("Kernel executed successfully. Results match!")
        else:
            diff = (output != ref).sum().item()
            total = ref.numel()
            print(f"MISMATCH: {diff}/{total} elements differ")
            # Show some examples
            mismatch_idx = (output != ref).nonzero(as_tuple=False)[:5]
            for idx in mismatch_idx:
                b, c = idx[0].item(), idx[1].item()
                print(f"  [{b},{c}]: got {output[b,c].item()}, expected {ref[b,c].item()}")