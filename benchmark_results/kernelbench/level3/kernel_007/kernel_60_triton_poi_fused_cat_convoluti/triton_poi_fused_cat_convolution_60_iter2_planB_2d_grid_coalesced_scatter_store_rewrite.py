import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_cat_convolution_60(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D tiling: XBLOCK=128 covers columns (num_cols=112), program_id(0) covers rows
    # Each CTA handles one row tile (RBLOCK=16 rows) x XBLOCK=128 cols
    # This makes stores coalesced: consecutive threads write consecutive column addresses
    RBLOCK: tl.constexpr = 16
    num_cols: tl.constexpr = 112
    in_row_stride: tl.constexpr = 112
    out_row_stride: tl.constexpr = 528
    num_rows: tl.constexpr = 1960

    pid_row = tl.program_id(0)
    row_offsets = pid_row * RBLOCK + tl.arange(0, RBLOCK)
    row_mask = row_offsets < num_rows
    col_offsets = tl.arange(0, XBLOCK)
    col_mask = col_offsets < num_cols
    rows = row_offsets[:, None]
    cols = col_offsets[None, :]
    mask2d = row_mask[:, None] & col_mask[None, :]
    tmp0 = tl.load(in_ptr0 + rows * in_row_stride + cols, mask2d)
    tl.store(out_ptr0 + rows * out_row_stride + cols, tmp0, mask2d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 219520
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_convolution_60[grid](arg_0, arg_1, 219520, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1960, 112), (112, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 112, 14, 14), (103488, 1, 7392, 528), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-only", action="store_true")
    args = parser.parse_args()
    inputs = get_inputs()
    kernel_function(*inputs)
    torch.cuda.synchronize()
    if not args.profile_only:
        print("Kernel executed successfully.")
