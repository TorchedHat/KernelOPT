import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},  num_warps=1),
        triton.Config({'XBLOCK': 2},  num_warps=1),
        triton.Config({'XBLOCK': 4},  num_warps=2),
        triton.Config({'XBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 32}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_cat_convolution_49(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # XBLOCK controls rows per block (YBLOCK in plan terminology)
    # ROW_WIDTH=160, padded to next power of 2 = 256 for tl.arange
    COL_PAD: tl.constexpr = 256    # next power of 2 >= 160
    ROW_WIDTH: tl.constexpr = 160  # actual number of columns
    OUT_STRIDE: tl.constexpr = 512  # output stride in row dimension
    ynumel = 1960                   # total rows = 313600 / 160

    y_start = tl.program_id(0) * XBLOCK
    y_offs = y_start + tl.arange(0, XBLOCK)   # shape [XBLOCK] = rows
    x_offs = tl.arange(0, COL_PAD)            # shape [256] = padded columns

    y2d = y_offs[:, None]   # [XBLOCK, 1]
    x2d = x_offs[None, :]   # [1, 256]

    ymask = y2d < ynumel     # guard rows
    xmask = x2d < ROW_WIDTH  # guard columns (only first 160 are valid)
    mask = ymask & xmask

    # Input: contiguous (1960, 160), stride (160, 1)
    in_idx = y2d * ROW_WIDTH + x2d   # [XBLOCK, 256]
    tmp0 = tl.load(in_ptr0 + in_idx, mask, other=0.0)

    # Output: strided, stride (512, 1) in (row, col)
    out_idx = y2d * OUT_STRIDE + x2d  # [XBLOCK, 256]
    tl.store(out_ptr0 + out_idx, tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 313600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_convolution_49[grid](arg_0, arg_1, 313600)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1960, 160), (160, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 160, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
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
