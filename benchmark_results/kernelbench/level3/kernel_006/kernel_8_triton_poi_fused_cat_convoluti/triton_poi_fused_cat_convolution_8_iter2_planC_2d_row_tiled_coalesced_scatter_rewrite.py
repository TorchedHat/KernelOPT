import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_cat_convolution_8(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    # 2D row-tiled kernel: each CTA processes ROWS_PER_BLOCK=4 rows x COLS=64 cols
    # XBLOCK = ROWS_PER_BLOCK * COLS = 4 * 64 = 256 (set at launch)
    COLS: tl.constexpr = 64
    ROWS_PER_BLOCK: tl.constexpr = XBLOCK // COLS  # = 4 when XBLOCK=256
    pid = tl.program_id(0)
    row_start = pid * ROWS_PER_BLOCK
    total_rows = xnumel // COLS  # = 501760
    # 2D index grid: [ROWS_PER_BLOCK, COLS]
    rows = row_start + tl.arange(0, ROWS_PER_BLOCK)[:, None]  # [4, 1]
    cols = tl.arange(0, COLS)[None, :]                         # [1, 64]
    row_mask = rows < total_rows  # guard last partial block
    # Input: flat contiguous read — in_ptr0[row*64 + col]
    tmp0 = tl.load(in_ptr0 + rows * COLS + cols, mask=row_mask, other=0.0)
    # Output: coalesced store — out_ptr0[col + 512*row]
    # 64 consecutive cols per row = 256 bytes = 4 cache lines, fully coalesced
    tl.store(out_ptr0 + cols + 512 * rows, tmp0, mask=row_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 32112640
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_convolution_8[grid](arg_0, arg_1, 32112640, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((501760, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 64, 224, 224), (25690112, 1, 114688, 512), device='cuda:0', dtype=torch.float32)
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
