import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_22(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Persistent 1D kernel: flatten all ynumel*xnumel elements into a single linear index
    # Use YBLOCK as the tile block size (XBLOCK is ignored in this formulation)
    numel = ynumel * xnumel  # = 24576 * 9 = 221184
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tile_start = pid * YBLOCK
    stride = num_programs * YBLOCK
    while tile_start < numel:
        offsets = tile_start + tl.arange(0, YBLOCK)
        mask = offsets < numel
        # Recover 2D indices from flat index: flat = x2 + 9*y3
        y3 = offsets // xnumel   # in [0, 24576)
        x2 = offsets % xnumel    # in [0, 9)
        y0 = y3 % 128
        y1 = y3 // 128
        # Load: in_ptr0 + x2 + xnumel*y3 == in_ptr0 + offsets (algebraic identity)
        tmp0 = tl.load(in_ptr0 + offsets, mask=mask)
        # Store: permuted layout
        tl.store(out_ptr0 + (y0 + 128*x2 + 1152*y1), tmp0, mask=mask)
        tile_start += stride

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 24576
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_22[grid](arg_0, arg_1, 24576, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((192, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 128, 3, 3), (1152, 1, 384, 128), device='cuda:0', dtype=torch.float32)
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
