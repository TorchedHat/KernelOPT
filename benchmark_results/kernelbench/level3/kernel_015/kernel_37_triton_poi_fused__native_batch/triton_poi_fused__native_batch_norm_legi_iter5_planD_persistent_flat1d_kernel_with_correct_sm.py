import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_ytiles = tl.cdiv(ynumel, YBLOCK)
    total_xtiles = tl.cdiv(xnumel, XBLOCK)
    total_tiles = total_ytiles * total_xtiles
    tile_id = pid
    while tile_id < total_tiles:
        ytile = tile_id // total_xtiles
        xtile = tile_id % total_xtiles
        yoffset = ytile * YBLOCK
        xoffset = xtile * XBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)
        xindex = xoffset + tl.arange(0, XBLOCK)
        ymask = yindex < ynumel
        xmask = xindex < xnumel
        y3 = yindex[:, None]
        x2 = xindex[None, :]
        y0 = y3 % 448
        y1 = y3 // 448
        mask = ymask[:, None] & xmask[None, :]
        tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask=mask, other=0.0)
        tl.store(out_ptr0 + (y0 + 448 * x2 + 4032 * y1), tmp0, mask=mask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 14336
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37[grid](arg_0, arg_1, 14336, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 448, 3, 3), (4032, 1, 1344, 448), device='cuda:0', dtype=torch.float32)
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
