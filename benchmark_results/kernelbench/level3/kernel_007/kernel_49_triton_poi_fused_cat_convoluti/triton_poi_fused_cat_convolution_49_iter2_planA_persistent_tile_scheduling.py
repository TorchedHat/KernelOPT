import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_cat_convolution_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 313600
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tile_id = pid
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = (xindex % 160)
        x1 = xindex // 160
        tmp0 = tl.load(in_ptr0 + xindex, xmask)
        tl.store(out_ptr0 + (x0 + 512*x1), tmp0, xmask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 313600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_convolution_49[grid](arg_0, arg_1, 313600, XBLOCK=128)
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
