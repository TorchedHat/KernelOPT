import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_x_tiles = tl.cdiv(xnumel, XBLOCK)
    num_y_tiles = tl.cdiv(ynumel, YBLOCK)
    total_tiles = num_x_tiles * num_y_tiles
    tile_id = pid
    while tile_id < total_tiles:
        y_tile = tile_id // num_x_tiles
        x_tile = tile_id % num_x_tiles
        yoffset = y_tile * YBLOCK
        xoffset = x_tile * XBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        ymask = yindex < ynumel
        xmask = xindex < xnumel
        x2 = xindex
        y0 = yindex % 128
        y1 = yindex // 128
        y3 = yindex
        tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 8192*y1), mask=xmask & ymask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + y0, mask=ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (x2 + 64*y3), tmp0 + tmp1, mask=xmask & ymask)
        tile_id += grid_size

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 1280
    xnumel = 64
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 1280, 64, YBLOCK=8, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 128, 8, 8), (8192, 1, 1024, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 128, 8, 8), (8192, 64, 8, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
