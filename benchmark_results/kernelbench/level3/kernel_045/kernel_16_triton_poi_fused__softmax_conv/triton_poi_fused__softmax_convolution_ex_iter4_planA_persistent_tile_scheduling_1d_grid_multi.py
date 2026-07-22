import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_16(
        in_ptr0, out_ptr0, ynumel, xnumel,
        YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Persistent 1D grid: each CTA processes multiple tiles via a while loop
    num_x_tiles = tl.cdiv(xnumel, XBLOCK)   # = cdiv(9,16) = 1
    num_y_tiles = tl.cdiv(ynumel, YBLOCK)   # = cdiv(131072,1024) = 128
    num_tiles = num_x_tiles * num_y_tiles    # = 128
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)        # = 128 (grid size)
    tile_id = pid
    while tile_id < num_tiles:
        x_tile = tile_id % num_x_tiles
        y_tile = tile_id // num_x_tiles
        xoffset = x_tile * XBLOCK
        yoffset = y_tile * YBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xmask = xindex < xnumel
        ymask = yindex < ynumel
        x2 = xindex
        y3 = yindex
        y0 = (yindex % 256)
        y1 = yindex // 256
        tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 256 * x2 + 2304 * y1), tmp0, xmask & ymask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 131072
    xnumel = 9
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__softmax_convolution_exp_max_pool2d_with_indices_sub_16[grid](arg_0, arg_1, 131072, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda:0', dtype=torch.float32)
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
