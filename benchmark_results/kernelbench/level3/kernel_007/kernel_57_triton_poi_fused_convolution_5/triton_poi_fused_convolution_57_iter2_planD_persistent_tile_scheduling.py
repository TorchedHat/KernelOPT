import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_57(in_ptr0, out_ptr0, ynumel, xnumel,
                                    YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Persistent kernel: each CTA iterates over multiple tiles via grid-stride loop
    # num_x_tiles=cdiv(9,XBLOCK), num_y_tiles=cdiv(41472,YBLOCK), total_tiles=product
    num_x_tiles = tl.cdiv(9, XBLOCK)
    num_y_tiles = tl.cdiv(41472, YBLOCK)
    total_tiles = num_x_tiles * num_y_tiles

    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tile_id = pid
    while tile_id < total_tiles:
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
        y0 = yindex % 144
        y1 = yindex // 144
        combined_mask = xmask & ymask
        tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask=combined_mask, other=0.0, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 144 * x2 + 1296 * y1), tmp0, mask=combined_mask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 41472
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_57[grid](arg_0, arg_1, 41472, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((288, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((288, 144, 3, 3), (1296, 1, 432, 144), device='cuda:0', dtype=torch.float32)
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
