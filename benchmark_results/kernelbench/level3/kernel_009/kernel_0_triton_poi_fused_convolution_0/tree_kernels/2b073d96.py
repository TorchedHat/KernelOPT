import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Persistent kernel: each CTA processes multiple tiles in a while-loop
    num_x_tiles = tl.cdiv(xnumel, XBLOCK)
    num_y_tiles = tl.cdiv(ynumel, YBLOCK)
    total_tiles = num_x_tiles * num_y_tiles
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tile_id = pid
    while tile_id < total_tiles:
        ytile = tile_id // num_x_tiles
        xtile = tile_id % num_x_tiles
        yoffset = ytile * YBLOCK
        xoffset = xtile * XBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        ymask = yindex < ynumel
        xmask = xindex < xnumel
        combined_mask = xmask & ymask
        y0 = yindex % 3
        y1 = yindex // 3
        tmp0 = tl.load(in_ptr0 + (xindex + 50176 * yindex), mask=combined_mask, eviction_policy='evict_first')
        tl.store(out_ptr0 + (y0 + 3 * xindex + 150528 * y1), tmp0, mask=combined_mask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 30
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
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
