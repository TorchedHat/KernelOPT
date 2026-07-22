import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_12(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Persistent kernel: each SM loops over multiple tiles
    # Fixed: ynumel=12288, xnumel=9, XBLOCK=16, YBLOCK=8
    # num_tiles_x = cdiv(9, 16) = 1, num_tiles_y = cdiv(12288, 8) = 1536
    # total_tiles = 1 * 1536 = 1536
    num_tiles_x = tl.cdiv(xnumel, XBLOCK)   # = 1
    num_tiles_y = tl.cdiv(ynumel, YBLOCK)    # = 1536
    total_tiles = num_tiles_x * num_tiles_y   # = 1536
    sm_id = tl.program_id(0)
    num_sms = tl.num_programs(0)             # == 132
    tile_id = sm_id
    while tile_id < total_tiles:
        tile_y = tile_id // num_tiles_x
        tile_x = tile_id % num_tiles_x
        yoffset = tile_y * YBLOCK
        xoffset = tile_x * XBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        ymask = yindex < ynumel
        xmask = xindex < xnumel
        x2 = xindex
        y3 = yindex
        y0 = yindex % 96
        y1 = yindex // 96
        tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 96 * x2 + 864 * y1), tmp0, xmask & ymask)
        tile_id += num_sms

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 12288
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_12[grid](arg_0, arg_1, 12288, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 96, 3, 3), (864, 1, 288, 96), device='cuda:0', dtype=torch.float32)
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
