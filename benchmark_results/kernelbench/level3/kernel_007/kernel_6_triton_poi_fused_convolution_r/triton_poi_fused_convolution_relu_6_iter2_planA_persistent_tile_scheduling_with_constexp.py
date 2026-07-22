import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_relu_6(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    YNUMEL: tl.constexpr = 12288
    XNUMEL: tl.constexpr = 9
    num_y_tiles = tl.cdiv(YNUMEL, YBLOCK)
    num_x_tiles = tl.cdiv(XNUMEL, XBLOCK)
    total_tiles = num_y_tiles * num_x_tiles

    num_sms = tl.num_programs(0)
    tile_id = tl.program_id(0)
    while tile_id < total_tiles:
        y_tile = tile_id // num_x_tiles
        x_tile = tile_id % num_x_tiles

        yoffset = y_tile * YBLOCK
        xoffset = x_tile * XBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        ymask = yindex < YNUMEL
        xmask = xindex < XNUMEL

        x2 = xindex
        y0 = yindex % 64
        y1 = yindex // 64
        y3 = yindex

        tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 64 * x2 + 576 * y1), tmp0, xmask & ymask)

        tile_id += num_sms

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 12288
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_relu_6[grid](arg_0, arg_1, 12288, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((192, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((192, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
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
