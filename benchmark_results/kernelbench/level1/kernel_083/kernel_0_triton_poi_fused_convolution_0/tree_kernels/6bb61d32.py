import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Persistent kernel: fixed XBLOCK=128, YBLOCK=8
    # NUM_X_TILES = cdiv(50176, 128) = 393
    # NUM_Y_TILES = cdiv(30, 8) = 4
    # TOTAL_TILES = 393 * 4 = 1572
    NUM_X_TILES: tl.constexpr = 393
    TOTAL_TILES: tl.constexpr = 1572
    pid = tl.program_id(0)
    grid_stride = tl.num_programs(0)
    tile_id = pid
    while tile_id < TOTAL_TILES:
        ytile = tile_id // NUM_X_TILES
        xtile = tile_id % NUM_X_TILES
        yoffset = ytile * YBLOCK
        xoffset = xtile * XBLOCK
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        ymask = yindex < 30
        xindex = tl.max_contiguous(tl.multiple_of(xoffset + tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
        xmask = xindex < 50176
        x2 = xindex
        y3 = yindex
        y0 = (yindex % 3)
        y1 = yindex // 3
        tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask)
        tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask, eviction_policy='evict_first')
        tile_id += grid_stride

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
