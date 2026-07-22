import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32,  'XBLOCK': 16}, num_warps=2),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 16}, num_warps=4),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4),
        triton.Config({'YBLOCK': 256, 'XBLOCK': 16}, num_warps=8),
        triton.Config({'YBLOCK': 512, 'XBLOCK': 16}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_67(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    ynumel = 51200
    xnumel = 9
    num_tiles_x = (xnumel + XBLOCK - 1) // XBLOCK
    num_tiles_y = (ynumel + YBLOCK - 1) // YBLOCK
    num_tiles = num_tiles_x * num_tiles_y
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tile_id = pid
    while tile_id < num_tiles:
        tile_x = tile_id % num_tiles_x
        tile_y = tile_id // num_tiles_x
        xoffset = tile_x * XBLOCK
        yoffset = tile_y * YBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xmask = xindex < xnumel
        ymask = yindex < ynumel
        x2 = xindex
        y3 = yindex
        y0 = yindex % 160
        y1 = yindex // 160
        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 160*x2 + 1440*y1), tmp0, xmask & ymask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 51200
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_67[grid](arg_0, arg_1, 51200, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((320, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((320, 160, 3, 3), (1440, 1, 480, 160), device='cuda:0', dtype=torch.float32)
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
