import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16, 'YBLOCK': 32},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8, num_stages=1),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_7(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 9
    pid = tl.program_id(0)
    num_sm = tl.num_programs(0)
    x_tiles = tl.cdiv(xnumel, XBLOCK)
    y_tiles = tl.cdiv(ynumel, YBLOCK)
    total_tiles = x_tiles * y_tiles
    tile_id = pid
    while tile_id < total_tiles:
        xtile = tile_id % x_tiles
        ytile = tile_id // x_tiles
        xoffset = xtile * XBLOCK
        yoffset = ytile * YBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
        yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
        xmask = xindex < xnumel
        ymask = yindex < ynumel
        x2 = xindex
        y3 = yindex
        y0 = yindex % 64
        y1 = yindex // 64
        tmp0 = tl.load(in_ptr0 + (x2 + 9*y3), xmask & ymask, eviction_policy='evict_last')
        tl.store(out_ptr0 + (y0 + 64*x2 + 576*y1), tmp0, xmask & ymask)
        tile_id += num_sm

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 8192
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_7[grid](arg_0, arg_1, 8192, 9)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda:0', dtype=torch.float32)
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
