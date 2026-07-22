import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 256},  num_warps=8),
        triton.Config({'XBLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=4),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_cat_223(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    tile_id = pid
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex % 32
        x1 = xindex // 32
        tmp0 = tl.load(in_ptr0 + xindex, xmask)
        tl.store(out_ptr0 + (x0 + 1792 * x1), tmp0, xmask)
        tl.store(out_ptr1 + (x0 + 1824 * x1), tmp0, xmask)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 15680
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_223[grid](arg_0, arg_1, arg_2, 15680)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 7, 7), (87808, 1, 12544, 1792), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 7, 7), (89376, 1, 12768, 1824), device='cuda:0', dtype=torch.float32)
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
