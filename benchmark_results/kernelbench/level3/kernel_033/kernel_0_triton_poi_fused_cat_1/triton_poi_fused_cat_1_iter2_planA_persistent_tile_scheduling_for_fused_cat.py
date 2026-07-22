import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_cat_1(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    tile_id = pid
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex % 32768
        x1 = xindex // 32768
        tmp3 = tl.full([1], 16384, tl.int64)
        tmp4 = x0 < tmp3
        tmp6 = x0 >= tmp3
        mask_load0 = xmask & tmp4
        mask_load1 = xmask & tmp6
        tmp5 = tl.load(in_ptr0 + (16384 * x1 + x0), mask=mask_load0, eviction_policy='evict_last', other=0.0)
        tmp9 = tl.load(in_ptr1 + (16384 * x1 + ((-16384) + x0)), mask=mask_load1, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.where(tmp4, tmp5, tmp9)
        tl.store(out_ptr0 + xindex, tmp10, mask=xmask)
        tile_id += grid_size

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 8388608
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_1[grid](arg_0, arg_1, arg_2, 8388608, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 32768), (32768, 1), device='cuda:0', dtype=torch.float32)
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
