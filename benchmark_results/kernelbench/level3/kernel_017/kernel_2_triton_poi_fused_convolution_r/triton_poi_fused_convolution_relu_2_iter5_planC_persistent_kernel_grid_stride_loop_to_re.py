import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_relu_2(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK: tl.constexpr):
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    pid = tl.program_id(0)
    grid_size = tl.num_programs(0)
    tile_id = pid
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        tmp0 = tl.load(in_ptr0 + xindex, xmask, eviction_policy='evict_first')
        tmp1 = tl.full([1], 0, tl.int32)
        tmp2 = tl.maximum(tmp1, tmp0)
        tl.store(out_ptr0 + xindex, tmp2, xmask, eviction_policy='evict_first')
        tl.store(out_ptr1 + xindex, tmp2, xmask, eviction_policy='evict_last')
        tile_id += grid_size

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_relu_2[grid](arg_0, arg_1, arg_2, 50331648, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((8388608, 6), (6, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 6, 256, 256), (393216, 1, 1536, 6), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 6, 256, 256), (393216, 1, 1536, 6), device='cuda:0', dtype=torch.float32)
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
