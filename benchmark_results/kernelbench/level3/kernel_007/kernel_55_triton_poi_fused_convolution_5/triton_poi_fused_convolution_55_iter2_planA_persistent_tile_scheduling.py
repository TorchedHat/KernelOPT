import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_55(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)  # == 132 at runtime
    num_tiles = tl.cdiv(501760, XBLOCK)  # = 3920 for XBLOCK=128
    for tile_id in tl.range(pid, num_tiles, num_programs):
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x2 = xindex
        x0 = (xindex % 256)
        x1 = xindex // 256
        tmp0 = tl.load(in_ptr0 + (x2), xmask)
        tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
        tmp2 = tmp0 + tmp1
        tl.store(out_ptr0 + (x0 + 512*x1), tmp2, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 501760
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_55[grid](arg_0, arg_1, arg_2, 501760, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((10, 256, 14, 14), (50176, 1, 3584, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 256, 14, 14), (100352, 1, 7168, 512), device='cuda:0', dtype=torch.float32)
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
