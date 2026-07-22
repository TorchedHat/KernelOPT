import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_gelu_view_4(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    tile_id = pid
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex % 384
        x2 = xindex
        tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_out_ptr0 + x2, xmask)
        tmp2 = tmp0 + tmp1
        tmp6 = tmp2 * 0.7071067811865476
        tmp7 = libdevice.erf(tmp6)
        tmp8 = tmp7 + 1.0
        tmp9 = tmp2 * 0.5
        tmp10 = tmp9 * tmp8
        tl.store(in_out_ptr0 + x2, tmp10, xmask, eviction_policy='evict_first')
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 12042240
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_gelu_view_4[grid](arg_0, arg_1, 12042240, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3136, 384), (1204224, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
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
