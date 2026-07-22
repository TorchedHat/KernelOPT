import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_gelu_view_55(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    NUM_SMS = tl.num_programs(0)
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    sm_id = tl.program_id(0)
    tile_id = sm_id
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        bias_idx = xindex % 3072
        tmp0 = tl.load(in_ptr0 + bias_idx, xmask)
        tmp1 = tl.load(in_out_ptr0 + xindex, xmask)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1], 0.5, tl.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
        tmp6 = tmp2 * tmp5
        tmp7 = libdevice.erf(tmp6)
        tmp8 = tl.full([1], 1.0, tl.float32)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp4 * tmp9
        tl.store(in_out_ptr0 + xindex, tmp10, xmask)
        tile_id += NUM_SMS

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 1505280
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_gelu_view_55[grid](arg_0, arg_1, 1505280, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 49, 3072), (150528, 3072, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3072,), (1,), device='cuda:0', dtype=torch.float32)
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
