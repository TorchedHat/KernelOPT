import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_gelu_view_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    # Persistent kernel: each CTA iterates over multiple tiles
    # num_tiles = 6021120 / 128 = 47040 (exact, no remainder)
    # num_programs = 132 * 4 = 528 CTAs
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    tile_id = pid
    while tile_id * XBLOCK < xnumel:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        x0 = xindex % 768
        tmp0 = tl.load(in_ptr0 + x0, eviction_policy='evict_last')
        tmp1 = tl.load(in_out_ptr0 + xindex)
        tmp2 = tmp0 + tmp1
        erf_arg = libdevice.fma(tmp2, 0.7071067811865476, 0.0)
        erf_val = libdevice.erf(erf_arg)
        gelu_gate = erf_val + 1.0
        tmp10 = libdevice.fma(tmp2, 0.5 * gelu_gate, 0.0)
        tl.store(in_out_ptr0 + xindex, tmp10)
        tile_id += num_programs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 6021120
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_gelu_view_20[grid](arg_0, arg_1, 6021120, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
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
