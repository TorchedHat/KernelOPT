import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 512},  num_warps=4),
    triton.Config({'XBLOCK': 512},  num_warps=8),
    triton.Config({'XBLOCK': 1024}, num_warps=4),
    triton.Config({'XBLOCK': 1024}, num_warps=8),
    triton.Config({'XBLOCK': 2048}, num_warps=4),
    triton.Config({'XBLOCK': 2048}, num_warps=8),
], key=['xnumel'])
@triton.jit
def triton_poi_fused_addmm_tanh_3(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)  # in [0, NUM_SMS)
    num_sms = tl.num_programs(0)  # equals grid size = 132
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    tile_id = pid
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        # NO xmask: xnumel=4194304 divisible by all XBLOCK values
        x0 = xindex % 16384  # bias broadcast index
        tmp0 = tl.load(in_ptr0 + x0, eviction_policy='evict_last')
        tmp1 = tl.load(in_out_ptr0 + xindex)
        tmp2 = tmp0 + tmp1
        tmp3 = libdevice.tanh(tmp2)
        tl.store(in_out_ptr0 + xindex, tmp3)
        tile_id += num_sms

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 4194304
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_tanh_3[grid](arg_0, arg_1, 4194304)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((256, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16384,), (1,), device='cuda:0', dtype=torch.float32)
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
