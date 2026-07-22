import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_gelu_view_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    num_sms = tl.num_programs(0)
    sm_id = tl.program_id(0)   # 0..num_sms-1, one CTA per SM
    tile_id = sm_id             # first tile assigned to this SM
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        # xnumel=6021120 exactly divisible by XBLOCK=128 -- no mask needed
        x0 = xindex % 768       # bias index, stays hot in L1 across tiles
        tmp0 = tl.load(in_ptr0 + x0, eviction_policy='evict_last')
        tmp1 = tl.load(in_out_ptr0 + xindex)
        tmp2 = tmp0 + tmp1
        # GELU: 0.5 * x * (1 + erf(x / sqrt(2)))
        # Use FMA: half*(tmp7+1) = half*tmp7 + half = fma(half, tmp7, half)
        tmp6 = tmp2 * 0.7071067811865476
        tmp7 = libdevice.erf(tmp6)
        half = tmp2 * 0.5
        tmp10 = tl.math.fma(half, tmp7, half)  # half*tmp7 + half = half*(tmp7+1)
        tl.store(in_out_ptr0 + xindex, tmp10)
        tile_id += num_sms      # stride by SM count

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
