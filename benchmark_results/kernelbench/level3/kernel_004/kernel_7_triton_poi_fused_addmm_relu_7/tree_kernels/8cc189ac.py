import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_relu_7(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    num_sms = tl.num_programs(0)
    tile_id = pid
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = (xindex % 84)
        x2 = xindex
        # bias stays hot in L1 across all iterations (evict_last)
        tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
        # streaming tensor: evict_first to avoid L1 pollution
        tmp1 = tl.load(in_out_ptr0 + (x2), xmask, eviction_policy='evict_first')
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1], 0, tl.int32)
        tmp4 = tl.maximum(tmp3, tmp2)
        tl.store(in_out_ptr0 + (x2), tmp4, xmask)
        tile_id += num_sms

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 344064
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_relu_7[grid](arg_0, arg_1, 344064, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((4096, 84), (84, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((84,), (1,), device='cuda:0', dtype=torch.float32)
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
