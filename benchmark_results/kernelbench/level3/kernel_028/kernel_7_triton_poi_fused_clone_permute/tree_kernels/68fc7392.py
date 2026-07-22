import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_permute_7(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_sms = tl.num_programs(0)    # equals grid size = num_sms launched
    num_tiles = tl.cdiv(xnumel, XBLOCK)
    tile_id = pid                   # first tile for this SM
    while tile_id < num_tiles:
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        # Index decomposition UNCHANGED from original
        x0 = (xindex % 512)
        x1 = ((xindex // 512) % 197)
        x2 = xindex // 100864
        tmp0 = tl.load(in_ptr0 + (x0 + 512*x2 + 1024*x1), xmask, cache_modifier='.cg')
        tl.store(out_ptr0 + xindex, tmp0, xmask)
        tile_id += num_sms          # stride by number of persistent CTAs

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 201728
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_permute_7[grid](arg_0, arg_1, 201728, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((197, 8, 2, 64), (1024, 64, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2, 197, 8, 64), (100864, 512, 64, 1), device='cuda:0', dtype=torch.float32)
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
