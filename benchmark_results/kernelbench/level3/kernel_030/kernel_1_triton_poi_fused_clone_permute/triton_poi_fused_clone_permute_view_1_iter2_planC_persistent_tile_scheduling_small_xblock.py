import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_clone_permute_view_1(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    for tile_id in tl.range(pid, xnumel, num_programs):
        xoffset = tile_id * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = tl.full([XBLOCK], True, tl.int1)
        x0 = (xindex % 672)
        x1 = ((xindex // 672) % 7)
        x2 = ((xindex // 4704) % 8)
        x3 = xindex // 37632
        x4 = xindex
        tmp0 = tl.load(in_ptr0 + (x0 + 672*x2 + 5376*x1 + 37632*x3), xmask)
        tl.store(out_ptr0 + (x4), tmp0, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 3010560
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_permute_view_1[grid](arg_0, arg_1, 3010560, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 8, 8, 7, 7, 96), (301056, 37632, 4704, 672, 96, 1), device='cuda:0', dtype=torch.float32)
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
