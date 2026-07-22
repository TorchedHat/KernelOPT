import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_clone_permute_roll_view_72(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 7)
    x2 = ((xindex // 2688) % 7)
    x3 = ((xindex // 18816) % 2)
    x4 = ((xindex // 37632) % 2)
    x5 = xindex // 75264
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 384*(((3 + x1 + 7*x3) % 14)) + 5376*(((3 + x2 + 7*x4) % 14)) + 75264*x5), xmask)
    tl.store(out_ptr0 + (x6), tmp0, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 752640
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_permute_roll_view_72[grid](arg_0, arg_1, 752640, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 2, 2, 7, 7, 384), (75264, 37632, 18816, 2688, 384, 1), device='cuda:0', dtype=torch.float32)
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
