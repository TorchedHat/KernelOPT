import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_clone_transpose_view_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 32)
    x1 = ((xindex // 32) % 3)
    x2 = ((xindex // 96) % 49)
    x3 = xindex // 4704
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 32*x2 + 1568*x1 + 4704*x3), None)
    tl.store(out_ptr0 + (x4), tmp0, None)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 3010560
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_transpose_view_13[grid](arg_0, arg_1, 3010560, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1920, 49, 32), (1568, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((640, 49, 3, 32), (4704, 96, 32, 1), device='cuda:0', dtype=torch.float32)
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
