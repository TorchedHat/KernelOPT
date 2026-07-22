import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_clone_permute_roll_view_46(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 7)
    x2 = ((xindex // 1344) % 7)
    x3 = ((xindex // 9408) % 4)
    x4 = ((xindex // 37632) % 4)
    x5 = xindex // 150528
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 192*(((3 + x1 + 7*x3) % 28)) + 5376*(((3 + x2 + 7*x4) % 28)) + 150528*x5), xmask)
    tl.store(out_ptr0 + (x6), tmp0, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 1505280
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_clone_permute_roll_view_46[grid](arg_0, arg_1, 1505280, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 784, 192), (150528, 192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 4, 4, 7, 7, 192), (150528, 37632, 9408, 1344, 192, 1), device='cuda:0', dtype=torch.float32)
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
