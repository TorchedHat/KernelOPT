import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_cat_195(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 15680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = (xindex % 32)
    x1 = xindex // 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + 1344*x1), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + 1376*x1), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + 1408*x1), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + 1440*x1), tmp0, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 15680
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_195[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 15680, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 7, 7), (65856, 1, 9408, 1344), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 7, 7), (67424, 1, 9632, 1376), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 7, 7), (68992, 1, 9856, 1408), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 7, 7), (70560, 1, 10080, 1440), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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
