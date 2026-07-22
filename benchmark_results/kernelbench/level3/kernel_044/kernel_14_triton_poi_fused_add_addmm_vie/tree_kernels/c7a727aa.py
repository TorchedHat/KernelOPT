import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_addmm_view_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK: tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < xnumel
    col = xindex % 768
    bias1 = tl.load(in_ptr1 + col, mask=xmask)
    bias2 = tl.load(in_ptr2 + col, mask=xmask)
    v0 = tl.load(in_ptr0 + xindex, mask=xmask)
    v2 = tl.load(in_out_ptr0 + xindex, mask=xmask)
    v6 = tl.load(in_ptr3 + xindex, mask=xmask)
    result = v0 + v2 + v6 + bias1 + bias2
    tl.store(in_out_ptr0 + xindex, result, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 50331648
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_addmm_view_14[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 50331648, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((65536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
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
