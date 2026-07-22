import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x0 = (xindex % 96)
    x1 = ((xindex // 96) % 3136)
    x2 = xindex // 301056
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (32*(((4 + ((x1 % 56))) % 7)) + 224*(((4 + (x1 // 56)) % 7)) + 1568*(x0 // 32) + 4704*((4 + ((x1 % 56))) // 7) + 42336*((4 + (x1 // 56)) // 7) + 381024*x2 + ((x0 % 32))), None)
    tmp2 = tl.load(in_ptr1 + (7*(((4 + (x1 // 56)) % 7)) + 49*(x0 // 32) + (((4 + ((x1 % 56))) % 7))), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x3), None)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.store(in_out_ptr0 + (x3), tmp8, None)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 3010560
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__unsafe_view_add_addmm_clone_constant_pad_nd_convolution_native_layer_norm_permute_slice_transpose_view_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 3010560, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((810, 147, 32), (4704, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((147,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((31360, 96), (96, 1), device='cuda:0', dtype=torch.float32)
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
