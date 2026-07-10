import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_convolution_gelu_native_group_norm_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8520192
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y1 = yindex // 66564
    y0 = (yindex % 66564)
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (8*y1 + (x2 // 8)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (8*y1 + (x2 // 8)), xmask & ymask, eviction_policy='evict_last')
    tmp20 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1, 1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp12 = tmp10 - tmp11
    tmp14 = tl.full([1, 1], 532512.0, tl.float32)
    tmp15 = (tmp13 / tmp14)
    tmp16 = tl.full([1, 1], 1e-05, tl.float32)
    tmp17 = tmp15 + tmp16
    tmp18 = libdevice.rsqrt(tmp17)
    tmp19 = tmp12 * tmp18
    tmp21 = tmp19 * tmp20
    tmp23 = tmp21 + tmp22
    tl.store(out_ptr0 + (y0 + 66564*x2 + 4260096*y1), tmp23, xmask & ymask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    ynumel = 8520192
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_gelu_native_group_norm_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8520192, 64, YBLOCK=8, XBLOCK=128)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((128, 64, 258, 258), (4260096, 1, 16512, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128, 64, 258, 258), (4260096, 66564, 258, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
