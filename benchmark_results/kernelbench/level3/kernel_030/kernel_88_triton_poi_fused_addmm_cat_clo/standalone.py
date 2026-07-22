import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_addmm_cat_clone_permute_select_view_zeros_like_88(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 32)
    x2 = ((xindex // 1568) % 24)
    x1 = ((xindex // 32) % 49)
    x3 = xindex // 37632
    x4 = xindex
    tmp19 = tl.load(in_ptr2 + (1536 + x0 + 32*x2 + 2304*x1 + 112896*x3), xmask)
    tmp0 = 1536 + x0 + 32*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 768, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1536 + x0 + 32*x2), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 1536, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.full([1], 0.0, tl.float32)
    tmp11 = tl.full(tmp10.shape, 0.0, tmp10.dtype)
    tmp12 = tl.where(tmp9, tmp10, tmp11)
    tmp13 = tmp0 >= tmp7
    tmp14 = tl.full([1], 2304, tl.int64)
    tmp15 = tmp0 < tmp14
    tmp16 = tl.load(in_ptr1 + (x0 + 32*x2), tmp13 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tl.where(tmp9, tmp12, tmp16)
    tmp18 = tl.where(tmp4, tmp5, tmp17)
    tmp20 = tmp18 + tmp19
    tl.store(out_ptr0 + (x4), tmp20, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 376320
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_cat_clone_permute_select_view_zeros_like_88[grid](arg_0, arg_1, arg_2, arg_3, 376320, XBLOCK=128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((490, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 24, 49, 32), (37632, 1568, 32, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
