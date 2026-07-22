import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=3),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5017600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 160)
    x1 = xindex // 160
    x2 = xindex
    tmp23 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (64*x1 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 96, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (32*x1 + ((-64) + x0)), mask=tmp9 & xmask, eviction_policy='evict_first', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (32*x1 + ((-96) + x0)), mask=tmp14 & xmask, eviction_policy='evict_first', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1], 160, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr3 + (32*x1 + ((-128) + x0)), mask=tmp16 & xmask, eviction_policy='evict_first', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = tl.full([1], 1e-05, tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.math.rsqrt(tmp27)
    tmp33 = tmp24 * tmp28
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.full([1], 0, tl.int32)
    tmp39 = tl.maximum(tmp38, tmp37)
    tl.store(in_out_ptr0 + (x2), tmp39, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    arg_7 = inputs[7]
    arg_8 = inputs[8]
    xnumel = 5017600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 5017600)
    return arg_8

def get_inputs():
    arg_0 = rand_strided((10, 160, 56, 56), (501760, 1, 8960, 160), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8]

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
