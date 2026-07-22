import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 32}, num_warps=8),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 32}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 501760
    xnumel = 160
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 50176)
    y1 = yindex // 50176
    y3 = yindex
    tmp29 = tl.load(in_ptr5 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr7 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr8 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (y0 + 50176*(x2) + 1605632*y1), mask=tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1, 1], 64, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tmp6 & tmp8
    tmp10 = tl.load(in_ptr1 + (32*y3 + ((-32) + x2)), mask=tmp9 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp0 >= tmp7
    tmp12 = tl.full([1, 1], 96, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tmp11 & tmp13
    tmp15 = tl.load(in_ptr2 + (32*y3 + ((-64) + x2)), mask=tmp14 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp0 >= tmp12
    tmp17 = tl.full([1, 1], 128, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr3 + (32*y3 + ((-96) + x2)), mask=tmp19 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1, 1], 160, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr4 + (32*y3 + ((-128) + x2)), mask=tmp21 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tl.where(tmp14, tmp15, tmp25)
    tmp27 = tl.where(tmp9, tmp10, tmp26)
    tmp28 = tl.where(tmp4, tmp5, tmp27)
    tmp30 = tmp28 - tmp29
    tmp32 = tl.full([1, 1], 1e-05, tl.float32)
    tmp33 = tmp31 + tmp32
    tmp34 = tl.sqrt_rn(tmp33)
    tmp35 = tl.full([1, 1], 1, tl.int32)
    tmp36 = (tmp35 / tmp34)
    tmp37 = tl.full([1, 1], 1.0, tl.float32)
    tmp38 = tmp36 * tmp37
    tmp39 = tmp30 * tmp38
    tmp41 = tmp39 * tmp40
    tmp43 = tmp41 + tmp42
    tmp44 = tl.full([1, 1], 0, tl.int32)
    tmp45 = tl.maximum(tmp44, tmp43)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 160*y3), tmp45, mask=xmask & ymask)

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
    arg_9 = inputs[9]
    ynumel = 501760
    xnumel = 160
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_8[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 501760, 160)
    return arg_9

def get_inputs():
    arg_0 = rand_strided((10, 160, 224, 224), (8028160, 1, 35840, 160), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 224, 224), (1605632, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((160,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9]

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
