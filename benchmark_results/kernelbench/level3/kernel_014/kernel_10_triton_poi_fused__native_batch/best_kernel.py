import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8,   'YBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 8,   'YBLOCK': 64}, num_warps=4),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 16,  'YBLOCK': 64}, num_warps=8),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=8),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=8),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 16}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 501760
    xnumel = 192
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
    tmp35 = tl.load(in_ptr6 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr8 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr9 + (x2), mask=xmask, eviction_policy='evict_last')
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
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr4 + (32*y3 + ((-128) + x2)), mask=tmp24 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1, 1], 192, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tl.load(in_ptr5 + (32*y3 + ((-160) + x2)), mask=tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.where(tmp24, tmp25, tmp29)
    tmp31 = tl.where(tmp19, tmp20, tmp30)
    tmp32 = tl.where(tmp14, tmp15, tmp31)
    tmp33 = tl.where(tmp9, tmp10, tmp32)
    tmp34 = tl.where(tmp4, tmp5, tmp33)
    tmp36 = tmp34 - tmp35
    tmp38 = tl.full([1, 1], 1e-05, tl.float32)
    tmp39 = tmp37 + tmp38
    tmp40 = tl.sqrt_rn(tmp39)
    tmp41 = tl.full([1, 1], 1, tl.int32)
    tmp42 = (tmp41 / tmp40)
    tmp43 = tl.full([1, 1], 1.0, tl.float32)
    tmp44 = tmp42 * tmp43
    tmp45 = tmp36 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = tl.full([1, 1], 0, tl.int32)
    tmp51 = tl.maximum(tmp50, tmp49)
    tl.store(in_out_ptr0 + (x2 + 192*y3), tmp51, mask=xmask & ymask)

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
    arg_10 = inputs[10]
    ynumel = 501760
    xnumel = 192
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_10[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 501760, 192)
    return arg_10

def get_inputs():
    arg_0 = rand_strided((10, 192, 224, 224), (9633792, 1, 43008, 192), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 224, 224), (1605632, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10]

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
