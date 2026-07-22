import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK':  1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK':  2}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK':  4}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK':  8}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 501760
    xnumel = 128
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
    tmp23 = tl.load(in_ptr4 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr5 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr6 + (x2), mask=xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr7 + (x2), mask=xmask, eviction_policy='evict_last')
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
    tmp19 = tl.load(in_ptr3 + (32*y3 + ((-96) + x2)), mask=tmp16 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.where(tmp14, tmp15, tmp19)
    tmp21 = tl.where(tmp9, tmp10, tmp20)
    tmp22 = tl.where(tmp4, tmp5, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = tl.full([1, 1], 1e-05, tl.float32)
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt_rn(tmp27)
    tmp29 = tl.full([1, 1], 1, tl.int32)
    tmp30 = (tmp29 / tmp28)
    tmp31 = tl.full([1, 1], 1.0, tl.float32)
    tmp32 = tmp30 * tmp31
    tmp33 = tmp24 * tmp32
    tmp35 = tmp33 * tmp34
    tmp37 = tmp35 + tmp36
    tmp38 = tl.full([1, 1], 0, tl.int32)
    tmp39 = tl.maximum(tmp38, tmp37)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + 128*y3), tmp39, mask=xmask & ymask)

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
    ynumel = 501760
    xnumel = 128
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, 501760, 128)
    return arg_8

def get_inputs():
    arg_0 = rand_strided((10, 128, 224, 224), (6422528, 1, 28672, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 32, 224, 224), (1605632, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 224, 224), (1605632, 1, 7168, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
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
