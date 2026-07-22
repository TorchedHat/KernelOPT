import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=4),
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_19(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 192)
    x1 = ((xindex // 192) % 28)
    x2 = xindex // 5376
    x3 = xindex // 192
    x4 = xindex
    tmp27 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
    tmp6 = tl.load(in_ptr0 + (128 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr0 + (7168 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr0 + (7296 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 4.0, tl.float32)
    tmp13 = (tmp11 / tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 160, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-128) + x0)), mask=tmp19 & xmask, eviction_policy='evict_first', other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 192, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tl.load(in_ptr2 + (32*x3 + ((-160) + x0)), mask=tmp21 & xmask, eviction_policy='evict_first', other=0.0)
    tmp25 = tl.where(tmp19, tmp20, tmp24)
    tmp26 = tl.where(tmp4, tmp15, tmp25)
    tmp28 = tmp26 - tmp27
    tmp30 = tl.full([1], 1e-05, tl.float32)
    tmp31 = tmp29 + tmp30
    tmp32 = tl.sqrt_rn(tmp31)
    tmp33 = tl.full([1], 1, tl.int32)
    tmp34 = (tmp33 / tmp32)
    tmp35 = tl.full([1], 1.0, tl.float32)
    tmp36 = tmp34 * tmp35
    tmp37 = tmp28 * tmp36
    tmp39 = tmp37 * tmp38
    tmp41 = tmp39 + tmp40
    tmp42 = tl.full([1], 0, tl.int32)
    tmp43 = tl.maximum(tmp42, tmp41)
    tl.store(in_out_ptr0 + (x4), tmp43, mask=xmask, cache_modifier='.cs')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    arg_7 = inputs[7]
    xnumel = 1505280
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_19[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 1505280)
    return arg_7

def get_inputs():
    arg_0 = rand_strided((10, 192, 28, 28), (150528, 1, 5376, 192), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((31360, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((192,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7]

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
