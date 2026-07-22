import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=3),  # PRIMARY: pipeline reduces live regs
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=3),  # smaller tile, fewer live regs
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=2),  # smaller tile baseline
        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=2),  # fallback: prior best
        triton.Config({'XBLOCK': 64},  num_warps=1, num_stages=3),  # minimal warps, max blocks/SM
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_23(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2257920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 288)
    x1 = ((xindex // 288) % 28)
    x2 = xindex // 8064
    x3 = xindex // 288
    x4 = xindex
    tmp45 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr8 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp58 = tl.load(in_ptr9 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (128 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr0 + (7168 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr0 + (7296 + 256*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 4.0, tl.float32)
    tmp13 = (tmp11 / tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 160, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-128) + x0)), mask=tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 192, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-160) + x0)), mask=tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 224, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-192) + x0)), mask=tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 256, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tl.load(in_ptr4 + (32*x3 + ((-224) + x0)), mask=tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp0 >= tmp32
    tmp37 = tl.full([1], 288, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tl.load(in_ptr5 + (32*x3 + ((-256) + x0)), mask=tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp40 = tl.where(tmp34, tmp35, tmp39)
    tmp41 = tl.where(tmp29, tmp30, tmp40)
    tmp42 = tl.where(tmp24, tmp25, tmp41)
    tmp43 = tl.where(tmp19, tmp20, tmp42)
    tmp44 = tl.where(tmp4, tmp15, tmp43)
    tmp46 = tmp44 - tmp45
    tmp48 = tl.full([1], 1e-05, tl.float32)
    tmp49 = tmp47 + tmp48
    tmp50 = libdevice.rsqrt(tmp49)
    tmp55 = tmp46 * tmp50
    tmp57 = tmp55 * tmp56
    tmp59 = tmp57 + tmp58
    tmp60 = tl.full([1], 0, tl.int32)
    tmp61 = tl.maximum(tmp60, tmp59)
    tl.store(in_out_ptr0 + (x4), tmp61, mask=xmask)

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
    xnumel = 2257920
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_23[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 2257920)
    return arg_10

def get_inputs():
    arg_0 = rand_strided((10, 288, 28, 28), (225792, 1, 8064, 288), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((31360, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 32, 28, 28), (25088, 1, 896, 32), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((288,), (1,), device='cuda:0', dtype=torch.float32)
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
