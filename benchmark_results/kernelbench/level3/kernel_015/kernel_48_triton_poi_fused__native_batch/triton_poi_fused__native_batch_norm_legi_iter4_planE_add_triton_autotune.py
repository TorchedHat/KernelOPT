import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=2),
        triton.Config({'XBLOCK': 128},  num_warps=4),
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 384)
    x1 = ((xindex // 384) % 14)
    x2 = xindex // 5376
    x3 = xindex // 384
    x4 = xindex
    tmp39 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp52 = tl.load(in_ptr8 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 4.0, tl.float32)
    tmp13 = (tmp11 / tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 288, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), mask=tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 320, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), mask=tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 352, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-320) + x0)), mask=tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 384, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tl.load(in_ptr4 + (32*x3 + ((-352) + x0)), mask=tmp31 & xmask, eviction_policy='evict_last', other=0.0)
    tmp35 = tl.where(tmp29, tmp30, tmp34)
    tmp36 = tl.where(tmp24, tmp25, tmp35)
    tmp37 = tl.where(tmp19, tmp20, tmp36)
    tmp38 = tl.where(tmp4, tmp15, tmp37)
    tmp40 = tmp38 - tmp39
    tmp42 = tl.full([1], 1e-05, tl.float32)
    tmp43 = tmp41 + tmp42
    tmp48 = libdevice.rsqrt(tmp43)
    tmp49 = tmp40 * tmp48
    tmp51 = tmp49 * tmp50
    tmp53 = tmp51 + tmp52
    tmp54 = tl.full([1], 0, tl.int32)
    tmp55 = tl.maximum(tmp54, tmp53)
    tl.store(in_out_ptr0 + (x4), tmp55, mask=xmask)

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
    xnumel = 752640
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_48[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, 752640)
    return arg_9

def get_inputs():
    arg_0 = rand_strided((10, 384, 14, 14), (75264, 1, 5376, 384), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((7840, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
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
