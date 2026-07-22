import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 878080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 448)
    x1 = ((xindex // 448) % 14)
    x2 = xindex // 6272
    x3 = xindex // 448
    x4 = xindex
    tmp51 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr8 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp62 = tl.load(in_ptr9 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp64 = tl.load(in_ptr10 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
    tmp6 = tl.load(in_ptr0 + (256 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr0 + (7168 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr0 + (7424 + 512*x1 + 14336*x2 + (x0)), mask=tmp4 & xmask, eviction_policy='evict_first', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp13 = tmp11 * 0.25
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 288, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-256) + x0)), mask=tmp19 & xmask, eviction_policy='evict_first', other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 320, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-288) + x0)), mask=tmp24 & xmask, eviction_policy='evict_first', other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 352, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-320) + x0)), mask=tmp29 & xmask, eviction_policy='evict_first', other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 384, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tl.load(in_ptr4 + (32*x3 + ((-352) + x0)), mask=tmp34 & xmask, eviction_policy='evict_first', other=0.0)
    tmp36 = tmp0 >= tmp32
    tmp37 = tl.full([1], 416, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp36 & tmp38
    tmp40 = tl.load(in_ptr5 + (32*x3 + ((-384) + x0)), mask=tmp39 & xmask, eviction_policy='evict_first', other=0.0)
    tmp41 = tmp0 >= tmp37
    tmp42 = tl.full([1], 448, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr6 + (32*x3 + ((-416) + x0)), mask=tmp41 & xmask, eviction_policy='evict_first', other=0.0)
    tmp45 = tl.where(tmp39, tmp40, tmp44)
    tmp46 = tl.where(tmp34, tmp35, tmp45)
    tmp47 = tl.where(tmp29, tmp30, tmp46)
    tmp48 = tl.where(tmp24, tmp25, tmp47)
    tmp49 = tl.where(tmp19, tmp20, tmp48)
    tmp50 = tl.where(tmp4, tmp15, tmp49)
    tmp52 = tmp50 - tmp51
    tmp54 = tl.full([1], 1e-05, tl.float32)
    tmp55 = tmp53 + tmp54
    inv_std = libdevice.rsqrt(tmp55)
    tmp61 = tmp52 * inv_std
    tmp65 = libdevice.fma(tmp61, tmp62, tmp64)
    tmp66 = tl.full([1], 0, tl.int32)
    tmp67 = tl.maximum(tmp66, tmp65)
    tl.store(in_out_ptr0 + (x4), tmp67, mask=xmask)

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
    arg_11 = inputs[11]
    xnumel = 878080
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_50[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, 878080)
    return arg_11

def get_inputs():
    arg_0 = rand_strided((10, 448, 14, 14), (87808, 1, 6272, 448), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((7840, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((10, 32, 14, 14), (6272, 1, 448, 32), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((448,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((448,), (1,), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((448,), (1,), device='cuda:0', dtype=torch.float32)
    arg_11 = rand_strided((448,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11]

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
