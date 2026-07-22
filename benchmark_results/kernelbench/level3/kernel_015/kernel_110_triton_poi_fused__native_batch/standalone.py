import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_110(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, xnumel, XBLOCK : tl.constexpr):
    xnumel = 360640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 736)
    x1 = ((xindex // 736) % 7)
    x2 = xindex // 5152
    x3 = xindex // 736
    x4 = xindex
    tmp57 = tl.load(in_ptr8 + (x0), xmask, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr9 + (x0), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr10 + (x0), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr11 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (1024*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (512 + 1024*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr0 + (7168 + 1024*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr0 + (7680 + 1024*x1 + 14336*x2 + (x0)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([1], 4.0, tl.float32)
    tmp13 = (tmp11 / tmp12)
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 544, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + (32*x3 + ((-512) + x0)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp0 >= tmp17
    tmp22 = tl.full([1], 576, tl.int64)
    tmp23 = tmp0 < tmp22
    tmp24 = tmp21 & tmp23
    tmp25 = tl.load(in_ptr2 + (32*x3 + ((-544) + x0)), tmp24 & xmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tmp0 >= tmp22
    tmp27 = tl.full([1], 608, tl.int64)
    tmp28 = tmp0 < tmp27
    tmp29 = tmp26 & tmp28
    tmp30 = tl.load(in_ptr3 + (32*x3 + ((-576) + x0)), tmp29 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp0 >= tmp27
    tmp32 = tl.full([1], 640, tl.int64)
    tmp33 = tmp0 < tmp32
    tmp34 = tmp31 & tmp33
    tmp35 = tl.load(in_ptr4 + (32*x3 + ((-608) + x0)), tmp34 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp0 >= tmp32
    tmp37 = tl.full([1], 672, tl.int64)
    tmp38 = tmp0 < tmp37
    tmp39 = tmp36 & tmp38
    tmp40 = tl.load(in_ptr5 + (32*x3 + ((-640) + x0)), tmp39 & xmask, eviction_policy='evict_last', other=0.0)
    tmp41 = tmp0 >= tmp37
    tmp42 = tl.full([1], 704, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tmp41 & tmp43
    tmp45 = tl.load(in_ptr6 + (32*x3 + ((-672) + x0)), tmp44 & xmask, eviction_policy='evict_last', other=0.0)
    tmp46 = tmp0 >= tmp42
    tmp47 = tl.full([1], 736, tl.int64)
    tmp48 = tmp0 < tmp47
    tmp49 = tl.load(in_ptr7 + (32*x3 + ((-704) + x0)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp50 = tl.where(tmp44, tmp45, tmp49)
    tmp51 = tl.where(tmp39, tmp40, tmp50)
    tmp52 = tl.where(tmp34, tmp35, tmp51)
    tmp53 = tl.where(tmp29, tmp30, tmp52)
    tmp54 = tl.where(tmp24, tmp25, tmp53)
    tmp55 = tl.where(tmp19, tmp20, tmp54)
    tmp56 = tl.where(tmp4, tmp15, tmp55)
    tmp58 = tmp56 - tmp57
    tmp60 = tl.full([1], 1e-05, tl.float32)
    tmp61 = tmp59 + tmp60
    tmp62 = tl.sqrt_rn(tmp61)
    tmp63 = tl.full([1], 1, tl.int32)
    tmp64 = (tmp63 / tmp62)
    tmp65 = tl.full([1], 1.0, tl.float32)
    tmp66 = tmp64 * tmp65
    tmp67 = tmp58 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = tl.full([1], 0, tl.int32)
    tmp73 = tl.maximum(tmp72, tmp71)
    tl.store(in_out_ptr0 + (x4), tmp73, xmask)



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
    arg_12 = inputs[12]
    xnumel = 360640
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_110[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12, 360640, XBLOCK=128)
    return arg_12

def get_inputs():
    arg_0 = rand_strided((10, 736, 7, 7), (36064, 1, 5152, 736), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1960, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((10, 32, 7, 7), (1568, 1, 224, 32), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((736,), (1,), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((736,), (1,), device='cuda:0', dtype=torch.float32)
    arg_11 = rand_strided((736,), (1,), device='cuda:0', dtype=torch.float32)
    arg_12 = rand_strided((736,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, arg_11, arg_12]

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
