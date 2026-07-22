import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_2(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4800
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex // 224
    x2 = (xindex % 224)
    x4 = xindex
    y5 = yindex
    y0 = (yindex % 480)
    y1 = yindex // 480
    tmp52 = tl.load(in_ptr0 + (x4 + 50176*y5), xmask & ymask, eviction_policy='evict_last')
    tmp0 = (-1) + x3
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 224, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x2
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-225) + x4 + 50176*y5), tmp10 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp12 = x2
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-224) + x4 + 50176*y5), tmp16 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp18 = tl.maximum(tmp11, tmp17)
    tmp19 = 1 + x2
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-223) + x4 + 50176*y5), tmp23 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp25 = tl.maximum(tmp18, tmp24)
    tmp26 = x3
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-1) + x4 + 50176*y5), tmp30 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp32 = tl.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x4 + 50176*y5), tmp33 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp35 = tl.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (1 + x4 + 50176*y5), tmp36 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp38 = tl.maximum(tmp35, tmp37)
    tmp39 = 1 + x3
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (223 + x4 + 50176*y5), tmp43 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp45 = tl.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (224 + x4 + 50176*y5), tmp46 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp48 = tl.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (225 + x4 + 50176*y5), tmp49 & xmask & ymask, eviction_policy='evict_last', other=float("-inf"))
    tmp51 = tl.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (y0 + 480*x4 + 24084480*y1), tmp51, xmask & ymask)
    tl.store(out_ptr1 + (y0 + 480*x4 + 24084480*y1), tmp52, xmask & ymask)
    tl.store(out_ptr2 + (y0 + 480*x4 + 24084480*y1), tmp52, xmask & ymask)
    tl.store(out_ptr3 + (y0 + 480*x4 + 24084480*y1), tmp52, xmask & ymask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    ynumel = 4800
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_max_pool2d_with_indices_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4800, 50176, YBLOCK=8, XBLOCK=128)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((10, 480, 224, 224), (24084480, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 480, 224, 224), (24084480, 1, 107520, 480), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 480, 224, 224), (24084480, 1, 107520, 480), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 480, 224, 224), (24084480, 1, 107520, 480), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 480, 224, 224), (24084480, 1, 107520, 480), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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
