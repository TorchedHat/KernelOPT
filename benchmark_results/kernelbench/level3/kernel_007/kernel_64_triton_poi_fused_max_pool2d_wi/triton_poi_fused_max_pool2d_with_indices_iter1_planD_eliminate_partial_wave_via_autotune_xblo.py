import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=8),
        triton.Config({'XBLOCK': 512}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=16),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_64(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1034880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 7392) % 14)
    x1 = ((xindex // 528) % 14)
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-7920) + x6), mask=tmp10 & xmask, other=float("-inf"))
    tmp12 = x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-7392) + x6), mask=tmp16 & xmask, other=float("-inf"))
    tmp18 = tl.maximum(tmp11, tmp17)
    tmp19 = 1 + x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-6864) + x6), mask=tmp23 & xmask, other=float("-inf"))
    tmp25 = tl.maximum(tmp18, tmp24)
    tmp26 = x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-528) + x6), mask=tmp30 & xmask, other=float("-inf"))
    tmp32 = tl.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x6), mask=tmp33 & xmask, other=float("-inf"))
    tmp35 = tl.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (528 + x6), mask=tmp36 & xmask, other=float("-inf"))
    tmp38 = tl.maximum(tmp35, tmp37)
    tmp39 = 1 + x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (6864 + x6), mask=tmp43 & xmask, other=float("-inf"))
    tmp45 = tl.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (7392 + x6), mask=tmp46 & xmask, other=float("-inf"))
    tmp48 = tl.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (7920 + x6), mask=tmp49 & xmask, other=float("-inf"))
    tmp51 = tl.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x6), tmp51, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 1034880
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_max_pool2d_with_indices_64[grid](arg_0, arg_1, 1034880)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 528, 14, 14), (103488, 1, 7392, 528), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 528, 14, 14), (103488, 1, 7392, 528), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

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
