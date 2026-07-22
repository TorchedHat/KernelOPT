import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=3),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_max_pool2d_with_indices_75(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 407680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = ((xindex // 5824) % 7)
    x1 = ((xindex // 832) % 7)
    x0 = (xindex % 832)
    x5 = xindex // 5824
    x6 = xindex
    tmp0 = (-1) + 2*x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + 2*x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-12480) + x0 + 1664*x1 + 23296*x5), mask=tmp10 & xmask, other=float("-inf"))
    tmp12 = 2*x1
    tmp13 = tmp12 >= tmp1
    tmp14 = tmp12 < tmp3
    tmp15 = tmp13 & tmp14
    tmp16 = tmp5 & tmp15
    tmp17 = tl.load(in_ptr0 + ((-11648) + x0 + 1664*x1 + 23296*x5), mask=tmp16 & xmask, other=float("-inf"))
    tmp18 = tl.maximum(tmp11, tmp17)
    tmp19 = 1 + 2*x1
    tmp20 = tmp19 >= tmp1
    tmp21 = tmp19 < tmp3
    tmp22 = tmp20 & tmp21
    tmp23 = tmp5 & tmp22
    tmp24 = tl.load(in_ptr0 + ((-10816) + x0 + 1664*x1 + 23296*x5), mask=tmp23 & xmask, other=float("-inf"))
    tmp25 = tl.maximum(tmp18, tmp24)
    tmp26 = 2*x2
    tmp27 = tmp26 >= tmp1
    tmp28 = tmp26 < tmp3
    tmp29 = tmp27 & tmp28
    tmp30 = tmp29 & tmp9
    tmp31 = tl.load(in_ptr0 + ((-832) + x0 + 1664*x1 + 23296*x5), mask=tmp30 & xmask, other=float("-inf"))
    tmp32 = tl.maximum(tmp25, tmp31)
    tmp33 = tmp29 & tmp15
    tmp34 = tl.load(in_ptr0 + (x0 + 1664*x1 + 23296*x5), mask=tmp33 & xmask, other=float("-inf"))
    tmp35 = tl.maximum(tmp32, tmp34)
    tmp36 = tmp29 & tmp22
    tmp37 = tl.load(in_ptr0 + (832 + x0 + 1664*x1 + 23296*x5), mask=tmp36 & xmask, other=float("-inf"))
    tmp38 = tl.maximum(tmp35, tmp37)
    tmp39 = 1 + 2*x2
    tmp40 = tmp39 >= tmp1
    tmp41 = tmp39 < tmp3
    tmp42 = tmp40 & tmp41
    tmp43 = tmp42 & tmp9
    tmp44 = tl.load(in_ptr0 + (10816 + x0 + 1664*x1 + 23296*x5), mask=tmp43 & xmask, other=float("-inf"))
    tmp45 = tl.maximum(tmp38, tmp44)
    tmp46 = tmp42 & tmp15
    tmp47 = tl.load(in_ptr0 + (11648 + x0 + 1664*x1 + 23296*x5), mask=tmp46 & xmask, other=float("-inf"))
    tmp48 = tl.maximum(tmp45, tmp47)
    tmp49 = tmp42 & tmp22
    tmp50 = tl.load(in_ptr0 + (12480 + x0 + 1664*x1 + 23296*x5), mask=tmp49 & xmask, other=float("-inf"))
    tmp51 = tl.maximum(tmp48, tmp50)
    tl.store(out_ptr0 + (x6), tmp51, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 407680
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_max_pool2d_with_indices_75[grid](arg_0, arg_1, 407680)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 832, 14, 14), (163072, 1, 11648, 832), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 832, 7, 7), (40768, 1, 5824, 832), device='cuda:0', dtype=torch.float32)
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
