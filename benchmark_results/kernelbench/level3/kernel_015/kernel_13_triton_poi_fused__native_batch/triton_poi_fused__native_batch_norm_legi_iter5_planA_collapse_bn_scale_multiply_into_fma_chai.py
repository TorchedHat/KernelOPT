import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7024640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 224)
    x1 = xindex // 224
    x2 = xindex
    tmp35 = tl.load(in_ptr6 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp37 = tl.load(in_ptr7 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr8 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr9 + (x0), mask=xmask, eviction_policy='evict_last')
    # Simple boolean group masks — mutually exclusive, collectively exhaustive over x0 in [0,224)
    in_grp0 = x0 < 64
    in_grp1 = (x0 >= 64)  & (x0 < 96)
    in_grp2 = (x0 >= 96)  & (x0 < 128)
    in_grp3 = (x0 >= 128) & (x0 < 160)
    in_grp4 = (x0 >= 160) & (x0 < 192)
    in_grp5 = x0 >= 192
    # Load with evict_first so large streaming tensors don't evict BN params from L1
    v0 = tl.load(in_ptr0 + (64*x1 + x0),         mask=in_grp0 & xmask, other=0.0, eviction_policy='evict_first')
    v1 = tl.load(in_ptr1 + (32*x1 + (x0 - 64)),  mask=in_grp1 & xmask, other=0.0, eviction_policy='evict_first')
    v2 = tl.load(in_ptr2 + (32*x1 + (x0 - 96)),  mask=in_grp2 & xmask, other=0.0, eviction_policy='evict_first')
    v3 = tl.load(in_ptr3 + (32*x1 + (x0 - 128)), mask=in_grp3 & xmask, other=0.0, eviction_policy='evict_first')
    v4 = tl.load(in_ptr4 + (32*x1 + (x0 - 160)), mask=in_grp4 & xmask, other=0.0, eviction_policy='evict_first')
    v5 = tl.load(in_ptr5 + (32*x1 + (x0 - 192)), mask=in_grp5 & xmask, other=0.0, eviction_policy='evict_first')
    # Sum is equivalent to tl.where cascade since masks are mutually exclusive and other=0.0
    tmp34 = v0 + v1 + v2 + v3 + v4 + v5
    tmp38 = tl.full([1], 1e-05, tl.float32)
    tmp39 = tmp37 + tmp38
    tmp40 = libdevice.rsqrt(tmp39)
    tmp49 = tl.math.fma(tmp34, tmp40 * tmp46, tl.math.fma(-tmp35, tmp40 * tmp46, tmp48))
    tmp50 = tl.full([1], 0, tl.int32)
    tmp51 = tl.maximum(tmp50, tmp49)
    tl.store(in_out_ptr0 + (x2), tmp51, mask=xmask)

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
    xnumel = 7024640
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_13[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, arg_8, arg_9, arg_10, 7024640)
    return arg_10

def get_inputs():
    arg_0 = rand_strided((10, 224, 56, 56), (702464, 1, 12544, 224), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 64, 56, 56), (200704, 1, 3584, 64), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 32, 56, 56), (100352, 1, 1792, 32), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((224,), (1,), device='cuda:0', dtype=torch.float32)
    arg_8 = rand_strided((224,), (1,), device='cuda:0', dtype=torch.float32)
    arg_9 = rand_strided((224,), (1,), device='cuda:0', dtype=torch.float32)
    arg_10 = rand_strided((224,), (1,), device='cuda:0', dtype=torch.float32)
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
