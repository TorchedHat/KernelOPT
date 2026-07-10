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
def triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3225600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 15)
    x1 = ((xindex // 15) % 15)
    x2 = ((xindex // 225) % 7)
    x3 = xindex // 1575
    x4 = (xindex % 1575)
    base = 2*x0 + 60*x1 + 1800*x2 + 12608*x3
    tmp0  = tl.load(in_ptr0 + base,        mask=xmask, eviction_policy='evict_first')
    tmp1  = tl.load(in_ptr0 + base + 1,    mask=xmask, eviction_policy='evict_first')
    tmp3  = tl.load(in_ptr0 + base + 30,   mask=xmask, eviction_policy='evict_first')
    tmp5  = tl.load(in_ptr0 + base + 31,   mask=xmask, eviction_policy='evict_first')
    tmp7  = tl.load(in_ptr0 + base + 900,  mask=xmask, eviction_policy='evict_first')
    tmp9  = tl.load(in_ptr0 + base + 901,  mask=xmask, eviction_policy='evict_first')
    tmp11 = tl.load(in_ptr0 + base + 930,  mask=xmask, eviction_policy='evict_first')
    tmp13 = tl.load(in_ptr0 + base + 931,  mask=xmask, eviction_policy='evict_first')
    tmp2  = tl.maximum(tmp0,  tmp1)
    tmp4  = tl.maximum(tmp2,  tmp3)
    tmp6  = tl.maximum(tmp4,  tmp5)
    tmp8  = tl.maximum(tmp6,  tmp7)
    tmp10 = tl.maximum(tmp8,  tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tl.store(out_ptr0 + (x4 + 1600*x3), tmp14, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 3225600
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_1[grid](arg_0, arg_1, 3225600)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 16, 14, 30, 30), (201728, 12608, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 7, 15, 15), (25600, 1600, 225, 15, 1), device='cuda:0', dtype=torch.float32)
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
