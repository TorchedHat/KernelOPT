import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 7)
    x1 = ((xindex // 7) % 7)
    x2 = ((xindex // 49) % 3)
    x3 = xindex // 147
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (15 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (16 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (225 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (226 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (240 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (241 + 2*x0 + 30*x1 + 450*x2 + 1600*x3), xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tl.store(out_ptr0 + (x4), tmp14, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 301056
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused__softmax_convolution_exp_max_pool3d_with_indices_sub_2[grid](arg_0, arg_1, 301056)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 16, 7, 15, 15), (25600, 1600, 225, 15, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 3, 7, 7), (2352, 147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
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
