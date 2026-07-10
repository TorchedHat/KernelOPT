import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 128},  num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_max_pool3d_with_indices_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 30505984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 31)
    x1 = ((xindex // 31) % 31)
    x2 = ((xindex // 961) % 31)
    x3 = xindex // 29791
    x4 = (xindex % 29791)
    base = 2*x0 + 126*x1 + 8000*x2 + 252000*x3
    tmp0 = tl.load(in_ptr0 + (base), xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.load(in_ptr0 + (1 + base), xmask, eviction_policy='evict_first', other=0.0)
    tmp3 = tl.load(in_ptr0 + (63 + base), xmask, eviction_policy='evict_first', other=0.0)
    tmp5 = tl.load(in_ptr0 + (64 + base), xmask, eviction_policy='evict_first', other=0.0)
    tmp7 = tl.load(in_ptr0 + (4000 + base), xmask, eviction_policy='evict_first', other=0.0)
    tmp9 = tl.load(in_ptr0 + (4001 + base), xmask, eviction_policy='evict_first', other=0.0)
    tmp11 = tl.load(in_ptr0 + (4063 + base), xmask, eviction_policy='evict_first', other=0.0)
    tmp13 = tl.load(in_ptr0 + (4064 + base), xmask, eviction_policy='evict_first', other=0.0)
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tl.store(out_ptr0 + (x4 + 29792*x3), tmp14, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 30505984
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_max_pool3d_with_indices_1[grid](arg_0, arg_1, 30505984)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 64, 63, 63, 63), (16128000, 252000, 4000, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 64, 31, 31, 31), (1906688, 29792, 961, 31, 1), device='cuda:0', dtype=torch.float32)
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
