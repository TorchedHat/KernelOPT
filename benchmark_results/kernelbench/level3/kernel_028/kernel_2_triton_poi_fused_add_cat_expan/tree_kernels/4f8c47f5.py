import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 4096}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 4096}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 2048}, num_warps=8,  num_stages=3),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_add_cat_expand_view_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 201728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = ((xindex // 512) % 197)
    x0 = (xindex % 512)
    x2 = xindex // 100864
    x3 = (xindex % 100864)
    x4 = xindex
    tmp11 = tl.load(in_ptr2 + (x3), mask=xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x0), mask=tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tmp0 >= tmp3
    tmp7 = tl.full([1], 197, tl.int64)
    tmp8 = tmp0 < tmp7
    tmp9 = tl.load(in_ptr1 + (x0 + 512*((-1) + x1) + 100352*x2), mask=tmp6 & xmask, other=0.0)
    tmp10 = tl.where(tmp4, tmp5, tmp9)
    tmp12 = libdevice.fma(tmp10, 1.0, tmp11)
    tl.store(out_ptr0 + (x4), tmp12, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 201728
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_cat_expand_view_2[grid](arg_0, arg_1, arg_2, arg_3, 201728)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((1, 1, 512), (512, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((392, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 197, 512), (100864, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2, 197, 512), (100864, 512, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
