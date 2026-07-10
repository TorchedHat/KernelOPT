import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 256},  num_warps=8),
        triton.Config({'XBLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 512},  num_warps=16),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=16),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_add_avg_pool3d_convolution_mul_1(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 29521920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 31)
    x1 = ((xindex // 31) % 31)
    x2 = ((xindex // 961) % 15)
    x6 = xindex // 14415
    x3 = ((xindex // 14415) % 16)
    x5 = (xindex % 14415)
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (63 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (64 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (4000 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (4001 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (4063 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (4064 + 2*x0 + 126*x1 + 8000*x2 + 124000*x6), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr1 + (x3), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (0), mask=xmask, other=0.0)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tl.full([1], 8.0, tl.float32)
    tmp16 = (tmp14 / tmp15)
    tmp18 = tmp16 + tmp17
    tmp21 = tmp18 * tmp20
    tl.store(out_ptr1 + (x7), tmp21, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 29521920
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_avg_pool3d_convolution_mul_1[grid](arg_0, arg_1, arg_2, arg_3, 29521920)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 16, 31, 63, 63), (1984000, 124000, 4000, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 16, 15, 31, 31), (230640, 14415, 961, 31, 1), device='cuda:0', dtype=torch.float32)
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
