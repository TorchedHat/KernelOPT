import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 512},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_mul_native_group_norm_3(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 130056192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 64)
    x2 = xindex // 1016064
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (16*x2 + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (16*x2 + (x0 // 4)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1], 63504.0, tl.float32)
    tmp7 = (tmp5 / tmp6)
    tmp8 = tl.full([1], 1e-05, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 130056192
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_mul_native_group_norm_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 130056192)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((128, 64, 126, 126), (1016064, 1, 8064, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 16, 1, 1), (16, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 16, 1, 1), (16, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((64, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
