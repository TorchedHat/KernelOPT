import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512, 'YBLOCK': 4}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 8}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 8}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_mul_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
    xnumel = 15876
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 64)
    y1 = yindex // 64
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 64*x2 + 1016064*y1), mask=xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), mask=ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (y0), mask=ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (y0), mask=ymask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (y0), mask=ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    tmp13 = tmp4 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1, 1], 2.0, tl.float32)
    tmp19 = tmp17 * tmp18
    tl.store(out_ptr0 + (x2 + 15876*y3), tmp19, mask=xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    ynumel = 8192
    xnumel = 15876
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_mul_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 8192, 15876)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((128, 64, 126, 126), (1016064, 1, 8064, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128, 64, 126, 126), (1016064, 15876, 126, 1), device='cuda:0', dtype=torch.float32)
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
