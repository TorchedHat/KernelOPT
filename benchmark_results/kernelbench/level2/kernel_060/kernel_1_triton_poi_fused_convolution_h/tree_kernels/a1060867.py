import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_hardswish_mul_native_group_norm_sigmoid_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 251983872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x5 = xindex
    x2 = ((xindex // 123039) % 16)
    x7 = xindex // 123039
    x0 = (xindex % 3969)
    x4 = xindex // 3969
    tmp0 = tl.load(in_ptr0 + (x5), xmask)
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x7 // 4), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x7 // 4), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 1.0 / (1.0 + tl.exp(-tmp2))
    tmp4 = tmp3 * tmp2
    tmp6 = tmp4 - tmp5
    tmp8 = tl.full([1], 492156.0, tl.float32)
    tmp9 = (tmp7 / tmp8)
    tmp10 = tl.full([1], 1e-05, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tmp18 = tl.full([1], 3.0, tl.float32)
    tmp19 = tmp17 + tmp18
    tmp20 = tl.full([1], 0.0, tl.float32)
    tmp21 = tl.maximum(tmp19, tmp20)
    tmp22 = tl.full([1], 6.0, tl.float32)
    tmp23 = tl.minimum(tmp21, tmp22)
    tmp24 = tmp17 * tmp23
    tmp25 = tl.full([1], 0.16666666666666666, tl.float32)
    tmp26 = tmp24 * tmp25
    tl.store(out_ptr1 + (x5), tmp26, xmask, cache_modifier='.cs')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 251983872
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_hardswish_mul_native_group_norm_sigmoid_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 251983872, XBLOCK=128)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((128, 16, 31, 63, 63), (1968624, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 4, 1, 1), (4, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 4, 1, 1), (4, 1, 512, 512), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128, 16, 31, 63, 63), (1968624, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
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
