import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_add_convolution_hardswish_logsumexp_native_group_norm_tanh_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2032128
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x1 = xindex // 15876
    x0 = (xindex % 15876)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*x3), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16*x1 + (r0_2 // 4)), xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (16*x1 + (r0_2 // 4)), xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr4 + (r0_2), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (r0_2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = tl.full([1, 1], 63504.0, tl.float32)
    tmp7 = (tmp5 / tmp6)
    tmp8 = tl.full([1, 1], 1e-05, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = libdevice.rsqrt(tmp9)
    tmp11 = tmp4 * tmp10
    tmp13 = tmp11 * tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = libdevice.tanh(tmp15)
    tmp17 = tl.full([1, 1], 3.0, tl.float32)
    tmp18 = tmp16 + tmp17
    tmp19 = tl.full([1, 1], 0.0, tl.float32)
    tmp20 = tl.maximum(tmp18, tmp19)
    tmp21 = tl.full([1, 1], 6.0, tl.float32)
    tmp22 = tl.minimum(tmp20, tmp21)
    tmp23 = tmp16 * tmp22
    tmp24 = tl.full([1, 1], 0.16666666666666666, tl.float32)
    tmp25 = tmp23 * tmp24
    tmp26 = tmp2 + tmp25
    tmp27 = tl.broadcast_to(tmp26, [XBLOCK, R0_BLOCK])
    tmp29 = tl.where(xmask, tmp27, float("-inf"))
    tmp30 = triton_helpers.max2(tmp29, 1)[:, None].to(tl.float32)
    tmp31 = tl_math.abs(tmp30)
    tmp32 = tl.full([1, 1], float("inf"), tl.float32)
    tmp33 = tmp31 == tmp32
    tmp34 = tl.where(tmp33, tmp19, tmp30)
    tmp35 = tmp26 - tmp34
    tmp36 = tl.exp(tmp35)
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    tmp39 = tl.where(xmask, tmp37, 0)
    tmp40 = tl.sum(tmp39, 1)[:, None].to(tl.float32)
    tmp41 = tl.log(tmp40)
    tmp42 = tmp41 + tmp34
    tl.store(out_ptr3 + (x3), tmp42, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 2032128
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_convolution_hardswish_logsumexp_native_group_norm_tanh_3[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 2032128, 64, XBLOCK=128)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((128, 64, 126, 126), (1016064, 1, 8064, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 16, 1, 1), (16, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 16, 1, 1), (16, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128, 1, 126, 126), (15876, 15876, 126, 1), device='cuda:0', dtype=torch.float32)
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
