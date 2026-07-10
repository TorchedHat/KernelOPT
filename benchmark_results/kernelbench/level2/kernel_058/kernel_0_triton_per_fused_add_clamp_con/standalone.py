import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_add_clamp_convolution_div_logsumexp_mul_sigmoid_sub_0(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 15748992
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    r0_3 = r0_index
    x2 = xindex // 123039
    x5 = (xindex % 123039)
    x0 = (xindex % 3969)
    x4 = xindex // 3969
    x6 = xindex
    tmp0 = tl.load(in_ptr0 + (x5 + 123039*r0_3 + 1968624*x2), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_3), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr2 + (0))
    tmp27 = tl.broadcast_to(tmp26, [1, 1])
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tl_math.abs(tmp6)
    tmp8 = tl.full([1, 1], float("inf"), tl.float32)
    tmp9 = tmp7 == tmp8
    tmp10 = tl.full([1, 1], 0.0, tl.float32)
    tmp11 = tl.where(tmp9, tmp10, tmp6)
    tmp12 = tmp2 - tmp11
    tmp13 = tl.exp(tmp12)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp16 = tl.where(xmask, tmp14, 0)
    tmp17 = tl.sum(tmp16, 1)[:, None].to(tl.float32)
    tmp18 = tl.log(tmp17)
    tmp19 = tmp18 + tmp11
    tmp20 = tl.full([1, 1], 3.0, tl.float32)
    tmp21 = tmp19 + tmp20
    tmp22 = tl.sigmoid(tmp21)
    tmp23 = tmp19 * tmp22
    tmp24 = tl.full([1, 1], 0.16666666666666666, tl.float32)
    tmp25 = tmp23 * tmp24
    tmp28 = tmp25 - tmp27
    tmp29 = tl.full([1, 1], -1.0, tl.float32)
    tmp30 = tl.maximum(tmp28, tmp29)
    tmp31 = tl.full([1, 1], 1.0, tl.float32)
    tmp32 = tl.minimum(tmp30, tmp31)
    tl.store(out_ptr2 + (x6), tmp32, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 15748992
    r0_numel = 16
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_clamp_convolution_div_logsumexp_mul_sigmoid_sub_0[grid](arg_0, arg_1, arg_2, arg_3, 15748992, 16, XBLOCK=128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 16, 31, 63, 63), (1968624, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 1, 31, 63, 63), (123039, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
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
