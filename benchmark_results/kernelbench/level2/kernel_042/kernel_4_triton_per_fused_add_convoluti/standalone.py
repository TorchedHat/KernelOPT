import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_per_fused_add_convolution_logsumexp_mean_mul_sum_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16
    r0_numel = 128
    R0_BLOCK: tl.constexpr = 128
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 128*x0), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp1 = tl.full([1, 1], 264196.0, tl.float32)
    tmp2 = (tmp0 / tmp1)
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, float("-inf"))
    tmp8 = triton_helpers.max2(tmp7, 1)[:, None].to(tl.float32)
    tmp9 = tl_math.abs(tmp8)
    tmp10 = tl.full([1, 1], float("inf"), tl.float32)
    tmp11 = tmp9 == tmp10
    tmp12 = tl.full([1, 1], 0.0, tl.float32)
    tmp13 = tl.where(tmp11, tmp12, tmp8)
    tmp14 = tmp4 - tmp13
    tmp15 = tl.exp(tmp14)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.where(xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp20 = tl.log(tmp19)
    tmp21 = tmp20 + tmp13
    tmp22 = tl.full([1, 1], 10.0, tl.float32)
    tmp23 = tmp21 * tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp23, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 16
    r0_numel = 128
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_convolution_logsumexp_mean_mul_sum_4[grid](arg_0, arg_1, arg_2, 16, 128, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((16, 1), (1, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 128, 1, 1), (128, 1, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
