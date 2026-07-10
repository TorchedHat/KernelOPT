import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_add_addmm_gelu_logsumexp_mean_sub_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr2 + (r0_1), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(r0_mask & xmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp27 = tl.load(in_ptr3 + (r0_1 + 8192*x0), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp8 = tl.full([1, 1], 8192.0, tl.float32)
        tmp9 = (tmp6 / tmp8)
        tmp10 = tl_math.abs(tmp9)
        tmp11 = tl.full([1, 1], float("inf"), tl.float32)
        tmp12 = tmp10 == tmp11
        tmp13 = tl.full([1, 1], 0.0, tl.float32)
        tmp14 = tl.where(tmp12, tmp13, tmp9)
        tmp15 = tmp9 - tmp14
        tmp16 = tl.exp(tmp15)
        tmp17 = tl.log(tmp16)
        tmp18 = tmp17 + tmp14
        tmp19 = tl.full([1, 1], 0.5, tl.float32)
        tmp20 = tmp18 * tmp19
        tmp21 = tl.full([1, 1], 0.7071067811865476, tl.float32)
        tmp22 = tmp18 * tmp21
        tmp23 = libdevice.erf(tmp22)
        tmp24 = tl.full([1, 1], 1.0, tl.float32)
        tmp25 = tmp23 + tmp24
        tmp26 = tmp20 * tmp25
        tmp28 = tmp26 + tmp27
        tl.store(out_ptr1 + (r0_1 + 8192*x0), tmp28, r0_mask & xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 2048
    r0_numel = 8192
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused_add_addmm_gelu_logsumexp_mean_sub_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2048, 8192, XBLOCK=128, R0_BLOCK=512)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((2048, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4]

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
