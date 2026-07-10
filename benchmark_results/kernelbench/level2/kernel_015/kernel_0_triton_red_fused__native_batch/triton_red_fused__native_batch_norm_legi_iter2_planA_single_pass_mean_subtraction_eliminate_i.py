import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_sub_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 512
    r0_numel = 123039
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 32)
    tmp1 = tl.load(in_ptr1 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
    # Precompute inv_std to avoid recomputing in both passes
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11
    # Pass 1: accumulate sum only (no store to out_ptr0)
    _tmp19 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_5 + 123039*x4), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
        tmp20 = _tmp19 + tmp18
        _tmp19 = tl.where(r0_mask & xmask, tmp20, _tmp19)
    tmp19 = tl.sum(_tmp19, 1)[:, None]
    # Compute mean over all r0_numel elements
    mean = tmp19 / 123039.0
    # Pass 2: recompute tmp17 from in_ptr0, subtract mean, store contiguously to out_ptr2
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_5 + 123039*x4), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp24 = tmp17 - mean
        tl.store(out_ptr2 + (r0_5 + 123039*x4), tmp24, mask=r0_mask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    arg_7 = inputs[7]
    xnumel = 512
    r0_numel = 123039
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_sub_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7, 512, 123039, XBLOCK=1, R0_BLOCK=2048)
    return arg_7

def get_inputs():
    arg_0 = rand_strided((16, 32, 31, 63, 63), (3937248, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((16, 32, 31, 63, 63), (3968000, 124000, 4000, 63, 1), device='cuda:0', dtype=torch.float32)
    arg_7 = rand_strided((16, 32, 31, 63, 63), (3937248, 123039, 3969, 63, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, arg_7]

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
