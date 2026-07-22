import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    tmp1 = tl.load(in_ptr0 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x5 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
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
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), tmp17, mask=xmask & r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    x3 = ((xindex // 256) % 16)
    x4 = xindex // 4096
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp21 = tl.load(in_out_ptr0 + (x0 + 256*r0_2 + 32768*x1), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = tl.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(out_ptr2 + (r0_2 + 128*x3 + 2048*x0 + 1048576*x4), tmp24, mask=xmask & r0_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 32768
    r0_numel = 128
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_prepare_softmax_online_sub_14[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 32768, 128)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((8, 256, 16, 128), (524288, 1, 32768, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((8, 256, 16, 128), (1048576, 2048, 128, 1), device='cuda:0', dtype=torch.float32)
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
