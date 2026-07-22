import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 32},  num_warps=4, num_stages=2, maxnreg=128),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 64},  num_warps=4, num_stages=2, maxnreg=128),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=4, num_stages=2, maxnreg=128),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 256}, num_warps=8, num_stages=2, maxnreg=128),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},  num_warps=4, num_stages=2, maxnreg=128),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2, maxnreg=128),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 256}, num_warps=8, num_stages=2, maxnreg=128),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2, maxnreg=128),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 256}, num_warps=4, num_stages=2, maxnreg=128),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 256}, num_warps=8, num_stages=2, maxnreg=128),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 256}, num_warps=4, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 256}, num_warps=4, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 256}, num_warps=4, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 256}, num_warps=8, num_stages=2, maxnreg=64),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 256
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base

    # NEW: channel is innermost (stride=1 in memory) -> COALESCED
    x_c = xindex % 128           # channel, stride=1
    x_h = (xindex // 128) % 32   # height, stride=32768
    x_b = xindex // (128 * 32)   # batch, stride=1048576

    # Hoist parameter loads outside the reduction loop (loop-invariant, indexed by x_c)
    tmp1  = tl.load(in_ptr0 + x_c, mask=xmask, eviction_policy='evict_last', other=0.0)
    tmp3  = tl.load(in_ptr1 + x_c, mask=xmask, eviction_policy='evict_last', other=0.0)
    tmp5  = tl.load(in_ptr2 + x_c, mask=xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + x_c, mask=xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tl.load(in_ptr4 + x_c, mask=xmask, eviction_policy='evict_last', other=0.0)

    # Precompute BN normalization factor (loop-invariant)
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt_rn(tmp7)
    tmp9 = tl.full([1, 1], 1, tl.int32)
    tmp10 = (tmp9 / tmp8)
    tmp11 = tl.full([1, 1], 1.0, tl.float32)
    tmp12 = tmp10 * tmp11

    _tmp19_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp19_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_w = r0_index  # width index, reduction dimension
        # NEW address: x_c + 128*r0_w + 32768*x_h + 1048576*x_b -> COALESCED
        tmp0 = tl.load(in_out_ptr0 + (x_c + 128*r0_w + 32768*x_h + 1048576*x_b), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])

        _tmp19_max_next, _tmp19_sum_next = triton_helpers.online_softmax_combine(
            _tmp19_max, _tmp19_sum, tmp18, False
        )

        _tmp19_max = tl.where(r0_mask, _tmp19_max_next, _tmp19_max)
        _tmp19_sum = tl.where(r0_mask, _tmp19_sum_next, _tmp19_sum)
        tl.store(in_out_ptr0 + (x_c + 128*r0_w + 32768*x_h + 1048576*x_b), tmp17, mask=xmask & r0_mask)

    tmp19, tmp20 = triton_helpers.online_softmax_reduce(
        _tmp19_max, _tmp19_sum, 1, False)
    tmp19 = tmp19[:, None]
    tmp20 = tmp20[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_w = r0_index  # width index
        tmp21 = tl.load(in_out_ptr0 + (x_c + 128*r0_w + 32768*x_h + 1048576*x_b), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp22 = tmp21 - tmp19
        tmp23 = tl.exp(tmp22)
        tmp24 = (tmp23 / tmp20)
        tl.store(in_out_ptr0 + (x_c + 128*r0_w + 32768*x_h + 1048576*x_b), tmp24, mask=xmask & r0_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 32768
    r0_numel = 256
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_7[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 256)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((8, 128, 32, 256), (1048576, 1, 32768, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5]

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
