import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2130048
    r0_numel = 128
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x3 = xindex
    x0 = (xindex % 16641)
    x1 = xindex // 16641
    _tmp4_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp4_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    # SINGLE PASS: load all inputs once, accumulate softmax state, then compute output inline
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr2 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        _tmp4_max_next, _tmp4_sum_next = triton_helpers.online_softmax_combine(
            _tmp4_max, _tmp4_sum, tmp3, False
        )
        _tmp4_max = tl.where(r0_mask & xmask, _tmp4_max_next, _tmp4_max)
        _tmp4_sum = tl.where(r0_mask & xmask, _tmp4_sum_next, _tmp4_sum)
        # Compute output inline using still-live register values
        # (safe because R0_BLOCK=128 == r0_numel=128, loop runs exactly once)
        tmp4_g, tmp5_g = triton_helpers.online_softmax_reduce(_tmp4_max, _tmp4_sum, 1, False)
        tmp4_g = tmp4_g[:, None]
        tmp5_g = tmp5_g[:, None]
        tmp9 = tmp2 - tmp4_g
        tmp10 = tl.exp(tmp9)
        tmp11 = tmp10 / tmp5_g
        tmp13 = tmp11 + tmp12
        tmp15 = tmp13 * tl.full([1, 1], 2.0, tl.float32)
        tmp16 = tl.sigmoid(tmp15)
        tl.store(out_ptr2 + (x0 + 16641*r0_2 + 2130048*x1), tmp16, mask=r0_mask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 2130048
    r0_numel = 128
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2[grid](arg_0, arg_1, arg_2, arg_3, 2130048, 128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 128, 129, 129), (2130048, 1, 16512, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 128, 129, 129), (2130048, 16641, 129, 1), device='cuda:0', dtype=torch.float32)
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
