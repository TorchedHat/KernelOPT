import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # num_stages=1 (baseline)
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=4,  num_stages=1),
        # num_stages=2: pipeline 1 iteration ahead, hides ~7-cycle L1TEX latency
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=8,  num_stages=2),
        # num_stages=3: pipeline 2 iterations ahead, maximally hides latency
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=16, num_stages=3),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512}, num_warps=8,  num_stages=3),
    ],
    key=['xnumel', 'r0_numel'],
    restore_value=['in_out_ptr0'],
)
@triton.jit
def triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 524288
    r0_numel = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 512)
    x3 = xindex

    # SINGLE PASS: accumulate max, sum, AND unnormalized exp weights
    _max = tl.full([XBLOCK, 1], float('-inf'), tl.float32)
    _sum = tl.zeros([XBLOCK, 1], tl.float32)
    _acc = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 1024*x0), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.full([1, 1], 0.0, tl.float32)
        tmp2 = tmp0 == tmp1
        tmp4 = tl.full([1, 1], 0.10206207261596577, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], float("-inf"), tl.float32)
        tmp7 = tl.where(tmp2, tmp6, tmp5)
        s = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])

        # Online softmax: update running max
        new_max = tl.maximum(_max, tl.max(s, axis=1, keep_dims=True))
        correction = tl.exp(_max - new_max)
        # HOISTED: compute exp_s once, reuse for both _acc and _sum
        exp_s = tl.exp(s - new_max)
        # FUSED: single FMA expression for _acc update
        _acc = _acc * correction + exp_s
        _sum = _sum * correction + tl.sum(exp_s, axis=1, keep_dims=True)
        _max = new_max

    # Normalize — since R0_BLOCK==r0_numel, _acc holds all weights in registers
    normalized = _acc / _sum

    # ELIMINATED store loop: direct store since R0_BLOCK==r0_numel (loop ran once anyway)
    r0_mask_store = r0_base < r0_numel
    tl.store(in_out_ptr0 + (r0_base + 512*x3), normalized, mask=xmask & r0_mask_store, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 524288
    r0_numel = 512
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6[grid](arg_0, arg_1, 524288, 512)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 8, 512, 512), (2097152, 262144, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 1, 1024, 1024), (1048576, 1048576, 1024, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

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
