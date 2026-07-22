import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 512}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 512}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512}, num_warps=8, num_stages=3),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__softmax_eq_exp_masked_fill_mul_prepare_softmax_online_slice_sub_view_6(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 524288
    r0_numel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    x0 = (xindex % 512)
    x3 = xindex

    # Single-pass: accumulate running max (_m) and normalizer (_l)
    _m = tl.full([XBLOCK, 1], float('-inf'), tl.float32)
    _l = tl.zeros([XBLOCK, 1], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 1024*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.full([1, 1], 0.0, tl.float32)
        tmp2 = tmp0 == tmp1
        tmp4 = tl.full([1, 1], 0.10206207261596577, tl.float32)
        tmp5 = tmp3 * tmp4
        tmp6 = tl.full([1, 1], float("-inf"), tl.float32)
        tmp7 = tl.where(tmp2, tmp6, tmp5)
        score = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        # Online softmax update
        tile_max = tl.max(score, axis=1, keep_dims=True)
        m_new = tl.maximum(_m, tile_max)
        alpha = tl.exp(_m - m_new)
        _l = _l * alpha + tl.sum(tl.exp(score - m_new), axis=1, keep_dims=True)
        _m = m_new

    # Second loop: write normalized output
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_2 = r0_index
        tmp11 = tl.load(in_ptr0 + (r0_2 + 1024*x0), xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_out_ptr0 + (r0_2 + 512*x3), xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        tmp12 = tl.full([1, 1], 0.0, tl.float32)
        tmp13 = tmp11 == tmp12
        tmp15 = tl.full([1, 1], 0.10206207261596577, tl.float32)
        tmp16 = tmp14 * tmp15
        tmp17 = tl.full([1, 1], float("-inf"), tl.float32)
        tmp18 = tl.where(tmp13, tmp17, tmp16)
        tmp19 = tmp18 - _m
        tmp20 = tl.exp(tmp19)
        tmp21 = (tmp20 / _l)
        tl.store(in_out_ptr0 + (r0_2 + 512*x3), tmp21, xmask & r0_mask)

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
