import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 2048}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=2,  num_stages=3),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512},  num_warps=2,  num_stages=3),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=2,  num_stages=3),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 256},  num_warps=2,  num_stages=4),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=2,  num_stages=3),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 256},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 128},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__softmax_avg_pool3d_clamp_convolution_exp_mul_prepare_softmax_online_sub_view_1(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 131072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x5 = xindex
    x0 = (xindex % 64)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    _tmp8_max = tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32)
    _tmp8_sum = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 131072*x5), xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 0.0, tl.float32)
        tmp4 = tl.maximum(tmp2, tmp3)
        tmp5 = tl.full([1, 1], 1.0, tl.float32)
        tmp6 = tl.minimum(tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])

        _tmp8_max_next, _tmp8_sum_next = triton_helpers.online_softmax_combine(
            _tmp8_max, _tmp8_sum, tmp7, False
        )

        _tmp8_max = tl.where(xmask, _tmp8_max_next, _tmp8_max)
        _tmp8_sum = tl.where(xmask, _tmp8_sum_next, _tmp8_sum)

    tmp8, tmp9 = triton_helpers.online_softmax_reduce(
        _tmp8_max, _tmp8_sum, 1, False)
    tmp8 = tmp8[:, None]
    tmp9 = tmp9[:, None]
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp10 = tl.load(in_ptr0 + (r0_2 + 131072*x5), xmask, eviction_policy='evict_first', other=0.0)
        tmp11 = tmp10 + tmp1
        tmp12 = tl.full([1, 1], 0.0, tl.float32)
        tmp13 = tl.maximum(tmp11, tmp12)
        tmp14 = tl.full([1, 1], 1.0, tl.float32)
        tmp15 = tl.minimum(tmp13, tmp14)
        tmp16 = tmp15 - tmp8
        tmp17 = tl.exp(tmp16)
        tmp18 = (tmp17 / tmp9)
        tmp20 = tmp18 * tmp19
        tl.store(out_ptr2 + (r0_2 + 131072*x5), tmp20, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 2048
    r0_numel = 131072
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused__softmax_avg_pool3d_clamp_convolution_exp_mul_prepare_softmax_online_sub_view_1[grid](arg_0, arg_1, arg_2, arg_3, 2048, 131072)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((32, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
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
