import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 128}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_convolution_div_max_pool3d_with_indices_mean_1(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 108544
    r0_numel = 127
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 53)
    x1 = xindex // 53
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 127*x0
        tmp1 = tl.full([1, 1], 6727, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), mask=r0_mask & tmp2 & xmask, other=0.0, eviction_policy='evict_last')
        tmp4 = tl.load(in_ptr0 + (1 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), mask=r0_mask & tmp2 & xmask, other=0.0, eviction_policy='evict_last')
        tmp5 = tl.maximum(tmp3, tmp4)
        tmp6 = tl.load(in_ptr0 + (62 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), mask=r0_mask & tmp2 & xmask, other=0.0, eviction_policy='evict_last')
        tmp7 = tl.maximum(tmp5, tmp6)
        tmp8 = tl.load(in_ptr0 + (63 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), mask=r0_mask & tmp2 & xmask, other=0.0, eviction_policy='evict_last')
        tmp9 = tl.maximum(tmp7, tmp8)
        tmp10 = tl.load(in_ptr0 + (3872 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), mask=r0_mask & tmp2 & xmask, other=0.0, eviction_policy='evict_last')
        tmp11 = tl.maximum(tmp9, tmp10)
        tmp12 = tl.load(in_ptr0 + (3873 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), mask=r0_mask & tmp2 & xmask, other=0.0, eviction_policy='evict_last')
        tmp13 = tl.maximum(tmp11, tmp12)
        tmp14 = tl.load(in_ptr0 + (3934 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), mask=r0_mask & tmp2 & xmask, other=0.0, eviction_policy='evict_last')
        tmp15 = tl.maximum(tmp13, tmp14)
        tmp16 = tl.load(in_ptr0 + (3935 + 2*(((r0_2 + 127*x0) % 31)) + 124*((((r0_2 + 127*x0) // 31) % 31)) + 7744*((((r0_2 + 127*x0) // 961) % 7)) + 54208*x1), mask=r0_mask & tmp2 & xmask, other=0.0, eviction_policy='evict_last')
        tmp17 = tl.maximum(tmp15, tmp16)
        tmp18 = tl.full(tmp17.shape, 0, tmp17.dtype)
        tmp19 = tl.where(tmp2, tmp17, tmp18)
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 108544
    r0_numel = 127
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused_convolution_div_max_pool3d_with_indices_mean_1[grid](arg_0, arg_1, 108544, 127)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 16, 14, 62, 62), (867328, 54208, 3872, 62, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 1, 1, 1, 53), (848, 53, 108544, 108544, 108544, 1), device='cuda:0', dtype=torch.float32)
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
