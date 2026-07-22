import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # Existing maxnreg=64 configs (preserved as fallback)
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4,  num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 4096}, num_warps=8,  num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=4,  num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 1024}, num_warps=8,  num_stages=3, maxnreg=64),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=16, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=16, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=16, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=16, num_stages=3, maxnreg=64),
        # NEW: maxnreg=80 (spill-reduction, ~37.5% occupancy)
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4,  num_stages=2, maxnreg=80),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 4096}, num_warps=8,  num_stages=2, maxnreg=80),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=2, maxnreg=80),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=2, maxnreg=80),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=2, maxnreg=80),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=3, maxnreg=80),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=4,  num_stages=2, maxnreg=80),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 1024}, num_warps=8,  num_stages=3, maxnreg=80),
        # NEW: maxnreg=96 (spill-free, ~31.25% occupancy)
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4,  num_stages=2, maxnreg=96),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 4096}, num_warps=8,  num_stages=2, maxnreg=96),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=2, maxnreg=96),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=2, maxnreg=96),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=2, maxnreg=96),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=3, maxnreg=96),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=4,  num_stages=2, maxnreg=96),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 1024}, num_warps=8,  num_stages=3, maxnreg=96),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 16384
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_index = tl.max_contiguous(tl.multiple_of(r0_index, [1, R0_BLOCK]), [1, R0_BLOCK])
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp5 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 * tmp2
        tmp4 = tmp0 - tmp3
        tmp6 = tl.sqrt_rn(tmp5)
        tmp7 = tl.full([1, 1], 1e-12, tl.float32)
        tmp8 = tl.maximum(tmp6, tmp7)
        tmp9 = (tmp4 / tmp8)
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(r0_mask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_index = tl.max_contiguous(tl.multiple_of(r0_index, [1, R0_BLOCK]), [1, R0_BLOCK])
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp14 = tl.load(in_ptr0 + (512*((r0_1 % 32)) + 16384*x0 + (r0_1 // 32)), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp15 = tl.load(in_ptr1 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp16 = tl.load(in_ptr2 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp19 = tl.load(in_ptr3 + (32*x0 + ((r0_1 % 32))), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp17 = tmp15 * tmp16
        tmp18 = tmp14 - tmp17
        tmp20 = tl.sqrt_rn(tmp19)
        tmp21 = tl.full([1, 1], 1e-12, tl.float32)
        tmp22 = tl.maximum(tmp20, tmp21)
        tmp23 = (tmp18 / tmp22)
        tmp24 = tl.sqrt_rn(tmp12)
        tmp25 = tl.maximum(tmp24, tmp21)
        tmp26 = (tmp23 / tmp25)
        tl.store(out_ptr1 + (r0_1 + 16384*x0), tmp26, mask=r0_mask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 2048
    r0_numel = 16384
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2048, 16384)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((2048, 32, 512), (16384, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 1, 32), (32, 65536, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 512, 32), (16384, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048, 1, 32), (32, 65536, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
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
