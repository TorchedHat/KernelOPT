import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=2, num_stages=1, maxnreg=32),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1, maxnreg=32),
        triton.Config({'XBLOCK': 64}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64}, num_warps=2, num_stages=1, maxnreg=32),
        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1, maxnreg=32),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1, maxnreg=32),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 512}, num_warps=16, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused_addmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_view_zeros_like_3(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 94080
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    _r0_hinted = tl.max_contiguous(tl.multiple_of(tl.arange(0, R0_BLOCK), R0_BLOCK), R0_BLOCK)
    r0_index = _r0_hinted[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 3)
    x1 = xindex // 3
    x5 = xindex
    x3 = ((xindex // 3) % 49)
    x4 = xindex // 147
    tmp19 = tl.load(in_ptr2 + (r0_2 + 32*x0 + 288*x1), mask=xmask, other=0.0)
    # Dead-code branch chain eliminated: r0_2 in [0,32), x0 in {0,1,2},
    # so r0_2+32*x0 max = 31+64 = 95 < 96, meaning tmp4 is always True.
    # Therefore tmp18 always equals in_ptr0[r0_2+32*x0].
    tmp18 = tl.load(in_ptr0 + (r0_2 + 32*x0), mask=xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 + tmp19
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, R0_BLOCK])
    tmp24 = tl.where(xmask, tmp22, 0)
    tmp25 = tl.sum(tmp24, 1)[:, None].to(tl.float32)
    tmp_eps2 = tl.full([1, 1], 1e-24, tl.float32)
    tmp_safe = tl.maximum(tmp25, tmp_eps2)
    tmp_inv_norm = libdevice.rsqrt(tmp_safe)
    tmp29 = tmp20 * tmp_inv_norm
    tl.store(out_ptr1 + (r0_2 + 32*x3 + 1568*x0 + 4704*x4), tmp29, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 94080
    r0_numel = 32
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_addmm_cat_clamp_min_clone_div_expand_linalg_vector_norm_permute_select_view_zeros_like_3[grid](arg_0, arg_1, arg_2, arg_3, 94080, 32)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((31360, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((640, 3, 49, 32), (4704, 1568, 32, 1), device='cuda:0', dtype=torch.float32)
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
