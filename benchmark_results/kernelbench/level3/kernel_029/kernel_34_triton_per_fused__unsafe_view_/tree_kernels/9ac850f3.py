import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=1, maxnreg=128),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1, maxnreg=128),
        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1, maxnreg=128),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1, maxnreg=128),
        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=1, maxnreg=64),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1, maxnreg=64),
        triton.Config({'XBLOCK': 64}, num_warps=8, num_stages=1, maxnreg=64),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1, maxnreg=64),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1960
    r0_numel = 384
    R0_BLOCK: tl.constexpr = 512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 196)
    x1 = xindex // 196
    tmp0 = tl.load(in_ptr0 + (r0_2 + 384*x3), mask=r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (32*((((x0 % 14)) % 7)) + 224*(((x0 // 14) % 7)) + 1568*(r0_2 // 32) + 18816*(((x0 % 14)) // 7) + 37632*(x0 // 98) + 75264*x1 + ((r0_2 % 32))), mask=r0_mask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (7*(((x0 // 14) % 7)) + 49*(r0_2 // 32) + ((((x0 % 14)) % 7))), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 384, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 384.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_2 + 384*x3), tmp31, mask=r0_mask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 1960
    r0_numel = 384
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__unsafe_view_add_clone_convolution_native_layer_norm_permute_transpose_view_34[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1960, 384)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((1960, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((40, 588, 32), (18816, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((588,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((384,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 196, 384), (75264, 384, 1), device='cuda:0', dtype=torch.float32)
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
