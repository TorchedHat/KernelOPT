import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_add_addmm_native_layer_norm_view_10(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 768
    R0_BLOCK: tl.constexpr = 1024
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
    # 768 % 16 == 0, so multiple_of(16) is valid for the r0 dimension
    # For 2D tensor [1, R0_BLOCK], values must match shape: [1, 16] and [1, R0_BLOCK]
    r0_1 = tl.max_contiguous(tl.multiple_of(r0_index, [1, 16]), [1, R0_BLOCK])
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 768*x0), mask=xmask & r0_mask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_1), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tl.load(in_ptr2 + (r0_1 + 768*x0), mask=xmask & r0_mask, other=0.0)
    tmp28 = tl.load(in_ptr3 + (r0_1), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r0_1), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(r0_mask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    tmp10 = tl.where(r0_mask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 768, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.where(r0_mask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None].to(tl.float32)
    tmp21 = tmp4 - tmp14
    tmp22 = tl.full([1, 1], 768.0, tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tl.full([1, 1], 1e-05, tl.float32)
    tmp25 = tmp23 + tmp24
    tmp26 = libdevice.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tl.store(out_ptr2 + (r0_1 + 768*x0), tmp31, mask=xmask & r0_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 65536
    r0_numel = 768
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_addmm_native_layer_norm_view_10[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 65536, 768)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((65536, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128, 512, 768), (393216, 768, 1), device='cuda:0', dtype=torch.float32)
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
