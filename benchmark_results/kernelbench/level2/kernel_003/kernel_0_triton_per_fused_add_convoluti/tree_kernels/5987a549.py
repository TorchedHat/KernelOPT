import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK':  64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK':  64}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK':  64}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK':  64}, num_warps=2, num_stages=4),
        triton.Config({'XBLOCK':  64}, num_warps=1, num_stages=4),
        triton.Config({'XBLOCK':  64}, num_warps=1, num_stages=3),
        triton.Config({'XBLOCK':  32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK':  32}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK':  32}, num_warps=2, num_stages=4),
        triton.Config({'XBLOCK':  32}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK':  32}, num_warps=1, num_stages=4),
        triton.Config({'XBLOCK':  32}, num_warps=1, num_stages=3),
        triton.Config({'XBLOCK':  32}, num_warps=1, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_add_convolution_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 2048) % 64)
    tmp0 = tl.load(in_out_ptr0 + (r0_3 + 64*x4), mask=xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + (x1), mask=xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + tl.zeros([XBLOCK, 1], tl.int32), mask=xmask)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, 1])
    tmp26 = tl.load(in_ptr2 + (r0_3), mask=r0_mask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr3 + (r0_3), mask=r0_mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp2 + tmp4
    tmp6 = tl.broadcast_to(tmp5, [XBLOCK, R0_BLOCK])
    # tmp8 removed — use tmp6 directly (was redundant broadcast of tmp6 to same shape)
    tmp10 = tl.sum(tmp6, 1)[:, None].to(tl.float32)
    # merged tmp11 (int32 64) + tmp12 (cast to float) into single float constant
    tmp12 = tl.full([1, 1], 64.0, tl.float32)
    tmp13 = (tmp10 / tmp12)
    tmp14 = tmp6 - tmp13
    tmp15 = tmp14 * tmp14
    # tmp16 removed — use tmp15 directly (was redundant broadcast of tmp15 to same shape)
    tmp18 = tl.sum(tmp15, 1)[:, None].to(tl.float32)
    # tmp19 removed — reuse tmp14 (algebraically identical: tmp6==broadcast(tmp5), so tmp5-tmp13==tmp6-tmp13==tmp14)
    # tmp20 removed — reuse tmp12 (both are 64.0 float32)
    tmp21 = (tmp18 / tmp12)
    tmp22 = tl.full([1, 1], 1e-05, tl.float32)
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    tmp25 = tmp14 * tmp24  # was tmp19 * tmp24; tmp14 == tmp19
    tmp27 = tmp25 * tmp26
    tmp29 = tmp27 + tmp28
    tl.store(in_out_ptr0 + (r0_3 + 64*x4), tmp29, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 4194304
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_convolution_native_layer_norm_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 4194304, 64)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((32, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
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
