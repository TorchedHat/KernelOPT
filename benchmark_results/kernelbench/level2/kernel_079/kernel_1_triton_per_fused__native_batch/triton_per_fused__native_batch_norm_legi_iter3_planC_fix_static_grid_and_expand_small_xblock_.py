import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # NEW: very small XBLOCK for maximum latency hiding
        triton.Config({'XBLOCK': 16}, num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 16}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=1, num_stages=1),
        # existing configs
        triton.Config({'XBLOCK': 32}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_clamp_convolution_max_mul_view_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1612800
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 12600)
    x1 = xindex // 12600
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 12600*r0_2 + 201600*x1), xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2), xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (r0_2), xmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_ptr3 + (r0_2 + 16*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tl.load(in_ptr4 + (r0_2 + 16*x1), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = tl.full([1, 1], 1.0 / 12600.0, tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tl.full([1, 1], 1e-05, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = libdevice.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp14 = tl.full([1, 1], -1.0, tl.float32)
    tmp15 = tl.maximum(tmp13, tmp14)
    tmp16 = tl.full([1, 1], 1.0, tl.float32)
    tmp17 = tl.minimum(tmp15, tmp16)
    tmp18 = tmp17 * tmp3
    tmp19 = tl.broadcast_to(tmp18, [XBLOCK, R0_BLOCK])
    tmp21 = tl.where(xmask, tmp19, float("-inf"))
    tmp22 = triton_helpers.max2(tmp21, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp22, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 1612800
    r0_numel = 16
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__native_batch_norm_legit_clamp_convolution_max_mul_view_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 1612800, 16)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((128, 16, 14, 30, 30), (201600, 12600, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1, 2048, 1, 1, 1), (2048, 1, 2048, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1, 2048, 1, 1, 1), (2048, 1, 2048, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128, 14, 30, 30), (12600, 900, 30, 1), device='cuda:0', dtype=torch.float32)
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
