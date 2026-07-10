import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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
    r0_4 = r0_index
    x0 = (xindex % 64)
    x1 = ((xindex // 64) % 32)
    x2 = ((xindex // 2048) % 16)
    x3 = xindex // 32768
    x5 = (xindex % 16384)
    x6 = xindex
    base = x0 + 64*x1 + 4096*x2 + 131072*r0_4 + 2097152*x3
    tmp0  = tl.load(in_ptr0 + base,         mask=xmask, eviction_policy='evict_first')
    tmp1  = tl.load(in_ptr0 + base + 1,     mask=xmask, eviction_policy='evict_first')
    tmp3  = tl.load(in_ptr0 + base + 64,    mask=xmask, eviction_policy='evict_first')
    tmp5  = tl.load(in_ptr0 + base + 65,    mask=xmask, eviction_policy='evict_first')
    tmp7  = tl.load(in_ptr0 + base + 4096,  mask=xmask, eviction_policy='evict_first')
    tmp9  = tl.load(in_ptr0 + base + 4097,  mask=xmask, eviction_policy='evict_first')
    tmp11 = tl.load(in_ptr0 + base + 4160,  mask=xmask, eviction_policy='evict_first')
    tmp13 = tl.load(in_ptr0 + base + 4161,  mask=xmask, eviction_policy='evict_first')
    tmp28 = tl.load(in_ptr1 + (r0_4), mask=xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp10 = tl.maximum(tmp8, tmp9)
    tmp12 = tl.maximum(tmp10, tmp11)
    tmp14 = tl.maximum(tmp12, tmp13)
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp19 = triton_helpers.max2(tmp17, 1)[:, None].to(tl.float32)
    tmp20 = tmp17 - tmp19
    tmp21 = tl.exp(tmp20)
    tmp24 = tl.sum(tmp21, 1)[:, None].to(tl.float32)
    tmp27 = tmp21 / tmp24
    tmp29 = tmp27 - tmp28
    tmp30 = tl.sigmoid(tmp29)
    tmp31 = tmp30 * tmp29
    tmp32 = tl.broadcast_to(tmp31, [XBLOCK, R0_BLOCK])
    tmp34 = triton_helpers.max2(tmp32, 1)[:, None].to(tl.float32)
    tl.store(in_out_ptr0 + (x6), tmp34, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 2097152
    r0_numel = 16
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_convolution_exp_max_max_pool3d_with_indices_mul_prepare_softmax_online_sigmoid_sub_view_1[grid](arg_0, arg_1, arg_2, 2097152, 16)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 16, 32, 32), (16384, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 32, 64, 64), (2097152, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
