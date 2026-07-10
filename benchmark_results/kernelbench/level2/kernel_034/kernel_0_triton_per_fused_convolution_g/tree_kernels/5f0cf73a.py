import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=2,  num_stages=1),
        triton.Config({'XBLOCK': 1}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 1}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 1}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 1}, num_warps=2,  num_stages=2),
        triton.Config({'XBLOCK': 1}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 1}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 2}, num_warps=2,  num_stages=1),
        triton.Config({'XBLOCK': 2}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 2}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 2}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 4}, num_warps=2,  num_stages=1),
        triton.Config({'XBLOCK': 4}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 4}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 8}, num_warps=2,  num_stages=1),
        triton.Config({'XBLOCK': 8}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 8}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 8}, num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
    restore_value=['in_out_ptr0'],
)
@triton.jit
def triton_per_fused_convolution_gelu_mul_native_layer_norm_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
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
    r0_3 = r0_index
    x4 = xindex
    x1 = ((xindex // 2048) % 64)
    tmp0 = tl.load(in_out_ptr0 + (r0_3 + 64*x4), xmask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr1 + (r0_3), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr2 + (r0_3), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.sum(tmp5, 1)[:, None].to(tl.float32)
    tmp8 = tl.full([1, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = (tmp7 / tmp9)
    tmp11 = tmp3 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
    tmp15 = tl.sum(tmp13, 1)[:, None].to(tl.float32)
    tmp16 = tmp2 - tmp10
    tmp17 = tl.full([1, 1], 64.0, tl.float32)
    tmp18 = (tmp15 / tmp17)
    tmp19 = tl.full([1, 1], 1e-05, tl.float32)
    tmp20 = tmp18 + tmp19
    tmp21 = libdevice.rsqrt(tmp20)
    tmp22 = tmp16 * tmp21
    tmp24 = tmp22 * tmp23
    tmp26 = tmp24 + tmp25
    tmp27 = tl.full([1, 1], 0.5, tl.float32)
    tmp28 = tmp26 * tmp27
    tmp29 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp30 = tmp26 * tmp29
    tmp31 = libdevice.erf(tmp30)
    tmp32 = tl.full([1, 1], 1.0, tl.float32)
    tmp33 = tmp31 + tmp32
    tmp34 = tmp28 * tmp33
    tl.store(in_out_ptr0 + (r0_3 + 64*x4), tmp34, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 4194304
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_convolution_gelu_mul_native_layer_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 4194304, 64)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((32, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
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
