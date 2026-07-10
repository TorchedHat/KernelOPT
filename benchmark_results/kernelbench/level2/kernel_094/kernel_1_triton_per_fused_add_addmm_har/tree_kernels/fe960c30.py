import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=3),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused_add_addmm_hardtanh_mish_native_group_norm_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    GROUP_SIZE: tl.constexpr = 8
    pid = tl.program_id(0)
    num_pid = tl.cdiv(xnumel, XBLOCK)
    group_id = pid // GROUP_SIZE
    group_size_adj = tl.minimum(num_pid - group_id * GROUP_SIZE, GROUP_SIZE)
    local_id = pid % GROUP_SIZE
    swizzled_pid = group_id * GROUP_SIZE + (local_id % group_size_adj)
    xoffset = swizzled_pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 32*x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (r0_2 + 32*x3), xmask)
    tmp3 = tl.load(in_ptr2 + (r0_2 + 32*x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1, 1], -1.0, tl.float32)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp7 = tl.full([1, 1], 1.0, tl.float32)
    tmp8 = tl.minimum(tmp6, tmp7)
    tmp9 = tl.full([1, 1], 20.0, tl.float32)
    tmp10 = tmp8 > tmp9
    tmp11 = tl.exp(tmp8)
    tmp12 = libdevice.log1p(tmp11)
    tmp13 = tl.where(tmp10, tmp8, tmp12)
    tmp14 = libdevice.tanh(tmp13)
    tmp15 = tmp8 * tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, R0_BLOCK])
    tmp18 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp20 = tl.sum(tmp18, 1)[:, None].to(tl.float32)
    tmp21 = tl.full([1, 1], 32, tl.int32)
    tmp22 = tmp21.to(tl.float32)
    tmp23 = (tmp20 / tmp22)
    tmp24 = tmp16 - tmp23
    tmp25 = tmp24 * tmp24
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, R0_BLOCK])
    tmp28 = tl.sum(tmp26, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp23, xmask)
    tl.store(out_ptr1 + (x3), tmp28, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 262144
    r0_numel = 32
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_addmm_hardtanh_mish_native_group_norm_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 262144, 32)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1024, 256, 1, 1), (256, 1, 262144, 262144), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1024, 256, 1, 1), (256, 1, 262144, 262144), device='cuda:0', dtype=torch.float32)
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
