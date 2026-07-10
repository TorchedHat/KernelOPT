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
        triton.Config({'XBLOCK': 64}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_add_addmm_mul_native_group_norm_sigmoid_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2097152
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
    r0_2 = r0_index
    x0 = (xindex % 64)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2 + 64*x3), xmask, eviction_policy='evict_first', other=0.0)
    tmp5 = tl.load(in_ptr2 + (r0_2 + 64*x0), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp3 * tmp2
    tmp6 = tmp4 + tmp5
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
    tmp9 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
    tmp11 = tl.sum(tmp9, 1)[:, None].to(tl.float32)
    tmp12 = tl.full([1, 1], 64, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = (tmp11 / tmp13)
    tmp15 = tmp7 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, R0_BLOCK])
    tmp19 = tl.sum(tmp17, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
    tl.store(out_ptr1 + (x3), tmp19, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 2097152
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_addmm_mul_native_group_norm_sigmoid_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2097152, 64)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((4096,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32768, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((4096,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32768, 64, 1, 1), (64, 1, 2097152, 2097152), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((32768, 64, 1, 1), (64, 1, 2097152, 2097152), device='cuda:0', dtype=torch.float32)
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
