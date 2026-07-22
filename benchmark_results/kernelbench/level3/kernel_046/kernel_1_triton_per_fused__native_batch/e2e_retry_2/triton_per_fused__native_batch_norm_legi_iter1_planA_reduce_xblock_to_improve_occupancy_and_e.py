import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 204800
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (r0_1 + 32*x0), xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr2 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
    tmp14 = tl.load(in_ptr3 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 + tl.full([1, 1], 1e-05, tl.float32)
    inv_std = libdevice.rsqrt(tmp5)
    tmp11 = tmp2 * inv_std
    tmp15 = tmp11 * tmp12 + tmp14
    row_max = tl.max(tmp15, axis=1)[:, None]
    shifted = tmp15 - row_max
    exps = tl.exp(shifted)
    row_sum = tl.sum(exps, axis=1)[:, None]
    result = exps / row_sum
    tl.store(in_out_ptr0 + (r0_1 + 32*x0), result, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 204800
    r0_numel = 32
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__native_batch_norm_legit_no_training__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 204800, 32)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((204800, 32), (32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
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
