import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 8},   num_warps=1, num_stages=1, maxnreg=64),
        triton.Config({'XBLOCK': 8},   num_warps=1, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 16},  num_warps=1, num_stages=1, maxnreg=64),
        triton.Config({'XBLOCK': 16},  num_warps=1, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 16},  num_warps=2, num_stages=1, maxnreg=64),
        triton.Config({'XBLOCK': 16},  num_warps=2, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=1, maxnreg=64),
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=2, maxnreg=64),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=1, maxnreg=64),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1, maxnreg=64),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1, maxnreg=64),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1, maxnreg=64),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=1, maxnreg=64),
    ],
    key=['xnumel', 'r0_numel'],
    restore_value=['in_out_ptr0'],
)
@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 32768
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
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
    x0 = (xindex % 1024)
    x1 = xindex // 1024
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + 1024*r0_2 + 32768*x1), xmask & r0_mask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    # rsqrt replaces sqrt+int_div+identity_mul (3 ops -> 1 MUFU.RSQ)
    inv_std = tl.math.rsqrt(tmp5 + 1e-5)
    tmp13 = tmp4 * inv_std
    # FMA fuses gamma*normalized + beta (2 ops -> 1 FFMA)
    tmp17 = tl.math.fma(tmp13, tmp14, tmp16)
    tmp17_tile = tl.broadcast_to(tmp17, [XBLOCK, R0_BLOCK])
    row_max = triton_helpers.max2(tmp17_tile, 1)[:, None].to(tl.float32)
    exp_shifted = tl.exp(tmp17_tile - row_max)
    row_sum = tl.sum(exp_shifted, 1)[:, None].to(tl.float32)
    tmp30 = exp_shifted / row_sum
    tl.store(in_out_ptr0 + (x0 + 1024*r0_2 + 32768*x1), tmp30, xmask & r0_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 32768
    r0_numel = 32
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__native_batch_norm_legit_no_training__softmax_convolution_exp_max_pool2d_with_indices_prepare_softmax_online_sub_22[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 32768, 32)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((8, 1024, 4, 32), (131072, 1, 32768, 1024), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((1024,), (1,), device='cuda:0', dtype=torch.float32)
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
