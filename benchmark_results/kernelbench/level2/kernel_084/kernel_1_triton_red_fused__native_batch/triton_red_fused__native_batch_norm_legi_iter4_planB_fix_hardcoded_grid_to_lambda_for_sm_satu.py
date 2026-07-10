import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 1,   'R0_BLOCK': 8192}, num_warps=4,  num_stages=1, maxnreg=128),
    triton.Config({'XBLOCK': 1,   'R0_BLOCK': 4096}, num_warps=4,  num_stages=1, maxnreg=128),
    triton.Config({'XBLOCK': 2,   'R0_BLOCK': 8192}, num_warps=4,  num_stages=1, maxnreg=128),
    triton.Config({'XBLOCK': 2,   'R0_BLOCK': 4096}, num_warps=4,  num_stages=1, maxnreg=128),
    triton.Config({'XBLOCK': 4,   'R0_BLOCK': 4096}, num_warps=4,  num_stages=1, maxnreg=128),
    triton.Config({'XBLOCK': 4,   'R0_BLOCK': 2048}, num_warps=8,  num_stages=1, maxnreg=128),
    triton.Config({'XBLOCK': 8,   'R0_BLOCK': 8192}, num_warps=8,  num_stages=1, maxnreg=128),
    triton.Config({'XBLOCK': 16,  'R0_BLOCK': 4096}, num_warps=8,  num_stages=2, maxnreg=128),
    triton.Config({'XBLOCK': 32,  'R0_BLOCK': 2048}, num_warps=8,  num_stages=2, maxnreg=128),
    triton.Config({'XBLOCK': 64,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=2, maxnreg=128),
    triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512},  num_warps=4,  num_stages=2, maxnreg=128),
], key=['xnumel', 'r0_numel'], restore_value=['in_out_ptr0'])
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 8192
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex

    # Load scale scalar once outside loops (scalar, no mask needed but use mask=None is fine)
    tmp_scale = tl.load(in_ptr5 + 0, mask=None)
    tmp_scale_bc = tl.broadcast_to(tmp_scale, [1, 1])
    tmp_zero = tl.full([1, 1], 0.0, tl.float32)
    tmp_pos = tl.full([1, 1], 1.0, tl.float32)
    tmp_neg = tl.full([1, 1], -1.0, tl.float32)
    sign = tl.where(tmp_scale_bc >= tmp_zero, tmp_pos, tmp_neg)

    # Loop 1 (ONLINE): compute scores inline, update running max+sum, NO store to in_out_ptr0
    running_max = tl.full([XBLOCK, 1], float('-inf'), tl.float32)
    running_sum = tl.full([XBLOCK, 1], 0.0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        # Compute batch-norm score
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        # Apply sign
        tmp24 = tmp17 * sign
        score = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        # Online softmax update — mask out-of-bounds elements with -inf for max, 0 for sum
        score_masked = tl.where(r0_mask & xmask, score, tl.full([XBLOCK, R0_BLOCK], float('-inf'), tl.float32))
        tile_max = tl.max(score_masked, axis=1)[:, None]
        new_max = tl.maximum(running_max, tile_max)
        # Rescale running_sum for new max, add current tile's contribution
        exp_score = tl.where(r0_mask & xmask, tl.exp(score - new_max), tl.zeros([XBLOCK, R0_BLOCK], tl.float32))
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.sum(exp_score, axis=1)[:, None]
        running_max = new_max

    # Loop 2: recompute scores, write final softmax (replaces original loops 2+3)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tl.load(in_ptr1 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr2 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (r0_1), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        # Recompute batch-norm score
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 - tmp3
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = tl.sqrt_rn(tmp7)
        tmp9 = tl.full([1, 1], 1, tl.int32)
        tmp10 = (tmp9 / tmp8)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tmp10 * tmp11
        tmp13 = tmp4 * tmp12
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        # Apply sign
        tmp24 = tmp17 * sign
        score = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        # Compute final softmax value
        softmax_val = tl.exp(score - running_max) / running_sum
        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), softmax_val, mask=r0_mask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 1024
    r0_numel = 8192
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__native_batch_norm_legit_no_training__softmax_addmm_amax_ge_mul_neg_scalar_tensor_sub_where_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 1024, 8192)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((1,), (1,), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6]

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
