import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 8192}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 4096}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 4096}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 2048}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 1024}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 512},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128,'R0_BLOCK': 512},  num_warps=4, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
    restore_value=['in_out_ptr0'],
)
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

    # --- Hoist scalar load and sign computation BEFORE all loops ---
    _scale_val = tl.load(in_ptr5 + tl.zeros([XBLOCK, 1], dtype=tl.int32), mask=xmask, other=0.0)
    _scale_bc  = tl.broadcast_to(_scale_val, [XBLOCK, 1])
    _sign      = tl.where(_scale_bc >= 0.0, 1.0, -1.0)   # preserves +1 for scale=0
    _abs_scale = _sign * _scale_bc                         # |scale| = sign(scale)*scale

    _tmp26 = tl.full([XBLOCK, R0_BLOCK], float("-inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0  = tl.load(in_ptr0 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1  = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3  = tl.load(in_ptr1 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5  = tl.load(in_ptr2 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp14 = tl.load(in_ptr3 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp16 = tl.load(in_ptr4 + (r0_1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2  = tmp0 + tmp1
        tmp4  = tmp2 - tmp3
        # Replace two-step (1/sqrt_rn(x))*1.0 with single rsqrt instruction
        tmp_inv_std = tl.math.rsqrt(tmp5 + 1e-5)
        tmp13 = tmp4 * tmp_inv_std
        tmp15 = tmp13 * tmp14
        tmp17 = tmp15 + tmp16
        # Use hoisted _sign instead of recomputing
        tmp24 = tmp17 * _sign
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, R0_BLOCK])
        tmp27 = tl.maximum(_tmp26, tmp25)
        _tmp26 = tl.where(r0_mask & xmask, tmp27, _tmp26)
        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp24, mask=r0_mask & xmask)
    tmp26 = triton_helpers.max2(_tmp26, 1)[:, None]

    _tmp41 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp28 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp29 = tmp28 - tmp26
        # Use hoisted _abs_scale instead of recomputing sign * scale
        tmp38 = tmp29 * _abs_scale
        tmp39 = tl.exp(tmp38)
        tmp40 = tl.broadcast_to(tmp39, [XBLOCK, R0_BLOCK])
        tmp42 = _tmp41 + tmp40
        _tmp41 = tl.where(r0_mask & xmask, tmp42, _tmp41)
    tmp41 = tl.sum(_tmp41, 1)[:, None]

    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp43 = tl.load(in_out_ptr0 + (r0_1 + 8192*x0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp44 = tmp43 - tmp26
        # Use hoisted _abs_scale instead of recomputing sign * scale
        tmp53 = tmp44 * _abs_scale
        tmp54 = tl.exp(tmp53)
        tmp55 = (tmp54 / tmp41)
        tl.store(in_out_ptr0 + (r0_1 + 8192*x0), tmp55, mask=r0_mask & xmask)

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
