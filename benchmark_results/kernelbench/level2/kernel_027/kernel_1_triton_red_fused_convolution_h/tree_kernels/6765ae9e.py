import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 2048}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK':  512}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 1024}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_convolution_hardswish_mean_native_group_norm_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 16384
    r0_numel = 10933
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 16)
    tmp1 = tl.load(in_ptr1 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr2 + (x4 // 4), mask=xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr3 + (x4 // 4), mask=xmask, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
    # Move loop-invariant rsqrt computation outside the loop
    tmp19 = libdevice.rsqrt(tmp14 / 43732.0 + 1e-05)
    # Precompute loop-invariant scale and bias adjustment
    tmp_scale = tmp21 * tmp19                  # gamma * inv_std
    tmp_bias_adj = tmp23 - tmp12 * tmp_scale   # beta - mean * gamma * inv_std
    _tmp26 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_5 + 10933*x4), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp_xb = tmp0 + tmp1                                                          # x + conv_bias
        tmp_xb3 = tmp_xb + 3.0                                                        # x + bias + 3
        tmp_clamp = tl.where(tmp_xb3 <= 0.0, 0.0, tl.where(tmp_xb3 >= 6.0, 6.0, tmp_xb3))  # clamp(x+3,0,6)
        tmp_hs = tmp_xb * tmp_clamp * 0.16666666666666666                             # hardswish(x+bias)
        tmp_norm = tl.broadcast_to(tmp_hs * tmp_scale + tmp_bias_adj, [XBLOCK, R0_BLOCK])  # fused norm+affine
        _tmp26 = tl.where(r0_mask, _tmp26 + tmp_norm, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp28 = tl.full([1, 1], 10933.0, tl.float32)
    tmp29 = (tmp26 / tmp28)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x4), tmp29, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 16384
    r0_numel = 10933
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused_convolution_hardswish_mean_native_group_norm_1[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 16384, 10933)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((1024, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 16, 13, 29, 29), (174928, 10933, 841, 29, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1024, 4, 1, 1), (4, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1024, 4, 1, 1), (4, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
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
