import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 256}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512}, num_warps=8, num_stages=3),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x0 = xindex
    _tmp12 = tl.zeros([XBLOCK, 1], tl.float32)
    for a in tl.static_range(32):
        tmp1_h = tl.load(in_ptr1 + (32 * x0 + a), mask=xmask, eviction_policy='evict_last', other=0.0)
        tmp5_h = tl.load(in_ptr3 + (32 * x0 + a), mask=xmask, eviction_policy='evict_last', other=0.0)
        tmp6_h = tl.sqrt_rn(tmp5_h)
        tmp7 = tl.full([1, 1], 1e-12, tl.float32)
        tmp8_h = tl.maximum(tmp6_h, tmp7)
        for b_offset in tl.range(0, 512, R0_BLOCK):
            b = b_offset + tl.arange(0, R0_BLOCK)[None, :]
            tmp0 = tl.load(in_ptr0 + (512 * a + 16384 * x0 + b), mask=xmask, eviction_policy='evict_first', other=0.0)
            tmp2 = tl.load(in_ptr2 + (32 * b + a), mask=xmask, eviction_policy='evict_last', other=0.0)
            tmp3 = tmp1_h * tmp2
            tmp4 = tmp0 - tmp3
            tmp9 = tmp4 / tmp8_h
            tmp10 = tmp9 * tmp9
            _tmp12 = _tmp12 + tl.sum(tmp10, 1)[:, None]
    tmp12 = _tmp12
    for a in tl.static_range(32):
        tmp1_h = tl.load(in_ptr1 + (32 * x0 + a), mask=xmask, eviction_policy='evict_last', other=0.0)
        tmp5_h = tl.load(in_ptr3 + (32 * x0 + a), mask=xmask, eviction_policy='evict_last', other=0.0)
        tmp20_h = tl.sqrt_rn(tmp5_h)
        tmp21 = tl.full([1, 1], 1e-12, tl.float32)
        tmp22_h = tl.maximum(tmp20_h, tmp21)
        tmp24 = tl.sqrt_rn(tmp12)
        tmp25 = tl.maximum(tmp24, tmp21)
        for b_offset in tl.range(0, 512, R0_BLOCK):
            b = b_offset + tl.arange(0, R0_BLOCK)[None, :]
            tmp14 = tl.load(in_ptr0 + (512 * a + 16384 * x0 + b), mask=xmask, eviction_policy='evict_first', other=0.0)
            tmp16 = tl.load(in_ptr2 + (32 * b + a), mask=xmask, eviction_policy='evict_last', other=0.0)
            tmp17 = tmp1_h * tmp16
            tmp18 = tmp14 - tmp17
            tmp23 = tmp18 / tmp22_h
            tmp26 = tmp23 / tmp25
            tl.store(out_ptr1 + (32 * b + a + 16384 * x0), tmp26, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 2048
    r0_numel = 16384
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused__unsafe_view_clamp_min_clone_div_expand_linalg_vector_norm_mul_sub_transpose_5[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2048, 16384)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((2048, 32, 512), (16384, 512, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048, 1, 32), (32, 65536, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 512, 32), (16384, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((2048, 1, 32), (32, 65536, 1), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((2048, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
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
