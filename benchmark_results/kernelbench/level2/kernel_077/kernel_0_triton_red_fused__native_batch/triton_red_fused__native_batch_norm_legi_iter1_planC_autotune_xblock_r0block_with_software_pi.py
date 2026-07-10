import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=8, num_stages=3),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_mul_0(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 25920
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 128)
    tmp1 = tl.load(in_ptr1 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr3 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr4 + (x0), mask=xmask, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), mask=xmask, eviction_policy='evict_last')
    _tmp21 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 25920*x3), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 2.0, tl.float32)
        tmp4 = tmp2 * tmp3
        tmp6 = tmp4 - tmp5
        tmp8 = tl.full([1, 1], 1e-05, tl.float32)
        tmp9 = tmp7 + tmp8
        tmp10 = tl.sqrt_rn(tmp9)
        tmp11 = tl.full([1, 1], 1, tl.int32)
        tmp12 = (tmp11 / tmp10)
        tmp13 = tl.full([1, 1], 1.0, tl.float32)
        tmp14 = tmp12 * tmp13
        tmp15 = tmp6 * tmp14
        tmp17 = tmp15 * tmp16
        tmp19 = tmp17 + tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, R0_BLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(r0_mask & xmask, tmp22, _tmp21)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tmp23 = tl.full([1, 1], 25920.0, tl.float32)
    tmp24 = (tmp21 / tmp23)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp24, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 2048
    r0_numel = 25920
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused__native_batch_norm_legit_no_training_convolution_mean_mul_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 2048, 25920)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((16, 128, 1, 1, 1), (128, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 128, 20, 36, 36), (3317760, 25920, 1296, 36, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
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
