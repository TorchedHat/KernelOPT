import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 256},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_convolution_hardswish_native_group_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 43732
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 4)
    tmp13_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp13_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_3 = r0_index // 10933
        tmp0 = tl.load(in_ptr0 + (r0_5 + 43732*x4), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 4*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 3.0, tl.float32)
        tmp4 = tmp2 + tmp3
        tmp5 = tl.full([1, 1], 0.0, tl.float32)
        tmp6 = tl.maximum(tmp4, tmp5)
        tmp7 = tl.full([1, 1], 6.0, tl.float32)
        tmp8 = tl.minimum(tmp6, tmp7)
        tmp9 = tmp2 * tmp8
        tmp10 = tl.full([1, 1], 0.16666666666666666, tl.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp13_mean_next, tmp13_m2_next, tmp13_weight_next = triton_helpers.welford_reduce(
            tmp12, tmp13_mean, tmp13_m2, tmp13_weight, roffset == 0
        )
        tmp13_mean = tl.where(r0_mask, tmp13_mean_next, tmp13_mean)
        tmp13_m2 = tl.where(r0_mask, tmp13_m2_next, tmp13_m2)
        tmp13_weight = tl.where(r0_mask, tmp13_weight_next, tmp13_weight)
    tmp14, tmp15, tmp16 = triton_helpers.welford(tmp13_mean, tmp13_m2, tmp13_weight, 1)
    tmp13 = tmp14[:, None]
    tmp17 = tmp15[:, None]
    tmp18 = tmp16[:, None]
    tl.store(out_ptr0 + (x4), tmp13, mask=xmask)
    tl.store(out_ptr1 + (x4), tmp17, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 4096
    r0_numel = 43732
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused_convolution_hardswish_native_group_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 4096, 43732)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((1024, 16, 13, 29, 29), (174928, 10933, 841, 29, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 4, 1, 1), (4, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1024, 4, 1, 1), (4, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
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
