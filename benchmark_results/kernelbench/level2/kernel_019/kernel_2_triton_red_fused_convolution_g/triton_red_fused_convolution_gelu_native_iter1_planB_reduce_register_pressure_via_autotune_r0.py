import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 256},  num_warps=8,  num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_convolution_gelu_native_group_norm_2(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 532512
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 8)
    x1 = xindex // 8
    tmp12_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp12_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp12_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 8)
        r0_3 = r0_index // 8
        tmp0 = tl.load(in_ptr0 + (r0_2 + 8*x0 + 64*r0_3 + 4260096*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 8*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.full([1, 1], 0.5, tl.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = tl.full([1, 1], 0.7071067811865476, tl.float32)
        tmp6 = tmp2 * tmp5
        tmp7 = libdevice.erf(tmp6)
        tmp8 = tl.full([1, 1], 1.0, tl.float32)
        tmp9 = tmp7 + tmp8
        tmp10 = tmp4 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
        tmp12_mean_next, tmp12_m2_next, tmp12_weight_next = triton_helpers.welford_reduce(
            tmp11, tmp12_mean, tmp12_m2, tmp12_weight, roffset == 0
        )
        tmp12_mean = tl.where(r0_mask & xmask, tmp12_mean_next, tmp12_mean)
        tmp12_m2 = tl.where(r0_mask & xmask, tmp12_m2_next, tmp12_m2)
        tmp12_weight = tl.where(r0_mask & xmask, tmp12_weight_next, tmp12_weight)
    tmp13, tmp14, tmp15 = triton_helpers.welford(tmp12_mean, tmp12_m2, tmp12_weight, 1)
    tmp12 = tmp13[:, None]
    tmp16 = tmp14[:, None]
    tmp17 = tmp15[:, None]
    tl.store(out_ptr0 + (x4), tmp12, mask=xmask)
    tl.store(out_ptr1 + (x4), tmp16, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 1024
    r0_numel = 532512
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused_convolution_gelu_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, 1024, 532512)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 64, 258, 258), (4260096, 1, 16512, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 8, 1, 1), (8, 1, 1024, 1024), device='cuda:0', dtype=torch.float32)
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
