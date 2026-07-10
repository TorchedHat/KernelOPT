import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 256},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256},  num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 128},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 64},   num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 64},   num_warps=8,  num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 256},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 64},   num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 64},   num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 64},   num_warps=8,  num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_convolution_native_group_norm_0(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 1024
    r0_numel = 59400
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x4 = xindex
    x0 = (xindex % 8)
    tmp4_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp4_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_5 = r0_index
        r0_3 = r0_index // 19800
        tmp0 = tl.load(in_ptr0 + (r0_5 + 59400*x4), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_3 + 3*x0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
        tmp4_mean_next, tmp4_m2_next, tmp4_weight_next = triton_helpers.welford_reduce(
            tmp3, tmp4_mean, tmp4_m2, tmp4_weight, roffset == 0
        )
        tmp4_mean = tl.where(r0_mask & xmask, tmp4_mean_next, tmp4_mean)
        tmp4_m2 = tl.where(r0_mask & xmask, tmp4_m2_next, tmp4_m2)
        tmp4_weight = tl.where(r0_mask & xmask, tmp4_weight_next, tmp4_weight)
    tmp5, tmp6, tmp7 = triton_helpers.welford(tmp4_mean, tmp4_m2, tmp4_weight, 1)
    tmp4 = tmp5[:, None]
    tmp8 = tmp6[:, None]
    tmp9 = tmp7[:, None]
    tl.store(out_ptr0 + (x4), tmp4, mask=xmask, eviction_policy='evict_last')
    tl.store(out_ptr1 + (x4), tmp8, mask=xmask, eviction_policy='evict_last')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 1024
    r0_numel = 59400
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused_convolution_native_group_norm_0[grid](arg_0, arg_1, arg_2, arg_3, 1024, 59400)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 24, 22, 30, 30), (475200, 19800, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((24,), (1,), device='cuda:0', dtype=torch.float32)
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
