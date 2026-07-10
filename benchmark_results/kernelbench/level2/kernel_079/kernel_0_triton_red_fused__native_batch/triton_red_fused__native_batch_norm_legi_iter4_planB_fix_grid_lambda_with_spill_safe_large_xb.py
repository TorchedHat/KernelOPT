import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # Original 10 configs (kept unchanged)
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 256},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 512},  num_warps=8, num_stages=1),
        # num_warps=2: 64 threads/block -> 3072 regs/block -> more blocks/SM
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 512},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 2,  'R0_BLOCK': 512},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=2, num_stages=1),
        # num_stages=2: software pipeline to hide L1TEX latency
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256},  num_warps=2, num_stages=2),
        # large XBLOCK: coalesce stores + eliminate 728-block partial wave
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=1),
        # NEW: safe large-XBLOCK configs with small R0_BLOCK (XBLOCK*R0_BLOCK <= 2048)
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 32},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 16},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 8},   num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 4},   num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 32},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 16},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 8},   num_warps=2, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_convolution_mul_view_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 12600
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + ((x0 % 16)), mask=xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((x0 % 16)), mask=xmask, eviction_policy='evict_last')
    tmp6_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 12600*x0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 1)
    tmp6 = tmp7[:, None]
    tmp10 = tmp8[:, None]
    tmp11 = tmp9[:, None]
    tl.store(out_ptr0 + (x0), tmp6, mask=xmask)
    tl.store(out_ptr1 + (x0), tmp10, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    xnumel = 2048
    r0_numel = 12600
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused__native_batch_norm_legit_convolution_mul_view_0[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 2048, 12600)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((128, 16, 14, 30, 30), (201600, 12600, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((1, 2048, 1, 1, 1), (2048, 1, 2048, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((1, 2048, 1, 1, 1), (2048, 1, 2048, 2048, 2048), device='cuda:0', dtype=torch.float32)
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
