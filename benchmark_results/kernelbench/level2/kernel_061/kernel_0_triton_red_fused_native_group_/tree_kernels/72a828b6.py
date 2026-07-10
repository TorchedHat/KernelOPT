import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # num_stages=1 baseline
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=8, num_stages=1),
        # num_stages=2: overlap load with Welford compute
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 2048}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 2048}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 1024}, num_warps=8, num_stages=2),
        # num_stages=3: deeper pipeline for higher latency hiding
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 512},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 1024}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512},  num_warps=8, num_stages=3),
        # large XBLOCK + small R0_BLOCK to cut register pressure
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 128},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 64},   num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 128},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 64},   num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 64},   num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 128},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 64},   num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 128},  num_warps=8, num_stages=3),
        # NEW: tiny R0_BLOCK configs for more waves and lower register pressure
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 32}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 16}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 32}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 16}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 32}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 32}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 32}, num_warps=4, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_native_group_norm_relu_0(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 640
    r0_numel = 125773
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 5)
    x1 = xindex // 5
    tmp17_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp17_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x3 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = r0_2 + 125773*x0
        tmp1 = tl.full([1, 1], 628864, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (628864*x1 + (((r0_2 + 125773*x0) % 628864))), mask=r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full([1, 1], 0, tl.int32)
        tmp5 = tl.maximum(tmp4, tmp3)
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp11 = tl.full([1, 1], 1.0, tl.float32)
        tmp12 = tl.full(tmp11.shape, 0, tmp11.dtype)
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp16 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        _new_count    = tmp17_weight + tmp16
        _delta        = tmp14 - tmp17_mean
        _weight_ratio = tmp16 / _new_count
        _mean_next    = tl.math.fma(_delta, _weight_ratio, tmp17_mean)
        _delta2       = tmp14 - _mean_next
        _m2_next      = tl.math.fma(_delta, _delta2, tmp17_m2)
        _valid        = r0_mask & xmask
        tmp17_mean    = tl.where(_valid, _mean_next,   tmp17_mean)
        tmp17_m2      = tl.where(_valid, _m2_next,     tmp17_m2)
        tmp17_weight  = tl.where(_valid, _new_count,   tmp17_weight)
    tmp18, tmp19, tmp20 = triton_helpers.welford(tmp17_mean, tmp17_m2, tmp17_weight, 1)
    tmp17 = tmp18[:, None]
    tmp21 = tmp19[:, None]
    tmp22 = tmp20[:, None]
    tl.store(out_ptr0 + (x3), tmp17, mask=xmask, eviction_policy='evict_first')
    tl.store(out_ptr1 + (x3), tmp21, mask=xmask, eviction_policy='evict_first')
    tl.store(out_ptr2 + (x3), tmp22, mask=xmask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 640
    r0_numel = 125773
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused_native_group_norm_relu_0[grid](arg_0, arg_1, arg_2, arg_3, 640, 125773)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((16, 128, 34, 34, 34), (5030912, 39304, 1156, 34, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16, 8, 1, 1, 5), (40, 5, 640, 640, 1), device='cuda:0', dtype=torch.float32)
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
