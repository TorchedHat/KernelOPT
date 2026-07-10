import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 1024}, num_warps=2, num_stages=1),
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
    # Scalar-shaped accumulators: [XBLOCK, 1] — O(1) registers regardless of R0_BLOCK
    _mean   = tl.zeros([XBLOCK, 1], tl.float32)
    _m2     = tl.zeros([XBLOCK, 1], tl.float32)
    _weight = tl.zeros([XBLOCK, 1], tl.float32)
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
        tmp4 = tmp2 + 3.0
        tmp6 = tl.maximum(tmp4, 0.0)
        tmp8 = tl.minimum(tmp6, 6.0)
        tmp11 = tmp2 * tmp8 * 0.16666666666666666
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        # Compute tile welford stats from scratch (zero-initialized tile accumulators)
        _tile_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        _tile_m2   = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        _tile_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
        _tile_mean_next, _tile_m2_next, _tile_weight_next = triton_helpers.welford_reduce(
            tmp12, _tile_mean, _tile_m2, _tile_weight, True
        )
        _tile_mean   = tl.where(r0_mask, _tile_mean_next,   _tile_mean)
        _tile_m2     = tl.where(r0_mask, _tile_m2_next,     _tile_m2)
        _tile_weight = tl.where(r0_mask, _tile_weight_next, _tile_weight)
        # Collapse tile to scalar via welford merge along dim=1
        _tile_mean_s, _tile_m2_s, _tile_weight_s = triton_helpers.welford(_tile_mean, _tile_m2, _tile_weight, 1)
        # Merge tile scalar result with running scalar accumulator using Welford parallel merge
        # Running: (_mean[:, 0], _m2[:, 0], _weight[:, 0])
        # Tile:    (_tile_mean_s, _tile_m2_s, _tile_weight_s)
        _run_mean   = _mean[:, 0]
        _run_m2     = _m2[:, 0]
        _run_weight = _weight[:, 0]
        _merged_mean, _merged_m2, _merged_weight = triton_helpers.welford(
            tl.join(_run_mean[:, None], _tile_mean_s[:, None]),
            tl.join(_run_m2[:, None],   _tile_m2_s[:, None]),
            tl.join(_run_weight[:, None], _tile_weight_s[:, None]),
            1
        )
        _mean   = _merged_mean[:, None]
        _m2     = _merged_m2[:, None]
        _weight = _merged_weight[:, None]
    tl.store(out_ptr0 + (x4), _mean, mask=xmask)
    tl.store(out_ptr1 + (x4), _m2,   mask=xmask)

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
