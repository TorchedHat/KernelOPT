import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=3, maxnreg=64),
    triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=3, maxnreg=64),
    triton.Config({'XBLOCK': 64,  'R0_BLOCK': 64},  num_warps=4, num_stages=4, maxnreg=64),
    triton.Config({'XBLOCK': 128, 'R0_BLOCK': 64},  num_warps=4, num_stages=4, maxnreg=64),
    triton.Config({'XBLOCK': 64,  'R0_BLOCK': 32},  num_warps=2, num_stages=3, maxnreg=96),
    triton.Config({'XBLOCK': 128, 'R0_BLOCK': 32},  num_warps=8, num_stages=2, maxnreg=96),
    triton.Config({'XBLOCK': 64,  'R0_BLOCK': 16},  num_warps=2, num_stages=4, maxnreg=96),
    triton.Config({'XBLOCK': 128, 'R0_BLOCK': 16},  num_warps=4, num_stages=4, maxnreg=96),
    triton.Config({'XBLOCK': 8,   'R0_BLOCK': 128}, num_warps=1, num_stages=2),
    triton.Config({'XBLOCK': 8,   'R0_BLOCK': 64},  num_warps=1, num_stages=2),
    triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=1, num_stages=2),
    triton.Config({'XBLOCK': 16,  'R0_BLOCK': 64},  num_warps=1, num_stages=2),
], key=['xnumel', 'r0_numel'])
@triton.jit
def triton_red_fused_min_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 64)

    # Precompute per-group mean and inv_std outside the reduction loop.
    # Load each of the 8 groups as a separate [XBLOCK, 1] tensor (contiguous per x-row).
    mean0 = tl.load(in_ptr1 + (8*x3 + 0), mask=xmask, eviction_policy='evict_last', other=0.0)
    mean1 = tl.load(in_ptr1 + (8*x3 + 1), mask=xmask, eviction_policy='evict_last', other=0.0)
    mean2 = tl.load(in_ptr1 + (8*x3 + 2), mask=xmask, eviction_policy='evict_last', other=0.0)
    mean3 = tl.load(in_ptr1 + (8*x3 + 3), mask=xmask, eviction_policy='evict_last', other=0.0)
    mean4 = tl.load(in_ptr1 + (8*x3 + 4), mask=xmask, eviction_policy='evict_last', other=0.0)
    mean5 = tl.load(in_ptr1 + (8*x3 + 5), mask=xmask, eviction_policy='evict_last', other=0.0)
    mean6 = tl.load(in_ptr1 + (8*x3 + 6), mask=xmask, eviction_policy='evict_last', other=0.0)
    mean7 = tl.load(in_ptr1 + (8*x3 + 7), mask=xmask, eviction_policy='evict_last', other=0.0)

    var0 = tl.load(in_ptr2 + (8*x3 + 0), mask=xmask, eviction_policy='evict_last', other=0.0)
    var1 = tl.load(in_ptr2 + (8*x3 + 1), mask=xmask, eviction_policy='evict_last', other=0.0)
    var2 = tl.load(in_ptr2 + (8*x3 + 2), mask=xmask, eviction_policy='evict_last', other=0.0)
    var3 = tl.load(in_ptr2 + (8*x3 + 3), mask=xmask, eviction_policy='evict_last', other=0.0)
    var4 = tl.load(in_ptr2 + (8*x3 + 4), mask=xmask, eviction_policy='evict_last', other=0.0)
    var5 = tl.load(in_ptr2 + (8*x3 + 5), mask=xmask, eviction_policy='evict_last', other=0.0)
    var6 = tl.load(in_ptr2 + (8*x3 + 6), mask=xmask, eviction_policy='evict_last', other=0.0)
    var7 = tl.load(in_ptr2 + (8*x3 + 7), mask=xmask, eviction_policy='evict_last', other=0.0)

    # Compute inv_std for each group once, outside the loop
    inv_std0 = libdevice.rsqrt(var0 / 16.0 + 1e-05)
    inv_std1 = libdevice.rsqrt(var1 / 16.0 + 1e-05)
    inv_std2 = libdevice.rsqrt(var2 / 16.0 + 1e-05)
    inv_std3 = libdevice.rsqrt(var3 / 16.0 + 1e-05)
    inv_std4 = libdevice.rsqrt(var4 / 16.0 + 1e-05)
    inv_std5 = libdevice.rsqrt(var5 / 16.0 + 1e-05)
    inv_std6 = libdevice.rsqrt(var6 / 16.0 + 1e-05)
    inv_std7 = libdevice.rsqrt(var7 / 16.0 + 1e-05)

    _tmp15 = tl.full([XBLOCK, R0_BLOCK], float("inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), mask=xmask & r0_mask, eviction_policy='evict_first', other=0.0)
        g = r0_2 // 16  # shape [1, R0_BLOCK], values in [0..7]

        # Select mean and inv_std for the correct group using tl.where chain
        # mean0..mean7 are [XBLOCK, 1]; g is [1, R0_BLOCK] -> broadcast to [XBLOCK, R0_BLOCK]
        tmp1 = tl.where(g == 0, mean0,
               tl.where(g == 1, mean1,
               tl.where(g == 2, mean2,
               tl.where(g == 3, mean3,
               tl.where(g == 4, mean4,
               tl.where(g == 5, mean5,
               tl.where(g == 6, mean6,
                                mean7)))))))
        tmp8 = tl.where(g == 0, inv_std0,
               tl.where(g == 1, inv_std1,
               tl.where(g == 2, inv_std2,
               tl.where(g == 3, inv_std3,
               tl.where(g == 4, inv_std4,
               tl.where(g == 5, inv_std5,
               tl.where(g == 6, inv_std6,
                                inv_std7)))))))

        tmp10 = tl.load(in_ptr3 + (r0_2 + 128*x0), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r0_2 + 128*x0), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp9 = (tmp0 - tmp1) * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = tl.minimum(_tmp15, tmp14)
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp15 = triton_helpers.min2(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    xnumel = 65536
    r0_numel = 128
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused_min_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 65536, 128)
    return arg_5

def get_inputs():
    arg_0 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 512, 1, 1), (512, 1, 524288, 524288), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 512, 1, 1), (512, 1, 524288, 524288), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((1024, 1, 64), (64, 65536, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3, arg_4, arg_5]

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
