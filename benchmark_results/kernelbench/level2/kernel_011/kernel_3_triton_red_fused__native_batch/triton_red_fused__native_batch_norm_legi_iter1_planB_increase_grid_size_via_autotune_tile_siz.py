import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 512}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 64}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 512}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 512}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 4, 'R0_BLOCK': 256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 512}, num_warps=2, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 4624
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 8)
    x1 = xindex // 8
    tmp8_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp8_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = (r0_index % 16)
        r0_3 = r0_index // 16
        tmp0 = tl.load(in_ptr0 + (r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr0 + (128 + r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr0 + (4352 + r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr0 + (4480 + r0_2 + 16*x0 + 256*((r0_3 % 17)) + 8704*(r0_3 // 17) + 147968*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.maximum(tmp0, tmp1)
        tmp4 = tl.maximum(tmp2, tmp3)
        tmp6 = tl.maximum(tmp4, tmp5)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, R0_BLOCK])
        tmp8_mean_next, tmp8_m2_next, tmp8_weight_next = triton_helpers.welford_reduce(
            tmp7, tmp8_mean, tmp8_m2, tmp8_weight, roffset == 0
        )
        tmp8_mean = tl.where(r0_mask, tmp8_mean_next, tmp8_mean)
        tmp8_m2 = tl.where(r0_mask, tmp8_m2_next, tmp8_m2)
        tmp8_weight = tl.where(r0_mask, tmp8_weight_next, tmp8_weight)
    tmp9, tmp10, tmp11 = triton_helpers.welford(tmp8_mean, tmp8_m2, tmp8_weight, 1)
    tmp8 = tmp9[:, None]
    tmp12 = tmp10[:, None]
    tmp13 = tmp11[:, None]
    tl.store(out_ptr0 + (x4), tmp8, mask=xmask)
    tl.store(out_ptr1 + (x4), tmp12, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 4096
    r0_numel = 4624
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_native_group_norm_tanh_3[grid](arg_0, arg_1, arg_2, 4096, 4624)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((512, 128, 34, 34), (147968, 1, 4352, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 8, 1, 1), (8, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((512, 8, 1, 1), (8, 1, 4096, 4096), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
