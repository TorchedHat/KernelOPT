import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 31360
    r0_numel = 96
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 3136)
    x1 = xindex // 3136
    x3 = xindex
    tmp10_mean = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_m2 = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    tmp10_weight = tl.zeros([XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 3136*r0_2 + 301056*x1), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (32*((((x0 % 56)) % 7)) + 224*(((x0 // 56) % 7)) + 1568*(r0_2 // 32) + 4704*(((x0 % 56)) // 7) + 37632*(x0 // 392) + 301056*x1 + ((r0_2 % 32))), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (7*(((x0 // 56) % 7)) + 49*(r0_2 // 32) + ((((x0 % 56)) % 7))), r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tl.load(in_ptr3 + (r0_2), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp6 = tl.load(in_out_ptr0 + (r0_2 + 96*x3), r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp7 = tmp5 + tmp6
        tmp8 = tmp4 + tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
        tmp10_mean_next, tmp10_m2_next, tmp10_weight_next = triton_helpers.welford_reduce(
            tmp9, tmp10_mean, tmp10_m2, tmp10_weight, roffset == 0
        )
        tmp10_mean = tl.where(r0_mask & xmask, tmp10_mean_next, tmp10_mean)
        tmp10_m2 = tl.where(r0_mask & xmask, tmp10_m2_next, tmp10_m2)
        tmp10_weight = tl.where(r0_mask & xmask, tmp10_weight_next, tmp10_weight)
        tl.store(in_out_ptr0 + (r0_2 + 96*x3), tmp8, r0_mask & xmask)
    tmp11, tmp12, tmp13 = triton_helpers.welford(tmp10_mean, tmp10_m2, tmp10_weight, 1)
    tmp10 = tmp11[:, None]
    tmp14 = tmp12[:, None]
    tmp15 = tmp13[:, None]
    tl.store(out_ptr0 + (x3), tmp10, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    arg_5 = inputs[5]
    arg_6 = inputs[6]
    xnumel = 31360
    r0_numel = 96
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__unsafe_view_add_addmm_clone_convolution_native_layer_norm_permute_transpose_view_6[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, arg_6, 31360, 96, XBLOCK=128, R0_BLOCK=512)
    return arg_6

def get_inputs():
    arg_0 = rand_strided((10, 3136, 96), (301056, 96, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3136, 96), (301056, 1, 3136), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((640, 147, 32), (4704, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((147,), (1,), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_5 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
    arg_6 = rand_strided((10, 3136, 1), (3136, 1, 31360), device='cuda:0', dtype=torch.float32)
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
