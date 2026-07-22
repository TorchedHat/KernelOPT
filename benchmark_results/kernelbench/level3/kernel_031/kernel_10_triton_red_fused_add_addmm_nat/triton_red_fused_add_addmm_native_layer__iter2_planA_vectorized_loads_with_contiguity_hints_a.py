import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_red_fused_add_addmm_native_layer_norm_permute_view_10(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, ynumel, xnumel, r0_numel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 2
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None, None]
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, None, :]
    rbase = r0_base
    x1 = xindex
    y0 = yindex
    tmp6_mean = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    tmp6_m2 = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    tmp6_weight = tl.zeros([YBLOCK, XBLOCK, R0_BLOCK], tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = tl.max_contiguous(tl.multiple_of(r0_index, [1, 1, R0_BLOCK]), [1, 1, R0_BLOCK])
        tmp0 = tl.load(in_ptr0 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r0_2 + 128*x1 + 256*y0), mask=r0_mask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (y0 + 16384*r0_2 + 2097152*x1), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [YBLOCK, XBLOCK, R0_BLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight, roffset == 0
        )
        tmp6_mean = tl.where(r0_mask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(r0_mask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(r0_mask & xmask, tmp6_weight_next, tmp6_weight)
    tmp7, tmp8, tmp9 = triton_helpers.welford(tmp6_mean, tmp6_m2, tmp6_weight, 2)
    tmp6 = tmp7[:, :, None]
    tmp10 = tmp8[:, :, None]
    tmp11 = tmp9[:, :, None]
    tl.store(out_ptr0 + (x1 + 2*y0), tmp6, xmask)
    tl.store(out_ptr1 + (x1 + 2*y0), tmp10, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    arg_4 = inputs[4]
    ynumel = 16384
    xnumel = 2
    r0_numel = 128
    grid = (triton.cdiv(xnumel, 512), triton.cdiv(ynumel, 128))
    triton_red_fused_add_addmm_native_layer_norm_permute_view_10[grid](arg_0, arg_1, arg_2, arg_3, arg_4, 16384, 2, 128, YBLOCK=8, XBLOCK=128, R0_BLOCK=512)
    return arg_4

def get_inputs():
    arg_0 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32768, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2, 128, 128, 128), (2097152, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((16384, 2, 1), (2, 1, 32768), device='cuda:0', dtype=torch.float32)
    arg_4 = rand_strided((16384, 2, 1), (2, 1, 32768), device='cuda:0', dtype=torch.float32)
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
