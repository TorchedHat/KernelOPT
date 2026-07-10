import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_min_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x3 = xindex
    x0 = (xindex % 64)
    _tmp15 = tl.full([XBLOCK, R0_BLOCK], float("inf"), tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), r0_mask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr1 + (8*x3 + (r0_2 // 16)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = tl.load(in_ptr2 + (8*x3 + (r0_2 // 16)), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp10 = tl.load(in_ptr3 + (r0_2 + 128*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp12 = tl.load(in_ptr4 + (r0_2 + 128*x0), r0_mask, eviction_policy='evict_last', other=0.0)
        tmp2 = tmp0 - tmp1
        tmp4 = tl.full([1, 1], 16.0, tl.float32)
        tmp5 = (tmp3 / tmp4)
        tmp6 = tl.full([1, 1], 1e-05, tl.float32)
        tmp7 = tmp5 + tmp6
        tmp8 = libdevice.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp11 = tmp9 * tmp10
        tmp13 = tmp11 + tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
        tmp16 = tl.minimum(_tmp15, tmp14)
        _tmp15 = tl.where(r0_mask, tmp16, _tmp15)
    tmp15 = triton_helpers.min2(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, None)



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
    triton_red_fused_min_native_group_norm_2[grid](arg_0, arg_1, arg_2, arg_3, arg_4, arg_5, 65536, 128, XBLOCK=128, R0_BLOCK=512)
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
