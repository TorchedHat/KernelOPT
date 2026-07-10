import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 64, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 32, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 64, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 32, 'R0_BLOCK': 128}, num_warps=2, num_stages=2),
], key=['xnumel', 'r0_numel'])
@triton.jit
def triton_red_fused_min_native_group_norm_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 65536
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = (xindex < xnumel)
    x3 = xindex
    x0 = (xindex % 64)
    # R0_BLOCK == r0_numel == 128: single pass, no loop, no mask needed on r0
    r0_1d = tl.arange(0, R0_BLOCK)
    r0_1d = tl.max_contiguous(tl.multiple_of(r0_1d, 128), 128)
    r0_index = r0_1d[None, :]
    r0_2 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), mask=xmask, eviction_policy='evict_first', other=0.0)
    tmp1 = tl.load(in_ptr1 + (8*x3 + (r0_2 // 16)), mask=xmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tl.load(in_ptr2 + (8*x3 + (r0_2 // 16)), mask=xmask, eviction_policy='evict_last', other=0.0)
    tmp10 = tl.load(in_ptr3 + (r0_2 + 128*x0), mask=xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.load(in_ptr4 + (r0_2 + 128*x0), mask=xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 - tmp1
    tmp5 = tmp3 * 0.0625
    tmp6 = tl.full([1, 1], 1e-05, tl.float32)
    tmp7 = tmp5 + tmp6
    tmp8 = libdevice.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, R0_BLOCK])
    tmp15 = triton_helpers.min2(tmp14, 1)[:, None]
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
