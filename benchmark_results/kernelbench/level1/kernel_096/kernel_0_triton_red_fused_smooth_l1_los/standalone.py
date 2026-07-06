import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_red_fused_smooth_l1_loss_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 131072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = tl.full([XBLOCK], True, tl.int1)[:, None]
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp13 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 131072*x0), None, eviction_policy='evict_first')
        tmp1 = tl.load(in_ptr1 + (r0_1 + 131072*x0), None, eviction_policy='evict_first')
        tmp2 = tmp0 - tmp1
        tmp3 = tl_math.abs(tmp2)
        tmp4 = tl.full([1, 1], 1.0, tl.float32)
        tmp5 = tmp3 < tmp4
        tmp6 = tmp3 * tmp3
        tmp7 = tl.full([1, 1], 0.5, tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8 * tmp4
        tmp10 = tmp3 - tmp7
        tmp11 = tl.where(tmp5, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, R0_BLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tmp14
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp13, None)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 8192
    r0_numel = 131072
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused_smooth_l1_loss_0[grid](arg_0, arg_1, arg_2, 8192, 131072, XBLOCK=1, R0_BLOCK=2048)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((32768, 32768), (32768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32768, 32768), (32768, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((8192,), (1,), device='cuda:0', dtype=torch.float32)
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
