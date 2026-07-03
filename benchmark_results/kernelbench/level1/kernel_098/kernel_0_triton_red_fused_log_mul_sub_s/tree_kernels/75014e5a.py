import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK':  512}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK':  256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK':  512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK':  256}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32, 'R0_BLOCK':  128}, num_warps=8, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_log_mul_sub_sum_xlogy_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 131072
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp14 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
        roffset = r0_offset
        rindex = r0_index
        r0_1 = r0_index
        tmp0 = tl.load(in_ptr0 + (r0_1 + 131072*x0), xmask, eviction_policy='evict_first', other=0.0)
        tmp9 = tl.load(in_ptr1 + (r0_1 + 131072*x0), xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = libdevice.isnan(tmp0).to(tl.int1)
        tmp2 = tl.full([1, 1], 0.0, tl.float32)
        tmp3 = tmp0 == tmp2
        tmp4 = tl.log(tmp0)
        tmp5 = tmp0 * tmp4
        tmp6 = tl.where(tmp3, tmp2, tmp5)
        tmp7 = tl.full([1, 1], float("nan"), tl.float32)
        tmp8 = tl.where(tmp1, tmp7, tmp6)
        tmp10 = tl.log(tmp9)
        tmp11 = tmp0 * tmp10
        tmp12 = tmp8 - tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, R0_BLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(xmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp14, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 2048
    r0_numel = 131072
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused_log_mul_sub_sum_xlogy_0[grid](arg_0, arg_1, arg_2, 2048, 131072)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((16384, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16384, 16384), (16384, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
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
