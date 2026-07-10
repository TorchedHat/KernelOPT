import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 2048}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1,  'R0_BLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 512},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 256},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 128},  num_warps=4, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_sigmoid_sum_4(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 640
    r0_numel = 115520
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_1 = tl.max_contiguous(tl.multiple_of(r0_index, [1, 16]), [1, 16])
        tmp0 = tl.load(in_ptr0 + (r0_1 + 115520*x0), mask=r0_mask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.sigmoid(tmp0)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
        tmp4 = _tmp3 + tmp2
        _tmp3 = tl.where(r0_mask & xmask, tmp4, _tmp3)
    tmp3 = tl.sum(_tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp3, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 640
    r0_numel = 115520
    grid = (triton.cdiv(xnumel, 1),)
    triton_red_fused_sigmoid_sum_4[grid](arg_0, arg_1, 640, 115520)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 64, 95, 95), (577600, 9025, 95, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 5), (5, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

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
