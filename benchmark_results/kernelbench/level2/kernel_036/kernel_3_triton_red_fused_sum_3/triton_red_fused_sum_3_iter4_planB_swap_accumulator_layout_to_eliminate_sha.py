import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 512, 'R0_BLOCK': 128}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=4, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_sum_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 8192
    r0_numel = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x0 = (xindex % 256)
    x1 = xindex // 256
    _tmp2 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_2 = r0_index
        tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_2 + 32768*x1), mask=xmask & r0_mask, eviction_policy='evict_last', other=0.0)
        tmp3 = _tmp2 + tmp0
        _tmp2 = tl.where(r0_mask, tmp3, _tmp2)
    tmp2 = tl.sum(_tmp2, 1)[:, None]
    store_offset = (x1 // 2) * 512 + (x1 % 2) * 256 + x0
    tl.store(out_ptr0 + store_offset, tmp2, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 8192
    r0_numel = 128
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused_sum_3[grid](arg_0, arg_1, 8192, 128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 1, 256, 256), (65536, 1048576, 256, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 1, 1, 256, 2), (512, 8192, 8192, 1, 256), device='cuda:0', dtype=torch.float32)
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
