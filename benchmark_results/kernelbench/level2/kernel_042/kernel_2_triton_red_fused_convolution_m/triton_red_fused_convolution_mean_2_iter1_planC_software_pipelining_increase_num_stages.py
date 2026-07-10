import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 512},  num_warps=8),
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 512},  num_warps=4),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 256},  num_warps=4),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 256},  num_warps=8),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 256},  num_warps=4),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused_convolution_mean_2(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 264192
    r0_numel = 2049
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_base = tl.arange(0, R0_BLOCK)[None, :]
    rbase = r0_base
    x1 = ((xindex // 128) % 129)
    x0 = (xindex % 128)
    x2 = xindex // 16512
    _tmp9 = tl.full([XBLOCK, R0_BLOCK], 0, tl.float32)
    x4 = xindex
    for r0_offset in tl.range(0, r0_numel, R0_BLOCK):
        r0_index = r0_offset + r0_base
        r0_mask = r0_index < r0_numel
        roffset = r0_offset
        rindex = r0_index
        r0_3 = r0_index
        tmp0 = r0_3 + 2049*x1
        tmp1 = tl.full([1, 1], 264196, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + 128*(((r0_3 + 2049*x1) % 264196)) + 33817088*x2), mask=r0_mask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.load(in_ptr1 + (tl.broadcast_to(x0, [XBLOCK, R0_BLOCK])), mask=r0_mask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp5 = tmp3 + tmp4
        tmp6 = tl.full(tmp5.shape, 0, tmp5.dtype)
        tmp7 = tl.where(tmp2, tmp5, tmp6)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, R0_BLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(r0_mask & xmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tl.store(out_ptr0 + (x4), tmp9, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 264192
    r0_numel = 2049
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused_convolution_mean_2[grid](arg_0, arg_1, arg_2, 264192, 2049)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((16, 128, 514, 514), (33817088, 1, 65792, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 128, 1, 1, 129), (16512, 1, 264192, 264192, 128), device='cuda:0', dtype=torch.float32)
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
