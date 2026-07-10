import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16},  num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 16},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 32},  num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 64},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_convolution_min_tanh_2(in_ptr0, in_ptr1, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 8258048
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x3 = xindex
    x0 = (xindex % 64516)
    x1 = xindex // 64516
    tmp0 = tl.load(in_ptr0 + (r0_2 + 64*x3), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.load(in_ptr1 + (r0_2), xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, float("inf"))
    tmp6 = triton_helpers.min2(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = libdevice.tanh(tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tl.store(out_ptr1 + (x3), tmp8, xmask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 8258048
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_convolution_min_tanh_2[grid](arg_0, arg_1, arg_2, 8258048, 64)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 64, 254, 254), (4129024, 1, 16256, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 1, 254, 254), (64516, 64516, 254, 1), device='cuda:0', dtype=torch.float32)
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
