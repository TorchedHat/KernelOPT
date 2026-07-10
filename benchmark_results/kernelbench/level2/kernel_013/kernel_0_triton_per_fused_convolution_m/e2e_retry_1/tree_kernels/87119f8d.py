import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 256},  num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=3),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_convolution_mean_0(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
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
    r0_3 = r0_index
    x0 = (xindex % 16384)
    x4 = xindex // 16384
    x1 = ((xindex // 16384) % 64)
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + 16384*r0_3 + 524288*x4), mask=xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x1), mask=xmask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x5), tmp5, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 16777216
    r0_numel = 32
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_convolution_mean_0[grid](arg_0, arg_1, arg_2, 16777216, 32)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((16, 64, 32, 128, 128), (33554432, 524288, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((16, 64, 1, 128, 128), (1048576, 16384, 16777216, 128, 1), device='cuda:0', dtype=torch.float32)
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
