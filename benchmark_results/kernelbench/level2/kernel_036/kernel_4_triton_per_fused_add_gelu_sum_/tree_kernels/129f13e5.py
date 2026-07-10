import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},  num_warps=1),
        triton.Config({'XBLOCK': 4},  num_warps=1),
        triton.Config({'XBLOCK': 8},  num_warps=1),
        triton.Config({'XBLOCK': 16}, num_warps=2),
        triton.Config({'XBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 64}, num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused_add_gelu_sum_4(in_out_ptr0, in_ptr0, in_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 4096
    r0_numel = 2
    R0_BLOCK: tl.constexpr = 2
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    tmp12 = tl.load(in_ptr1 + tl.full([1, 1], 0, tl.int32), mask=tl.full([1, 1], True, tl.int1), other=0.0)
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[None, :]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 256)
    x1 = xindex // 256
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (256*r0_2 + x0 + 512*x1), mask=xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.sum(tmp1, 1)[:, None].to(tl.float32)
    tmp4 = tl.full([1, 1], 0.5, tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.full([1, 1], 0.7071067811865476, tl.float32)
    tmp7 = tmp3 * tmp6
    tmp8 = libdevice.erf(tmp7)
    tmp9 = tl.full([1, 1], 1.0, tl.float32)
    tmp10 = tmp8 + tmp9
    tmp11 = tmp5 * tmp10
    tmp14 = tmp11 + tmp12
    tl.store(in_out_ptr0 + (x3), tmp14, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 4096
    r0_numel = 2
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_gelu_sum_4[grid](arg_0, arg_1, arg_2, 4096, 2)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((16, 1, 1, 256), (256, 256, 256, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 1, 1, 256, 2), (512, 8192, 8192, 1, 256), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
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
