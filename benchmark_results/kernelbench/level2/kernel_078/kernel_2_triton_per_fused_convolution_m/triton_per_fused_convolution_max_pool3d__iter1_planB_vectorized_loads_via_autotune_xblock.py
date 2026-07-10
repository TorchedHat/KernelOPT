import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=4),
        triton.Config({'XBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused_convolution_max_pool3d_with_indices_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1024000
    r0_numel = 27
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_4 = (r0_index % 3)
    r0_5 = ((r0_index // 3) % 3)
    r0_6 = r0_index // 9
    x0 = (xindex % 10)
    x1 = ((xindex // 10) % 10)
    x2 = ((xindex // 100) % 10)
    x3 = xindex // 1000
    x7 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_4 + 3*x0 + 31*r0_5 + 93*x1 + 961*r0_6 + 2883*x2 + 29792*x3), mask=xmask & r0_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x7), tmp4, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 1024000
    r0_numel = 27
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_convolution_max_pool3d_with_indices_2[grid](arg_0, arg_1, 1024000, 27)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 64, 31, 31, 31), (1906688, 29792, 961, 31, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 64, 10, 10, 10), (64000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
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
