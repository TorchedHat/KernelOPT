import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=1),
        triton.Config({'XBLOCK': 2}, num_warps=1),
        triton.Config({'XBLOCK': 4}, num_warps=1),
        triton.Config({'XBLOCK': 4}, num_warps=2),
        triton.Config({'XBLOCK': 8}, num_warps=2),
        triton.Config({'XBLOCK': 8}, num_warps=4),
        triton.Config({'XBLOCK': 16}, num_warps=2),
        triton.Config({'XBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 32}, num_warps=2),
        triton.Config({'XBLOCK': 32}, num_warps=4),
        triton.Config({'XBLOCK': 64}, num_warps=4),
        triton.Config({'XBLOCK': 64}, num_warps=8),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_clamp_convolution_max_pool3d_with_indices_mean_mul_2(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 66
    R0_BLOCK: tl.constexpr = 128
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 66*x0), mask=r0_mask & xmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.where(r0_mask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None].to(tl.float32)
    tmp5 = tl.full([1, 1], 14415.0, tl.float32)
    tmp6 = (tmp4 / tmp5)
    tmp7 = tl.full([1, 1], 0.0, tl.float32)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp9 = tl.full([1, 1], 1.0, tl.float32)
    tmp10 = tl.minimum(tmp8, tmp9)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp10, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 2048
    r0_numel = 66
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_clamp_convolution_max_pool3d_with_indices_mean_mul_2[grid](arg_0, arg_1, 2048, 66)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 16, 1, 1, 1), (16, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 16, 1, 1, 1, 66), (1056, 66, 135168, 135168, 135168, 1), device='cuda:0', dtype=torch.float32)
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
