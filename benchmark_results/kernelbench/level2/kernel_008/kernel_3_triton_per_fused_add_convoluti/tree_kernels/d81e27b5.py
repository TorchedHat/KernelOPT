import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},   num_warps=1,  num_stages=1),
        triton.Config({'XBLOCK': 1},   num_warps=1,  num_stages=2),
        triton.Config({'XBLOCK': 2},   num_warps=1,  num_stages=1),
        triton.Config({'XBLOCK': 2},   num_warps=1,  num_stages=2),
        triton.Config({'XBLOCK': 4},   num_warps=1,  num_stages=1),
        triton.Config({'XBLOCK': 4},   num_warps=1,  num_stages=2),
        triton.Config({'XBLOCK': 8},   num_warps=2,  num_stages=1),
        triton.Config({'XBLOCK': 8},   num_warps=2,  num_stages=2),
        triton.Config({'XBLOCK': 16},  num_warps=2,  num_stages=1),
        triton.Config({'XBLOCK': 16},  num_warps=2,  num_stages=2),
        triton.Config({'XBLOCK': 32},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 32},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4,  num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=16, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3(in_ptr0, in_ptr1, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 128
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
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
    r0_1 = r0_index
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r0_1 + 16*x0), xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r0_1), xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.full([1, 1], 6727.0, tl.float32)
    tmp2 = (tmp0 / tmp1)
    tmp4 = tmp2 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None].to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp8, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 128
    r0_numel = 16
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_add_convolution_div_max_pool3d_with_indices_mean_sum_3[grid](arg_0, arg_1, arg_2, 128, 16)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 16, 1, 1, 1), (16, 1, 2048, 2048, 2048), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
