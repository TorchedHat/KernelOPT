import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1},   num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 1},   num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 2},   num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 2},   num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 4},   num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 4},   num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 8},   num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 8},   num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 16},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 16},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64},  num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 256}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_sum_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 16000
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    r0_index = tl.arange(0, R0_BLOCK)[:, None]   # [R0_BLOCK, 1] — r0 is OUTER
    x_inner  = tl.arange(0, XBLOCK)[None, :]     # [1, XBLOCK]   — x is INNER (stride 1)
    xindex_2d = xoffset + x_inner                # [1, XBLOCK]
    xmask_2d  = xindex_2d < xnumel
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)[:, None]
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0_2d = (xindex_2d % 1000)
    x1_2d = xindex_2d // 1000
    tmp0 = tl.load(in_ptr0 + (x0_2d + 1000*r0_2 + 64000*x1_2d), xmask_2d, other=0.0)
    tmp1 = tl.where(xmask_2d, tmp0, 0.0)
    tmp4 = tl.sum(tmp1, 0).to(tl.float32)        # reduce axis 0 (r0) -> [XBLOCK]
    xindex_1d = xoffset + tl.arange(0, XBLOCK)   # 1D for store
    xmask_1d  = xindex_1d < xnumel
    tl.store(out_ptr0 + xindex_1d, tmp4, xmask_1d)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 16000
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_sum_3[grid](arg_0, arg_1, 16000, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 64, 10, 10, 10), (64000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 1, 10, 10, 10), (1000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
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
