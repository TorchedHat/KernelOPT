import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_stages=1, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_stages=2, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_stages=3, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_stages=4, num_warps=4),
        triton.Config({'XBLOCK': 128}, num_stages=2, num_warps=8),
        triton.Config({'XBLOCK': 128}, num_stages=3, num_warps=8),
        triton.Config({'XBLOCK': 128}, num_stages=4, num_warps=8),
        triton.Config({'XBLOCK': 64},  num_stages=2, num_warps=8),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_native_group_norm_1(in_ptr0, out_ptr0, out_ptr1, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 262144
    r0_numel = 32
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    x0 = xoffset + tl.arange(0, XBLOCK)[:, None]      # shape [XBLOCK, 1]
    r0_index_1d = tl.arange(0, R0_BLOCK)                                        # shape [R0_BLOCK], 1D
    r0_index_1d = tl.max_contiguous(tl.multiple_of(r0_index_1d, 32), 32)        # vectorized hint on 1D only
    r0_index = r0_index_1d[None, :]                                              # shape [1, R0_BLOCK]
    xmask = x0 < xnumel
    tmp0 = tl.load(in_ptr0 + (r0_index + 32*x0), xmask, other=0.0, eviction_policy='evict_first')
    # tmp0 shape: [XBLOCK, R0_BLOCK] — r0 varies fastest (stride-1), coalesced!
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.sum(tmp3, 1)[:, None].to(tl.float32)    # sum over axis=1 (R0_BLOCK)
    tmp8 = tmp5 * (1.0 / 32.0)
    tmp9 = tmp1 - tmp8
    tmp10 = tmp9 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.sum(tmp11, 1)[:, None].to(tl.float32)  # sum over axis=1 (R0_BLOCK)
    tl.store(out_ptr0 + (x0), tmp8, xmask, eviction_policy='evict_last')
    tl.store(out_ptr1 + (x0), tmp13, xmask, eviction_policy='evict_last')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 262144
    r0_numel = 32
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_native_group_norm_1[grid](arg_0, arg_1, arg_2, 262144, 32)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((1024, 8192), (8192, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 256, 1, 1), (256, 1, 262144, 262144), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((1024, 256, 1, 1), (256, 1, 262144, 262144), device='cuda:0', dtype=torch.float32)
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
