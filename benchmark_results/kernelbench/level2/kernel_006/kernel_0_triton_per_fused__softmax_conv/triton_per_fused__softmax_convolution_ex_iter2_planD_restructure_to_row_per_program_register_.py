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
        triton.Config({'XBLOCK': 64},  num_warps=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1612800
    r0_numel = 16
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)    # 1D: [XBLOCK]
    xmask = xindex < xnumel                     # [XBLOCK]
    r0_index = tl.arange(0, R0_BLOCK)           # 1D: [R0_BLOCK]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    xi = xindex[:, None]     # [XBLOCK, 1]
    ri = r0_index[None, :]   # [1, R0_BLOCK]
    r0_2 = ri
    x0 = (xi % 12600)        # [XBLOCK, 1]
    x1 = xi // 12600         # [XBLOCK, 1]
    tmp0 = tl.load(in_ptr0 + (x0 + 12600*r0_2 + 201600*x1), xmask[:, None], other=0.0)   # [XBLOCK, R0_BLOCK]
    tmp1 = tl.load(in_ptr1 + (r0_2), xmask[:, None], other=0.0, eviction_policy='evict_last')  # [XBLOCK, R0_BLOCK] broadcast
    tmp2 = tmp0 + tmp1                                                                           # [XBLOCK, R0_BLOCK]
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp7 = tl.where(xmask[:, None], tmp5, float("-inf"))
    tmp8 = tl.max(tmp7, axis=1, keep_dims=True).to(tl.float32)   # [XBLOCK, 1]
    tmp9 = tmp3 - tmp8
    tmp10 = tl.exp(tmp9)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, R0_BLOCK])
    tmp13 = tl.where(xmask[:, None], tmp11, 0)
    tmp14 = tl.sum(tmp13, axis=1, keep_dims=True).to(tl.float32)  # [XBLOCK, 1]
    tmp15 = tmp2 - tmp8
    tmp16 = tl.exp(tmp15)
    tmp17 = (tmp16 / tmp14)
    tl.store(out_ptr2 + (x0 + 12608*r0_2 + 201728*x1), tmp17, xmask[:, None])

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 1612800
    r0_numel = 16
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0[grid](arg_0, arg_1, arg_2, 1612800, 16)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 16, 14, 30, 30), (201600, 12600, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 16, 14, 30, 30), (201728, 12608, 900, 30, 1), device='cuda:0', dtype=torch.float32)
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
