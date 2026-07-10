import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    # xnumel = 3072 = 128*24 (batch * heads)
    # r0_numel = 900 = 30*30 (spatial, stride=1 -> CONTIGUOUS)
    xnumel = 3072
    r0_numel = 900
    R0_BLOCK: tl.constexpr = 1024  # next power-of-2 >= 900
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
    r0_2 = r0_index  # spatial index 0..899, stride=1 (CONTIGUOUS)
    x_batch = xindex // 24   # 0..127
    x_head  = xindex % 24    # 0..23
    # Address: spatial + head*900 + batch*21600
    # r0_2 strides by 1 -> contiguous -> no bank conflicts
    addr = r0_2 + 900*x_head + 21600*x_batch
    combined_mask = r0_mask & xmask
    tmp0 = tl.load(in_out_ptr0 + addr, mask=combined_mask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, R0_BLOCK])
    tmp3 = tl.broadcast_to(tmp1, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(combined_mask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 1)[:, None].to(tl.float32)
    tmp7 = tmp1 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, R0_BLOCK])
    tmp11 = tl.where(combined_mask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 1)[:, None].to(tl.float32)
    tmp13 = tmp0 - tmp6
    tmp14 = tl.exp(tmp13)
    tmp15 = (tmp14 / tmp12)
    tl.store(in_out_ptr0 + addr, tmp15, mask=combined_mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    xnumel = 115200
    r0_numel = 24
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, 115200, 24)
    return arg_0

def get_inputs():
    arg_0 = rand_strided((128, 24, 30, 30), (21600, 900, 30, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0]

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
