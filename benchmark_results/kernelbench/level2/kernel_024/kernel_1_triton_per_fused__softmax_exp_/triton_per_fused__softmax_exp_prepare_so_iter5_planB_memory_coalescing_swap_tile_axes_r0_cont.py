import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    # NEW: num_warps=2 gives larger register budget per warp (XBLOCK>=128 required for correctness with fixed grid=900)
    triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=2),
    triton.Config({'XBLOCK': 256}, num_warps=2, num_stages=2),
    triton.Config({'XBLOCK': 128}, num_warps=2, num_stages=3),
    triton.Config({'XBLOCK': 256}, num_warps=2, num_stages=3),
    # EXISTING: keep all original configs
    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
    # NEW: num_stages=3 variants for software pipelining
    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=3),
    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=3),
], key=['xnumel'], restore_value=['in_out_ptr0'])
@triton.jit
def triton_per_fused__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 115200
    r0_numel = 24
    R0_BLOCK: tl.constexpr = 32
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[:, None]
    r0_offset = 0
    r0_mask = r0_index < r0_numel
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    x0 = (xindex % 900)
    x1 = xindex // 900
    x3 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0 + 900*r0_2 + 21600*x1), mask=r0_mask & xmask, other=0.0, eviction_policy='evict_first')
    tmp1 = tl.broadcast_to(tmp0, [R0_BLOCK, XBLOCK])
    tmp3 = tl.broadcast_to(tmp1, [R0_BLOCK, XBLOCK])
    tmp5 = tl.where(r0_mask & xmask, tmp3, float("-inf"))
    tmp6 = triton_helpers.max2(tmp5, 0)[None, :].to(tl.float32)
    tmp7 = tmp1 - tmp6
    tmp8 = tl.exp(tmp7)
    tmp9 = tl.broadcast_to(tmp8, [R0_BLOCK, XBLOCK])
    tmp11 = tl.where(r0_mask & xmask, tmp9, 0)
    tmp12 = tl.sum(tmp11, 0)[None, :].to(tl.float32)
    tmp13 = tmp0 - tmp6
    tmp14 = tl.exp(tmp13)
    tmp15 = (tmp14 / tmp12)
    tl.store(in_out_ptr0 + (x0 + 900*r0_2 + 21600*x1), tmp15, mask=r0_mask & xmask, eviction_policy='evict_first')

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
