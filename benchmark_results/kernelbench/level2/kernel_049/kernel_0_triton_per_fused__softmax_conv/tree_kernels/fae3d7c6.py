import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # num_stages=1 (baseline, no pipelining)
        triton.Config({'XBLOCK': 1},  num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 1},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 2},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 4},  num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 4},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 8},  num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 16}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=1),
        # num_stages=2 (one prefetch ahead)
        triton.Config({'XBLOCK': 1},  num_warps=1, num_stages=2),
        triton.Config({'XBLOCK': 1},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 2},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 4},  num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 4},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=2),
        # num_stages=3 (two prefetches ahead)
        triton.Config({'XBLOCK': 1},  num_warps=1, num_stages=3),
        triton.Config({'XBLOCK': 1},  num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 2},  num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 4},  num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 4},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 8},  num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 16}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32}, num_warps=8, num_stages=3),
    ],
    key=['xnumel', 'r0_numel'],
    restore_value=['in_out_ptr0'],
)
@triton.jit
def triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    r0_index = tl.arange(0, R0_BLOCK)[:, None]
    r0_offset = 0
    r0_mask = tl.full([R0_BLOCK, 1], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_2 = r0_index
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x0 = (xindex % 131072)
    x1 = xindex // 131072
    tmp0 = tl.load(in_out_ptr0 + (x0 + 131072*r0_2 + 8388608*x1), mask=xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (r0_2), mask=r0_mask, eviction_policy='evict_last', other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [R0_BLOCK, XBLOCK])
    tmp5 = tl.broadcast_to(tmp3, [R0_BLOCK, XBLOCK])
    tmp7 = triton_helpers.max2(tmp5, 0)[None, :].to(tl.float32)
    tmp8 = tmp3 - tmp7
    tmp9 = tl.exp(tmp8)
    tmp10 = tl.broadcast_to(tmp9, [R0_BLOCK, XBLOCK])
    tmp12 = tl.sum(tmp10, 0)[None, :].to(tl.float32)
    tmp13 = tmp2 - tmp7
    tmp14 = tl.exp(tmp13)
    tmp15 = (tmp14 / tmp12)
    tmp16 = tl.sigmoid(tmp15)
    tl.store(in_out_ptr0 + (x0 + 131072*r0_2 + 8388608*x1), tmp16, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 2097152
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sigmoid_sub_0[grid](arg_0, arg_1, 2097152, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 64, 32, 64, 64), (8388608, 131072, 4096, 64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
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
