import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # XBLOCK=1: fully coalesced loads (r0_2 stride-1), minimal register pressure
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 128}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 1, 'R0_BLOCK': 128}, num_warps=2, num_stages=4),
        # XBLOCK=2: slight x-parallelism, still low register pressure
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 128}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 2, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
        # Fallback larger configs (autotune will reject if slower)
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2130048
    r0_numel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = (xindex % 16641)
    x1 = xindex // 16641
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_mask = r0_index < r0_numel
    r0_2 = r0_index
    tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), mask=r0_mask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + r0_2, mask=r0_mask, other=0.0)
    scores = tmp0 + tmp1
    scores_max = tl.max(scores, axis=1, keep_dims=True)
    scores_exp = tl.exp(scores - scores_max)
    scores_sum = tl.sum(scores_exp, axis=1, keep_dims=True)
    softmax_out = scores_exp / scores_sum
    tmp12 = tl.load(in_ptr2 + r0_2, mask=r0_mask, other=0.0)
    tmp13 = softmax_out + tmp12
    tmp15 = tmp13 * 2.0
    tmp16 = tl.sigmoid(tmp15)
    tl.store(out_ptr2 + (x0 + 16641*r0_2 + 2130048*x1), tmp16, mask=r0_mask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    xnumel = 2130048
    r0_numel = 128
    grid = (triton.cdiv(xnumel, 128),)
    triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2[grid](arg_0, arg_1, arg_2, arg_3, 2130048, 128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 128, 129, 129), (2130048, 1, 16512, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 128, 129, 129), (2130048, 16641, 129, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
