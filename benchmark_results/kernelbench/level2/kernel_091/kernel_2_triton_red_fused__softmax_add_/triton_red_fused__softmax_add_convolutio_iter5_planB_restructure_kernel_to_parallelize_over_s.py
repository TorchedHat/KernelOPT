import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 64,  'R0_BLOCK': 128}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'R0_BLOCK': 128}, num_warps=4, num_stages=3),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    # xnumel = 2130048 (batch * spatial = 128 * 16641)
    # r0_numel = 128 (channels)
    # New tiling: program_id(0) covers spatial x0 (stride-1 in output) grouped by batch x1
    # Grid: cdiv(16641, XBLOCK) * 128
    pid = tl.program_id(0)
    batch_blocks = tl.cdiv(16641, XBLOCK)
    x1 = pid // batch_blocks               # batch index 0..127
    spatial_block = pid % batch_blocks
    xoffset = spatial_block * XBLOCK

    x0_index = xoffset + tl.arange(0, XBLOCK)[None, :]  # [1, XBLOCK] spatial positions
    r0_index = tl.arange(0, R0_BLOCK)[:, None]           # [R0_BLOCK, 1] channels

    x0_mask = x0_index < 16641
    r0_mask = r0_index < 128

    # Load in_ptr0 with strides (2130048, 1, 16512, 128):
    # address = r0_index (channel, stride=1) + 128*x0_index (spatial, stride=128) + 2130048*x1 (batch)
    tmp0 = tl.load(in_ptr0 + (r0_index + 128 * x0_index + 2130048 * x1),
                   mask=r0_mask & x0_mask, other=0.0)  # [R0_BLOCK, XBLOCK]

    # Load in_ptr1 (bias, shape (128,), stride (1,)): broadcast over spatial
    tmp1 = tl.load(in_ptr1 + r0_index, mask=r0_mask, other=0.0)  # [R0_BLOCK, 1]

    scores = tmp0 + tmp1  # [R0_BLOCK, XBLOCK]

    # Softmax reduce over channels (axis=0) — no loop since R0_BLOCK==r0_numel==128
    scores_max = tl.max(scores, axis=0, keep_dims=True)     # [1, XBLOCK]
    scores_exp = tl.exp(scores - scores_max)                 # [R0_BLOCK, XBLOCK]
    scores_sum = tl.sum(scores_exp, axis=0, keep_dims=True)  # [1, XBLOCK]
    softmax_out = scores_exp / scores_sum                    # [R0_BLOCK, XBLOCK]

    # Load in_ptr2 (arg_2, shape (128, 1, 1), strides (1, 1, 1)): broadcast over spatial
    tmp12 = tl.load(in_ptr2 + r0_index, mask=r0_mask, other=0.0)  # [R0_BLOCK, 1]

    out = tl.sigmoid(2.0 * (softmax_out + tmp12))  # [R0_BLOCK, XBLOCK]

    # Store: out_ptr2 strides (2130048, 16641, 129, 1)
    # address = x0_index (stride-1, COALESCED) + 16641*r0_index + 2130048*x1
    tl.store(out_ptr2 + (x0_index + 16641 * r0_index + 2130048 * x1),
             out, mask=r0_mask & x0_mask)

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
