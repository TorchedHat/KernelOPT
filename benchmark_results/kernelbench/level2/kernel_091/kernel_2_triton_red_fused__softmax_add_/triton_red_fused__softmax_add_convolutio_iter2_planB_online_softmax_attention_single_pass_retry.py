import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 32, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16, 'R0_BLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 8,  'R0_BLOCK': 128}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 4,  'R0_BLOCK': 128}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 64, 'R0_BLOCK': 128}, num_warps=8, num_stages=2),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_red_fused__softmax_add_convolution_exp_mul_prepare_softmax_online_sigmoid_sub_2(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr, R0_BLOCK : tl.constexpr):
    xnumel = 2130048
    r0_numel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]   # [XBLOCK, 1]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]          # [1, R0_BLOCK]
    r0_mask = r0_index < r0_numel
    x3 = xindex
    x0 = (xindex % 16641)
    x1 = xindex // 16641
    r0_2 = r0_index

    # Single vectorized load (ONE pass over in_ptr0, not two)
    tmp0 = tl.load(in_ptr0 + (r0_2 + 128*x3), mask=r0_mask & xmask, other=0.0)   # [XBLOCK, R0_BLOCK]
    tmp1 = tl.load(in_ptr1 + r0_2, mask=r0_mask, other=0.0)                        # [1, R0_BLOCK]
    tmp12 = tl.load(in_ptr2 + r0_2, mask=r0_mask, other=0.0)                       # [1, R0_BLOCK]

    score = tmp0 + tmp1                                  # [XBLOCK, R0_BLOCK]

    # Numerically stable softmax in one shot (no loop-carried state)
    # Use tl.max with keep_dims=False then reshape to [XBLOCK, 1]
    row_max = tl.max(score, axis=1)                      # [XBLOCK]
    row_max = row_max[:, None]                           # [XBLOCK, 1]
    score_shifted = score - row_max                      # [XBLOCK, R0_BLOCK]
    exp_score = tl.exp(score_shifted)                    # [XBLOCK, R0_BLOCK]
    row_sum = tl.sum(exp_score, axis=1)                  # [XBLOCK]
    row_sum = row_sum[:, None]                           # [XBLOCK, 1]
    softmax_w = exp_score / row_sum                      # [XBLOCK, R0_BLOCK]

    # Epilogue: add tmp12, scale by 2, sigmoid
    tmp13 = softmax_w + tmp12                            # [XBLOCK, R0_BLOCK]
    tmp14 = tl.full([1, 1], 2.0, tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.sigmoid(tmp15)                            # [XBLOCK, R0_BLOCK]

    # Store with transposed output layout
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
