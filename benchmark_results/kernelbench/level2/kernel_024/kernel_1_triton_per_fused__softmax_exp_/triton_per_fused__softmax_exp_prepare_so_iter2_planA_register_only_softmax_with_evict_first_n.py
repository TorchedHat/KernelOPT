import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 115200
    r0_numel = 24
    R0_BLOCK: tl.constexpr = 32
    # Each CTA handles exactly one (batch, head, spatial) row
    x_id = tl.program_id(0)      # in [0, 115200)
    x0 = x_id % 900              # spatial position (stride=1 in memory)
    x1 = x_id // 900             # batch*head index
    # 1D register vector over the 24 reduction elements
    r0_index = tl.arange(0, R0_BLOCK)          # shape [32]
    r0_mask = r0_index < r0_numel              # True for indices 0..23
    ptrs = in_out_ptr0 + (x0 + 900 * r0_index + 21600 * x1)
    # Load with evict_first: streaming data, each element read once
    tmp0 = tl.load(ptrs, mask=r0_mask, other=float('-inf'),
                   eviction_policy='evict_first')
    # Softmax entirely in registers -- NO shared memory staging
    row_max = tl.max(tmp0, axis=0)             # scalar
    tmp_shifted = tl.where(r0_mask, tmp0 - row_max, float('-inf'))
    tmp_exp = tl.exp(tmp_shifted)
    tmp_exp_masked = tl.where(r0_mask, tmp_exp, 0.0)
    row_sum = tl.sum(tmp_exp_masked, axis=0)   # scalar
    tmp_out = tl.exp(tl.where(r0_mask, tmp0 - row_max, float('-inf'))) / row_sum
    # Store with evict_first: written once, no reuse
    tl.store(ptrs, tmp_out, mask=r0_mask,
             eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    xnumel = 115200
    r0_numel = 24
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_exp_prepare_softmax_online_sub_1[grid](arg_0, 115200, 24, XBLOCK=128)
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
