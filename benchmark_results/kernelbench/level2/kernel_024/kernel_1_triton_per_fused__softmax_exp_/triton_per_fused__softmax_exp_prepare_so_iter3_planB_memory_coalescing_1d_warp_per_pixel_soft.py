import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 128}, num_warps=1, num_stages=1),
    triton.Config({'XBLOCK': 256}, num_warps=1, num_stages=1),
    triton.Config({'XBLOCK': 512}, num_warps=1, num_stages=1),
    triton.Config({'XBLOCK': 1024}, num_warps=1, num_stages=1),
], key=['xnumel'], restore_value=['in_out_ptr0'])
@triton.jit
def triton_per_fused__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    R0_BLOCK: tl.constexpr = 32
    pid = tl.program_id(0)
    x0 = pid % 900
    x1 = pid // 900
    r0 = tl.arange(0, R0_BLOCK)
    r0_mask = r0 < 24
    ptrs = in_out_ptr0 + x0 + 900 * r0 + 21600 * x1
    vals = tl.load(ptrs, mask=r0_mask, other=float('-inf'))
    row_max = tl.max(vals, axis=0)
    vals_shifted = tl.where(r0_mask, vals - row_max, float('-inf'))
    exp_vals = tl.exp(vals_shifted)
    exp_vals_masked = tl.where(r0_mask, exp_vals, 0.0)
    row_sum = tl.sum(exp_vals_masked, axis=0)
    softmax_vals = tl.where(r0_mask, exp_vals_masked / row_sum, 0.0)
    tl.store(ptrs, softmax_vals, mask=r0_mask)

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
