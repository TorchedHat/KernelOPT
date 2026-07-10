import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1612800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)  # 1D shape [XBLOCK]
    xmask = xindex < xnumel
    x0 = xindex % 12600   # spatial position within batch*depth slice
    x1 = xindex // 12600  # batch index

    # Fused Pass 1: compute running max AND running sum via online softmax recurrence
    running_max = tl.full([XBLOCK], float('-inf'), dtype=tl.float32)
    running_sum = tl.zeros([XBLOCK], dtype=tl.float32)
    for r0 in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600 * r0 + 201600 * x1), mask=xmask, other=float('-inf'))
        bias = tl.load(in_ptr1 + r0, None)  # scalar load, r0 always in 0..15
        new_val = val + bias
        new_max = tl.maximum(running_max, new_val)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.exp(new_val - new_max)
        running_max = new_max

    # Pass 2 (store): write normalized softmax output (coalesced stores)
    for r0 in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600 * r0 + 201600 * x1), mask=xmask, other=float('-inf'))
        bias = tl.load(in_ptr1 + r0, None)  # CRITICAL: add bias here too
        val = val + bias
        result = tl.exp(val - running_max) / running_sum
        tl.store(out_ptr2 + (x0 + 12608 * r0 + 201728 * x1), result, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 1612800
    r0_numel = 16
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0[grid](arg_0, arg_1, arg_2, 1612800, 16, XBLOCK=128)
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
