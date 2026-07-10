import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 1612800
    r0_numel = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)  # [XBLOCK] 1D
    xmask = xindex < xnumel
    x0 = xindex % 12600   # spatial position (stride-1 in output)
    x1 = xindex // 12600  # batch index

    # Pass 1: find max over r0=0..15 in registers (no shared memory)
    running_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    for r0 in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r0 + 201600*x1), mask=xmask, other=float('-inf'))
        bias = tl.load(in_ptr1 + r0, None)  # scalar load, no mask (always valid, r0 in 0..15)
        val = val + bias
        running_max = tl.maximum(running_max, val)

    # Pass 2: compute sum of exp(val - max) in registers
    running_sum = tl.zeros([XBLOCK], tl.float32)
    for r0 in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r0 + 201600*x1), mask=xmask, other=float('-inf'))
        bias = tl.load(in_ptr1 + r0, None)  # scalar load, no mask (always valid, r0 in 0..15)
        val = val + bias
        running_sum = running_sum + tl.exp(val - running_max)

    # Pass 3: store normalized values COALESCED
    # Consecutive threads write consecutive x0 positions (stride-1)
    # r0 is a compile-time constant per iteration: zero bank conflicts
    for r0 in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r0 + 201600*x1), mask=xmask, other=float('-inf'))
        bias = tl.load(in_ptr1 + r0, None)  # scalar load, no mask (always valid, r0 in 0..15)
        val = val + bias
        softmax_val = tl.exp(val - running_max) / running_sum
        # Output stride is 12608 (padded), input stride is 12600 — strictly separate
        tl.store(out_ptr2 + (x0 + 12608*r0 + 201728*x1), softmax_val, mask=xmask)

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
