import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=2),
], key=['xnumel'])
@triton.jit
def triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 1612800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)  # 1D, shape [XBLOCK]
    xmask = xindex < xnumel
    x0 = xindex % 12600   # position within slice, range [0, 12599]
    x1 = xindex // 12600  # batch index, range [0, 127]

    # Pass 1: find running max over r0=16 channels in registers
    running_max = tl.full([XBLOCK], float('-inf'), dtype=tl.float32)
    for r0 in tl.static_range(16):
        # Load bias as block tensor (same index for all threads) with xmask
        bias = tl.load(in_ptr1 + tl.full([XBLOCK], r0, dtype=tl.int32), mask=xmask, other=0.0)
        val = tl.load(in_ptr0 + (x0 + 12600 * r0 + 201600 * x1), mask=xmask, other=float('-inf'))
        val = val + bias
        running_max = tl.maximum(running_max, val)

    # Pass 2: compute sum of exp(val - max) in registers
    running_sum = tl.zeros([XBLOCK], dtype=tl.float32)
    for r0 in tl.static_range(16):
        bias = tl.load(in_ptr1 + tl.full([XBLOCK], r0, dtype=tl.int32), mask=xmask, other=0.0)
        val = tl.load(in_ptr0 + (x0 + 12600 * r0 + 201600 * x1), mask=xmask, other=float('-inf'))
        val = val + bias
        running_sum = running_sum + tl.exp(val - running_max)

    # Pass 3: store normalized softmax -- NOTE output stride is 12608, NOT 12600
    for r0 in tl.static_range(16):
        bias = tl.load(in_ptr1 + tl.full([XBLOCK], r0, dtype=tl.int32), mask=xmask, other=0.0)
        val = tl.load(in_ptr0 + (x0 + 12600 * r0 + 201600 * x1), mask=xmask, other=float('-inf'))
        val = val + bias
        out_val = tl.exp(val - running_max) / running_sum
        tl.store(out_ptr2 + (x0 + 12608 * r0 + 201728 * x1), out_val, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 1612800
    r0_numel = 16
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0[grid](arg_0, arg_1, arg_2, 1612800, 16)
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
