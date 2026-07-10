import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 256}, num_warps=8),
        triton.Config({'XBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 64},  num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused__softmax_convolution_exp_prepare_softmax_online_sub_0(in_ptr0, in_ptr1, out_ptr2, xnumel, r0_numel, XBLOCK: tl.constexpr):
    R0_NUMEL: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)  # 1D: [XBLOCK]
    xmask = xindex < xnumel
    x0 = xindex % 12600
    x1 = xindex // 12600
    # Pass 1: compute running max over r0 in registers
    running_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    for r in tl.static_range(R0_NUMEL):
        # Load bias as block by broadcasting scalar offset; use xmask for consistency
        bias = tl.load(in_ptr1 + r + tl.zeros([XBLOCK], tl.int32), mask=xmask, other=0.0)
        val = tl.load(in_ptr0 + (x0 + 12600*r + 201600*x1), mask=xmask, other=float('-inf'))
        running_max = tl.maximum(running_max, val + bias)
    # Pass 2: compute sum of exp(val + bias - max)
    running_sum = tl.zeros([XBLOCK], tl.float32)
    for r in tl.static_range(R0_NUMEL):
        bias = tl.load(in_ptr1 + r + tl.zeros([XBLOCK], tl.int32), mask=xmask, other=0.0)
        val = tl.load(in_ptr0 + (x0 + 12600*r + 201600*x1), mask=xmask, other=float('-inf'))
        running_sum += tl.exp(val + bias - running_max)
    # Pass 3: normalize and store
    for r in tl.static_range(R0_NUMEL):
        bias = tl.load(in_ptr1 + r + tl.zeros([XBLOCK], tl.int32), mask=xmask, other=0.0)
        val = tl.load(in_ptr0 + (x0 + 12600*r + 201600*x1), mask=xmask, other=float('-inf'))
        out = tl.exp(val + bias - running_max) / running_sum
        tl.store(out_ptr2 + (x0 + 12608*r + 201728*x1), out, mask=xmask)

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
