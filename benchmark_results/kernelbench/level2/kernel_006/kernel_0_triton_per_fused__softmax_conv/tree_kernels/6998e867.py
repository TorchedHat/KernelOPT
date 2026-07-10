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
    R0_BLOCK: tl.constexpr = 16
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)  # 1D [XBLOCK] -- NOT [:, None]
    xmask = xindex < xnumel
    x0 = xindex % 12600
    x1 = xindex // 12600

    # Pass 1: find max over 16 channels in registers (no shared memory)
    running_max = tl.full([XBLOCK], float('-inf'), dtype=tl.float32)
    for r in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r + 201600*x1), mask=xmask, other=float('-inf'))
        # Load bias as a block by broadcasting scalar index r to [XBLOCK] offsets
        bias = tl.load(in_ptr1 + (tl.arange(0, XBLOCK) * 0 + r), mask=xmask, other=0.0)
        val = val + bias
        running_max = tl.maximum(running_max, val)  # elementwise max, no shared memory

    # Pass 2: accumulate exp-sum in registers
    exp_sum = tl.zeros([XBLOCK], dtype=tl.float32)
    for r in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r + 201600*x1), mask=xmask, other=float('-inf'))
        bias = tl.load(in_ptr1 + (tl.arange(0, XBLOCK) * 0 + r), mask=xmask, other=0.0)
        val = val + bias
        exp_val = tl.exp(val - running_max)
        exp_sum = exp_sum + tl.where(xmask, exp_val, 0.0)

    # Pass 3: write normalized softmax values to output
    for r in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r + 201600*x1), mask=xmask, other=float('-inf'))
        bias = tl.load(in_ptr1 + (tl.arange(0, XBLOCK) * 0 + r), mask=xmask, other=0.0)
        val = val + bias
        out_val = tl.exp(val - running_max) / exp_sum
        # Preserve output stride 12608 exactly
        tl.store(out_ptr2 + (x0 + 12608*r + 201728*x1),
                 tl.where(xmask, out_val, 0.0), mask=xmask)

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
