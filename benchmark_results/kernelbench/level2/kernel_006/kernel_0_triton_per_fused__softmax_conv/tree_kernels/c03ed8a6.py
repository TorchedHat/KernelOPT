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
    xindex = xoffset + tl.arange(0, XBLOCK)  # 1D [XBLOCK]
    xmask = xindex < xnumel
    x0 = xindex % 12600   # spatial position (stride-1 in output)
    x1 = xindex // 12600  # batch index

    # Pre-load all 16 bias values as scalars ONCE - only 64 bytes total
    bias0  = tl.load(in_ptr1 + 0,  mask=None, eviction_policy='evict_last')
    bias1  = tl.load(in_ptr1 + 1,  mask=None, eviction_policy='evict_last')
    bias2  = tl.load(in_ptr1 + 2,  mask=None, eviction_policy='evict_last')
    bias3  = tl.load(in_ptr1 + 3,  mask=None, eviction_policy='evict_last')
    bias4  = tl.load(in_ptr1 + 4,  mask=None, eviction_policy='evict_last')
    bias5  = tl.load(in_ptr1 + 5,  mask=None, eviction_policy='evict_last')
    bias6  = tl.load(in_ptr1 + 6,  mask=None, eviction_policy='evict_last')
    bias7  = tl.load(in_ptr1 + 7,  mask=None, eviction_policy='evict_last')
    bias8  = tl.load(in_ptr1 + 8,  mask=None, eviction_policy='evict_last')
    bias9  = tl.load(in_ptr1 + 9,  mask=None, eviction_policy='evict_last')
    bias10 = tl.load(in_ptr1 + 10, mask=None, eviction_policy='evict_last')
    bias11 = tl.load(in_ptr1 + 11, mask=None, eviction_policy='evict_last')
    bias12 = tl.load(in_ptr1 + 12, mask=None, eviction_policy='evict_last')
    bias13 = tl.load(in_ptr1 + 13, mask=None, eviction_policy='evict_last')
    bias14 = tl.load(in_ptr1 + 14, mask=None, eviction_policy='evict_last')
    bias15 = tl.load(in_ptr1 + 15, mask=None, eviction_policy='evict_last')
    bias_cache = [bias0, bias1, bias2, bias3, bias4, bias5, bias6, bias7,
                  bias8, bias9, bias10, bias11, bias12, bias13, bias14, bias15]

    # Pass 1: find max over r0=0..15 in registers (no shared memory)
    running_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    for r0 in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r0 + 201600*x1), mask=xmask, other=float('-inf'))
        val = val + bias_cache[r0]
        running_max = tl.maximum(running_max, val)

    # Pass 2: compute sum of exp(val - max) in registers
    exp_sum = tl.zeros([XBLOCK], tl.float32)
    for r0 in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r0 + 201600*x1), mask=xmask, other=float('-inf'))
        val = val + bias_cache[r0]
        exp_val = tl.exp(val - running_max)
        exp_sum = exp_sum + tl.where(xmask, exp_val, 0.0)

    # Pass 3: write normalized output
    for r0 in tl.static_range(16):
        val = tl.load(in_ptr0 + (x0 + 12600*r0 + 201600*x1), mask=xmask, other=float('-inf'))
        val = val + bias_cache[r0]
        out_val = tl.exp(val - running_max) / exp_sum
        tl.store(out_ptr2 + (x0 + 12608*r0 + 201728*x1), tl.where(xmask, out_val, 0.0), mask=xmask)

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
