import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 16},  num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
    restore_value=['in_out_ptr0'],
)
@triton.jit
def triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 262144
    r0_numel = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)  # 1D [XBLOCK]
    xmask = xindex < xnumel
    x0 = xindex % 16384
    x1 = xindex // 16384

    # Pass 1: online softmax — compute running_max and running_sum
    running_max = tl.full([XBLOCK], float('-inf'), tl.float32)
    running_sum = tl.zeros([XBLOCK], tl.float32)
    for r in tl.static_range(64):
        # Load bias scalar for this r position; broadcast to [XBLOCK] via addition
        bias_val = tl.load(in_ptr0 + r, mask=tl.full([1], True, tl.int1), other=0.0, eviction_policy='evict_last')
        ptr = in_out_ptr0 + (x0 + 16384 * r + 1048576 * x1)
        val = tl.load(ptr, mask=xmask, other=0.0)
        val = val / 32.0 + bias_val
        new_max = tl.maximum(running_max, val)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.exp(val - new_max)
        running_max = new_max

    # Pass 2: compute softmax output and apply tanh*2
    for r in tl.static_range(64):
        bias_val = tl.load(in_ptr0 + r, mask=tl.full([1], True, tl.int1), other=0.0, eviction_policy='evict_last')
        ptr = in_out_ptr0 + (x0 + 16384 * r + 1048576 * x1)
        val = tl.load(ptr, mask=xmask, other=0.0)
        val = val / 32.0 + bias_val
        softmax_val = tl.exp(val - running_max) / running_sum
        out = libdevice.tanh(softmax_val) * 2.0
        tl.store(ptr, out, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 262144
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__softmax_add_convolution_exp_mean_mul_prepare_softmax_online_sub_tanh_1[grid](arg_0, arg_1, 262144, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 64, 1, 128, 128), (1048576, 16384, 16384, 128, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1, 64, 1, 1, 1), (64, 1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1]

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
