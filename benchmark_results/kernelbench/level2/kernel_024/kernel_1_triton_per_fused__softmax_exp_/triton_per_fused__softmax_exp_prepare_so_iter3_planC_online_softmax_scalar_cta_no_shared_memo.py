import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    # Scalar-per-CTA online softmax: each program handles one (batch, spatial) position
    # Grid must be (115200,) = 128*900, one CTA per position
    pid = tl.program_id(0)  # in [0, 115200)
    spatial = pid % 900     # stride=1 in memory
    batch   = pid // 900    # stride=21600 in memory
    base    = spatial + 21600 * batch

    # Online softmax pass 1: compute running max and sum in one pass
    # Python range(24) — compiler unrolls into registers, no shared memory
    running_max = float('-inf')
    running_sum = 0.0
    for h in range(24):
        val = tl.load(in_out_ptr0 + base + 900 * h)
        new_max = tl.maximum(running_max, val)
        running_sum = running_sum * tl.exp(running_max - new_max) + tl.exp(val - new_max)
        running_max = new_max

    # Online softmax pass 2: write normalized values
    for h in range(24):
        val = tl.load(in_out_ptr0 + base + 900 * h)
        result = tl.exp(val - running_max) / running_sum
        tl.store(in_out_ptr0 + base + 900 * h, result)

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
