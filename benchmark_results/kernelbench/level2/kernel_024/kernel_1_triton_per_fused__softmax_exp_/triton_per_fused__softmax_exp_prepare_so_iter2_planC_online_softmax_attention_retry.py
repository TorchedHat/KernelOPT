import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__softmax_exp_prepare_softmax_online_sub_1(in_out_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    # Each program handles exactly ONE row (one spatial position in one batch*head)
    # Grid = xnumel = 115200; XBLOCK = 1 (kept for signature compatibility)
    row_idx = tl.program_id(0)  # in [0, 115200)
    x0 = row_idx % 900          # spatial position (h*w index, stride=1)
    x1 = row_idx // 900         # batch*head index (stride=21600 floats)

    # Pass 1: compute max over 24 heads using running max (register-only, no shared memory)
    m_i = tl.full([], float('-inf'), dtype=tl.float32)
    for r in tl.static_range(0, 24):
        ptr = in_out_ptr0 + (x0 + 900 * r + 21600 * x1)
        val = tl.load(ptr).to(tl.float32)
        m_i = tl.maximum(m_i, val)

    # Pass 2: compute sum of exp(x - max)
    l_i = tl.full([], 0.0, dtype=tl.float32)
    for r in tl.static_range(0, 24):
        ptr = in_out_ptr0 + (x0 + 900 * r + 21600 * x1)
        val = tl.load(ptr).to(tl.float32)
        l_i = l_i + tl.exp(val - m_i)

    # Pass 3: write normalized softmax values back in-place
    for r in tl.static_range(0, 24):
        ptr = in_out_ptr0 + (x0 + 900 * r + 21600 * x1)
        val = tl.load(ptr).to(tl.float32)
        tl.store(ptr, tl.exp(val - m_i) / l_i)

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
