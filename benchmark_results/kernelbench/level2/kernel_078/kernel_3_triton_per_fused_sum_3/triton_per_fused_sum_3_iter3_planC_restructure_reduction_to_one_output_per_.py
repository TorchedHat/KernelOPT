import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1}, num_warps=1, num_stages=1),
        triton.Config({'XBLOCK': 1}, num_warps=2, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_per_fused_sum_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    xnumel = 16000
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    x3 = tl.program_id(0)  # scalar: one program per output element
    xmask = x3 < xnumel
    x0 = x3 % 1000  # spatial position within flattened (10,10,10)
    x1 = x3 // 1000  # batch index 0..15
    r0_index = tl.arange(0, R0_BLOCK)  # shape [64], 1D
    tmp0 = tl.load(in_ptr0 + (x0 + 1000*r0_index + 64000*x1), mask=xmask, other=0.0)  # [64]
    tmp4 = tl.sum(tmp0, 0).to(tl.float32)  # reduce axis=0, scalar result
    tl.store(out_ptr0 + x3, tmp4, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 16000
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_sum_3[grid](arg_0, arg_1, 16000, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 64, 10, 10, 10), (64000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16, 1, 10, 10, 10), (1000, 1000, 100, 10, 1), device='cuda:0', dtype=torch.float32)
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
