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
        triton.Config({'XBLOCK': 1}, num_warps=4, num_stages=1),
    ],
    key=['xnumel', 'r0_numel'],
)
@triton.jit
def triton_per_fused_min_native_group_norm_3(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    R0_BLOCK: tl.constexpr = 64
    # Each program handles exactly one x-row (XBLOCK=1 from autotune)
    x0 = tl.program_id(0)
    xmask = x0 < xnumel
    r0_index = tl.arange(0, R0_BLOCK)
    tmp0 = tl.load(in_ptr0 + (r0_index + 64 * x0), mask=xmask, other=float('inf'))
    tmp1 = tl.where(xmask, tmp0, float('inf'))
    result = tl.min(tmp1, axis=0).to(tl.float32)
    tl.store(out_ptr0 + x0, result, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 1024
    r0_numel = 64
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_min_native_group_norm_3[grid](arg_0, arg_1, 1024, 64)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 1, 64), (64, 65536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1024, 1), (1, 1024), device='cuda:0', dtype=torch.float32)
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
