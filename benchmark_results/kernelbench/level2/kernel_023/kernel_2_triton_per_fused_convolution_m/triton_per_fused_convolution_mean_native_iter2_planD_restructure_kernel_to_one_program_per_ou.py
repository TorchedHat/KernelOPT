import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused_convolution_mean_native_group_norm_2(in_ptr0, out_ptr0, xnumel, r0_numel, XBLOCK: tl.constexpr):
    R0_BLOCK: tl.constexpr = 256
    xindex = tl.program_id(0)
    x0 = xindex % 6
    x1 = xindex // 6
    r0_index = tl.arange(0, R0_BLOCK)
    r0_mask = r0_index < 180
    tmp0 = tl.load(in_ptr0 + (r0_index + 180 * x0 + 1088 * x1), mask=r0_mask, other=0.0)
    tmp1 = tl.sum(tmp0, axis=0).to(tl.float32)
    tl.store(out_ptr0 + xindex, tmp1)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 768
    r0_numel = 180
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused_convolution_mean_native_group_norm_2[grid](arg_0, arg_1, 768, 180, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 6, 180), (1088, 180, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128, 6), (6, 1), device='cuda:0', dtype=torch.float32)
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
