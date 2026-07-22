import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_per_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_13(in_out_ptr0, in_ptr0, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 2048
    r0_numel = 7
    R0_BLOCK: tl.constexpr = 16
    x3 = tl.program_id(0)
    x0 = x3 % 256
    x1 = x3 // 256
    r0_index = tl.arange(0, R0_BLOCK)
    r0_mask = r0_index < r0_numel
    tmp0 = tl.load(in_ptr0 + (x0 + 256*r0_index + 1792*x1), mask=r0_mask, other=0.0)
    tmp3 = tl.where(r0_mask, tmp0, 0.0)
    tmp4 = tl.sum(tmp3, 0).to(tl.float32)
    tmp6 = tmp4 / 784.0
    tl.store(in_out_ptr0 + x3, tmp6)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 2048
    r0_numel = 7
    grid = (triton.cdiv(xnumel, 128),)
    triton_per_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_mean_relu_13[grid](arg_0, arg_1, 2048, 7, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((8, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((8, 256, 7), (1792, 1, 256), device='cuda:0', dtype=torch.float32)
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
