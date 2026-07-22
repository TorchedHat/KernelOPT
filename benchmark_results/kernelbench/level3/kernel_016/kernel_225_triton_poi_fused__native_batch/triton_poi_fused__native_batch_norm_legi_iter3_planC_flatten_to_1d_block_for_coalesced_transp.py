import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D kernel: treat the full (ynumel * xnumel) space as a 1D index
    # XBLOCK is repurposed as BLOCK_SIZE (512), YBLOCK is unused
    # numel = 57344 * 9 = 516096
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    numel = ynumel * xnumel
    mask = offsets < numel

    # Decompose flat index into (y3, x2) -- xnumel=9 is inner dim
    x2 = offsets % xnumel
    y3 = offsets // xnumel

    # Sub-indices for output addressing
    y0 = y3 % 1792
    y1 = y3 // 1792

    # Load: in_ptr0 stride (16128,9,3,1) -> flat = x2 + 9*y3
    tmp0 = tl.load(in_ptr0 + (x2 + xnumel * y3), mask=mask)

    # Store: out_ptr0 stride (16128,1,5376,1792) -> y0 + 1792*x2 + 16128*y1
    tl.store(out_ptr0 + (y0 + 1792 * x2 + 16128 * y1), tmp0, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 57344
    xnumel = 9
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225[grid](arg_0, arg_1, 57344, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 1792, 3, 3), (16128, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 1792, 3, 3), (16128, 1, 5376, 1792), device='cuda:0', dtype=torch.float32)
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
