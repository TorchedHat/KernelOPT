import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_225(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flatten the 2D iteration space into a 1D index for coalesced loads
    # XBLOCK is used as BLOCK (512), YBLOCK is unused (set to 1 in launch)
    pid = tl.program_id(0)
    flat_index = pid * XBLOCK + tl.arange(0, XBLOCK)
    numel = 516096  # 9 * 57344
    mask = flat_index < numel
    x2 = flat_index % 9
    y3 = flat_index // 9
    y0 = y3 % 1792
    y1 = y3 // 1792
    tmp0 = tl.load(in_ptr0 + flat_index, mask=mask)
    tl.store(out_ptr0 + (y0 + 1792*x2 + 16128*y1), tmp0, mask=mask)

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
