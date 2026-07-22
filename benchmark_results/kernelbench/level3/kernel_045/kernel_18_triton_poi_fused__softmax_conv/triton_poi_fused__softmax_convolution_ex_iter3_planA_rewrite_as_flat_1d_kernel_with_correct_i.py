import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_18(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat 1D kernel: treat YBLOCK as BLOCK_SIZE, ynumel as total numel
    pid = tl.program_id(0)
    offsets = pid * YBLOCK + tl.arange(0, YBLOCK)
    mask = offsets < ynumel
    # Recover (y3, x2) from flat index: flat = x2 + 9*y3
    y3 = offsets // 9
    x2 = offsets % 9
    # Recover sub-indices for output: y0 = y3 % 512, y1 = y3 // 512
    y0 = y3 % 512
    y1 = y3 // 512
    # Input stride layout (4608,9,3,1): address = x2 + 9*y3
    tmp0 = tl.load(in_ptr0 + (x2 + 9 * y3), mask, eviction_policy='evict_last')
    # Output permuted stride (4608,1,1536,512): address = y0 + 512*x2 + 4608*y1
    tl.store(out_ptr0 + (y0 + 512 * x2 + 4608 * y1), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 262144
    xnumel = 9
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__softmax_convolution_exp_sub_18[grid](arg_0, arg_1, 262144, 9, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda:0', dtype=torch.float32)
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
