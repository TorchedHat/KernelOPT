import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Input:  (32,128,126,250) strides (4032000,1,32000,128)  [channels-last]
    # Output: (32,128,126,250) strides (4032000,31500,250,1)  [contiguous NCHW]
    # Total elements: 32*128*126*250 = 129,024,000
    # Flat 1D approach: each thread handles one output element
    # YBLOCK is kept in signature but unused (set to 1 at launch)
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    numel = ynumel * xnumel
    mask = offsets < numel

    # Decompose flat OUTPUT index into (n, c, h, w)
    # Output strides: W=1, H=250, C=31500, N=4032000
    w = offsets % 250
    tmp1 = offsets // 250
    h = tmp1 % 126
    tmp2 = tmp1 // 126
    c = tmp2 % 128
    n = tmp2 // 128

    # Compute INPUT address using channels-last strides: C=1, W=128, H=32000, N=4032000
    in_addr = n * 4032000 + c + h * 32000 + w * 128

    tmp0 = tl.load(in_ptr0 + in_addr, mask=mask)
    # Store to flat output offset — consecutive threads write consecutive addresses
    tl.store(out_ptr0 + offsets, tmp0, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 4096
    xnumel = 31500
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_1[grid](arg_0, arg_1, 4096, 31500, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((32, 128, 126, 250), (4032000, 1, 32000, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32, 128, 126, 250), (4032000, 31500, 250, 1), device='cuda:0', dtype=torch.float32)
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
