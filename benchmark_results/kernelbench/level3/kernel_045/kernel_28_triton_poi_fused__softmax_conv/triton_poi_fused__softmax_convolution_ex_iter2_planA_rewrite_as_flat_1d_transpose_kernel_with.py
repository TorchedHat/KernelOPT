import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused__softmax_convolution_exp_sub_28(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat 1D indexing: treat XBLOCK as BLOCK_SIZE, use only pid(0)
    # numel = 524288, XBLOCK = 256, grid = (2048,)
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    numel = 524288
    mask = offsets < numel
    # Decompose flat index into (y3, x2): xnumel=4, ynumel=131072
    x2 = offsets % 4
    y3 = offsets // 4
    # Decompose y3 into spatial dims for output index
    y0 = y3 % 256
    y1 = y3 // 256
    # Load: in_ptr0 stride is (1024,4,2,1) so x2+4*y3 = offsets (contiguous read)
    tmp0 = tl.load(in_ptr0 + offsets, mask, eviction_policy='evict_last')
    # Store: out_ptr0 stride is (1024,1,512,256) -> permuted layout
    tl.store(out_ptr0 + (y0 + 256*x2 + 1024*y1), tmp0, mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 131072
    xnumel = 4
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused__softmax_convolution_exp_sub_28[grid](arg_0, arg_1, 131072, 4, YBLOCK=8, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((512, 256, 2, 2), (1024, 4, 2, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((512, 256, 2, 2), (1024, 1, 512, 256), device='cuda:0', dtype=torch.float32)
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
