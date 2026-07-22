import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_cat_convolution_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32112640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = xindex
    x0 = (xindex % 64)
    x1 = xindex // 64
    tmp0 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_first')
    tl.store(out_ptr0 + (x0 + 512*x1), tmp0, None)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 32112640
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_cat_convolution_8[grid](arg_0, arg_1, 32112640, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((501760, 64), (64, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 64, 224, 224), (25690112, 1, 114688, 512), device='cuda:0', dtype=torch.float32)
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
