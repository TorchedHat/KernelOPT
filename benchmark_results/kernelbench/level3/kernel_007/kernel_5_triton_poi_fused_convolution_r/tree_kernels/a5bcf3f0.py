import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 191102976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = tl.max_contiguous(tl.multiple_of(xindex, XBLOCK), XBLOCK)
    x0 = (xindex % 256)
    tmp0 = tl.load(in_out_ptr0 + (x2), None, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, None, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 191102976
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_relu_5[grid](arg_0, arg_1, 191102976, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 256, 27, 27), (186624, 1, 6912, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
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
