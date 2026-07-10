import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers

@triton.jit
def triton_poi_fused_convolution_div_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 110215168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x4 = xindex
    x2 = ((xindex // 53816) % 16)
    x0 = (xindex % 3844)
    x5 = xindex // 3844
    tmp0 = tl.load(in_ptr0 + (x4), None)
    tmp1 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tl.store(out_ptr0 + (x0 + 3872*x5), tmp4, None)



def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 110215168
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_div_0[grid](arg_0, arg_1, arg_2, 110215168, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 16, 14, 62, 62), (861056, 53816, 3844, 62, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((16,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 16, 14, 62, 62), (867328, 54208, 3872, 62, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2]

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
