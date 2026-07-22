import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_gelu_14(in_out_ptr0, in_ptr0, xnumel, XBLOCK: tl.constexpr):
    xindex = tl.arange(0, XBLOCK)[:]
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + x0, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + xindex)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = libdevice.erf(tmp6)
    tmp8 = tl.full([1], 1.0, tl.float32)
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tl.store(in_out_ptr0 + xindex, tmp10)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 4096
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_gelu_14[grid](arg_0, arg_1, 4096, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((2, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((2048,), (1,), device='cuda:0', dtype=torch.float32)
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
