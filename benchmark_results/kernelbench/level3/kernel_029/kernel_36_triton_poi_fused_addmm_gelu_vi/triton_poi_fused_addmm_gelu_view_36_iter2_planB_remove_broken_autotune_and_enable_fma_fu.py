import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_gelu_view_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    x0 = (xindex % 1536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tmp0 + tmp1
    inv_sqrt2 = tl.full([1], 0.7071067811865476, tl.float32)
    half = tl.full([1], 0.5, tl.float32)
    erf_val = libdevice.erf(tmp2 * inv_sqrt2)
    gelu_out = tmp2 * tl.math.fma(half, erf_val, half)
    tl.store(in_out_ptr0 + (x2), gelu_out, None)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 3010560
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_gelu_view_36[grid](arg_0, arg_1, 3010560, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 196, 1536), (301056, 1536, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((1536,), (1,), device='cuda:0', dtype=torch.float32)
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
