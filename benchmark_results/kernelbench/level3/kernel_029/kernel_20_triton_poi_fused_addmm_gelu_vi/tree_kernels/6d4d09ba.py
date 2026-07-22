import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_addmm_gelu_view_20(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x0 = (xindex % 768)
    x2 = xindex
    # Direct L1TEX load - bias (3KB) stays resident in L1 across all 22+ waves
    tmp0 = tl.load(in_ptr0 + x0, mask=xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + x2, mask=xmask)
    tmp2 = tmp0 + tmp1  # bias + matmul result
    # GELU: output = x * 0.5 * (1 + erf(x * 0.7071067811865476))
    tmp6 = tmp2 * 0.7071067811865476
    tmp7 = libdevice.erf(tmp6)
    tmp9 = tmp7 + 1.0
    # Fuse: result = tmp2 * 0.5 * tmp9 as single FMA
    tmp10 = libdevice.fma(tmp2, 0.5 * tmp9, 0.0)
    tl.store(in_out_ptr0 + x2, tmp10, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 6021120
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_gelu_view_20[grid](arg_0, arg_1, 6021120, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 784, 768), (602112, 768, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((768,), (1,), device='cuda:0', dtype=torch.float32)
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
