import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=2),
        triton.Config({'XBLOCK': 1024}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 512},  num_warps=4,  num_stages=2),
        triton.Config({'XBLOCK': 256},  num_warps=4,  num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_addmm_gelu_view_36(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = (xindex % 1536)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp2 = tmp0 + tmp1
    tmp_erf_arg = libdevice.fma(tmp2, 0.7071067811865476, 0.0)
    tmp_erf = libdevice.erf(tmp_erf_arg)
    tmp_half = libdevice.fma(tmp2, 0.5, 0.0)
    tmp_result = libdevice.fma(tmp_half, tmp_erf, tmp_half)
    tl.store(in_out_ptr0 + (x2), tmp_result, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 3010560
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_addmm_gelu_view_36[grid](arg_0, arg_1, 3010560)
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
