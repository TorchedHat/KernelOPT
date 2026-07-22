import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 1024}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 2048}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 2048}, num_warps=16, num_stages=1),
        triton.Config({'XBLOCK': 4096}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 4096}, num_warps=16, num_stages=1),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_add_addmm_mul_pow_tanh_view_12(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3072
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + x2, xmask)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.5, tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp2 * tmp2
    tmp6 = tmp5 * tmp2
    tmp7 = tl.full([1], 0.044715, tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = tmp2 + tmp8
    tmp10 = tl.full([1], 0.7978845608028654, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = libdevice.tanh(tmp11)
    tmp13 = tl.full([1], 1.0, tl.float32)
    tmp14 = tmp12 + tmp13
    tmp15 = tmp4 * tmp14
    tl.store(in_out_ptr0 + x2, tmp15, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 201326592
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_addmm_mul_pow_tanh_view_12[grid](arg_0, arg_1, 201326592)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((128, 512, 3072), (1572864, 3072, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((3072,), (1,), device='cuda:0', dtype=torch.float32)
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
