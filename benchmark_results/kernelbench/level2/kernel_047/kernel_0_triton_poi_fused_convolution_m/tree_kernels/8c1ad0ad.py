import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 128}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 256}, num_warps=4, num_stages=1),
    triton.Config({'XBLOCK': 512}, num_warps=8, num_stages=1),
    triton.Config({'XBLOCK': 1024}, num_warps=8, num_stages=1),
], key=['xnumel'], restore_value=['in_out_ptr0'])
@triton.jit
def triton_poi_fused_convolution_mish_tanh_0(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 118087680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 115320) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 20.0, tl.float32)
    tmp4 = tmp2 > tmp3
    tmp5 = tl.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = libdevice.tanh(tmp9)
    tl.store(in_out_ptr0 + (x3), tmp10, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 118087680
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_mish_tanh_0[grid](arg_0, arg_1, 118087680)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((16, 64, 30, 62, 62), (7380480, 115320, 3844, 62, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
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
