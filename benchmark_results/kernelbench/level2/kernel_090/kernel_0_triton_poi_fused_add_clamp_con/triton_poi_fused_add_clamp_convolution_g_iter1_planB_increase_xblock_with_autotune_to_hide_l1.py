import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 256}, num_warps=4),
    triton.Config({'XBLOCK': 512}, num_warps=8),
    triton.Config({'XBLOCK': 1024}, num_warps=16),
    triton.Config({'XBLOCK': 2048}, num_warps=32),
], key=['xnumel'])
@triton.jit
def triton_poi_fused_add_clamp_convolution_gelu_leaky_relu_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 440860672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = ((xindex // 53816) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0.0, tl.float32)
    tmp4 = tmp2 > tmp3
    tmp5 = tl.full([1], 0.2, tl.float32)
    tmp6 = tmp2 * tmp5
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full([1], -1.0, tl.float32)
    tmp11 = tl.maximum(tmp9, tmp10)
    tmp12 = tl.full([1], 1.0, tl.float32)
    tmp13 = tl.minimum(tmp11, tmp12)
    tmp14 = tl.full([1], 0.5, tl.float32)
    tmp15 = tmp13 * tmp14
    tmp16 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp17 = tmp13 * tmp16
    tmp18 = libdevice.erf(tmp17)
    tmp19 = tmp18 + tmp12
    tmp20 = tmp15 * tmp19
    tl.store(in_out_ptr0 + (x3), tmp20, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 440860672
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_clamp_convolution_gelu_leaky_relu_0[grid](arg_0, arg_1, arg_2, 440860672)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 64, 14, 62, 62), (3444224, 53816, 3844, 62, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
