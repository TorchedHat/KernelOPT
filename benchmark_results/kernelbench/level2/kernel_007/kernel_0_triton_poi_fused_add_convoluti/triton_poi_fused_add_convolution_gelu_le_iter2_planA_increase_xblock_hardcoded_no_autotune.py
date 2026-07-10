import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 236175360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1 = ((xindex // 115320) % 32)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tmp5 = tl.full([1], 0.0, tl.float32)
    tmp6 = tmp4 > tmp5
    tmp7 = tl.full([1], 0.01, tl.float32)
    tmp8 = tmp4 * tmp7
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    tmp10 = tl.full([1], 0.5, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full([1], 0.7071067811865476, tl.float32)
    tmp13 = tmp9 * tmp12
    tmp14 = libdevice.erf(tmp13)
    tmp15 = tl.full([1], 1.0, tl.float32)
    tmp16 = tmp14 + tmp15
    tmp17 = tmp11 * tmp16
    tmp18 = tl.sigmoid(tmp17)
    tmp20 = tmp18 + tmp19
    tl.store(in_out_ptr0 + (x3), tmp20, None)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 236175360
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_convolution_gelu_leaky_relu_relu_sigmoid_0[grid](arg_0, arg_1, arg_2, 236175360, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((64, 32, 30, 62, 62), (3690240, 115320, 3844, 62, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((32,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
