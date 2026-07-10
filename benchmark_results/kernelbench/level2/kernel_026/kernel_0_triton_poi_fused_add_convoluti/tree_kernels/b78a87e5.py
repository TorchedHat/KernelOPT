import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_convolution_hardswish_mul_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x3 = xindex
    x1_scalar = (xoffset // 32768) % 64  # scalar channel index for this block
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + x1_scalar, None)  # scalar load, broadcasts automatically
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    # hardswish(x) = x * clamp(x+3, 0, 6) / 6 = x^2 * clamp(x+3,0,6) / 6
    clamped = tl.minimum(tl.maximum(tmp4 + 3.0, 0.0), 6.0)
    scaled_clamp = clamped * 0.16666666666666666
    # FMA: libdevice.fma(a,b,c) = a*b+c in one hardware instruction
    # Fuses the two final multiplies: (tmp4*tmp4)*scaled_clamp + 0.0
    tmp14 = libdevice.fma(tmp4 * tmp4, scaled_clamp, 0.0)
    tl.store(in_out_ptr0 + (x3), tmp14, None)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 268435456
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_convolution_hardswish_mul_0[grid](arg_0, arg_1, arg_2, 268435456, XBLOCK=128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 64, 32, 32, 32), (2097152, 32768, 1024, 32, 1), device='cuda:0', dtype=torch.float32)
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
