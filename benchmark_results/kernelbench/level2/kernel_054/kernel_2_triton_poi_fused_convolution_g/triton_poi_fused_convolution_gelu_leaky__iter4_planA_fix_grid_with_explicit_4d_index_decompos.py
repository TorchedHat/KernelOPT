import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_gelu_leaky_relu_mul_2(
        in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel,
        YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D iteration over OUTPUT linear index (NCHW contiguous)
    # numel = 64 * 64 * 254 * 254 = 264,257,536
    numel = 264257536
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < numel

    # Decompose flat output index -> (b, c, h, w)
    # Output strides: (4129024, 64516, 254, 1) => NCHW, 254x254 spatial
    w   = offsets % 254
    tmp = offsets // 254
    h   = tmp % 254
    tmp = tmp // 254
    c   = tmp % 64
    b   = tmp // 64

    # Compute input address using input strides (4129024, 1, 16256, 64)
    # input[b, c, h, w] = b*4129024 + c*1 + h*16256 + w*64
    in_idx = b * 4129024 + h * 16256 + w * 64 + c

    tmp0 = tl.load(in_ptr0 + in_idx, mask=mask)   # conv output
    tmp1 = tl.load(in_ptr1 + c,      mask=mask)   # bias (64,)
    tmp3 = tl.load(in_ptr2 + c,      mask=mask)   # scale (64,1,1)

    tmp2 = tmp0 + tmp1          # add bias
    tmp4 = tmp2 * tmp3          # multiply scale
    # leaky_relu
    tmp6 = tmp4 > 0.0
    tmp8 = tmp4 * 0.01
    tmp9 = tl.where(tmp6, tmp4, tmp8)
    # gelu
    tmp11 = tmp9 * 0.5
    tmp13 = tmp9 * 0.7071067811865476
    tmp14 = libdevice.erf(tmp13)
    tmp16 = tmp14 + 1.0
    tmp17 = tmp11 * tmp16

    tl.store(out_ptr0 + offsets, tmp17, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 4129024
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_gelu_leaky_relu_mul_2[grid](arg_0, arg_1, arg_2, arg_3, 4129024, 64)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((64, 64, 254, 254), (4129024, 1, 16256, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((64, 64, 254, 254), (4129024, 64516, 254, 1), device='cuda:0', dtype=torch.float32)
    return [arg_0, arg_1, arg_2, arg_3]

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
