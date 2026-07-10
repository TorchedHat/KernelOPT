import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 512, 'YBLOCK': 1}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8, num_stages=2),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_gelu_leaky_relu_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D kernel over all output elements (xnumel = total numel = 264257536)
    # ynumel and YBLOCK are unused; XBLOCK is the tile size
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < xnumel

    # Decompose flat output index into (n, c, h, w) using output strides (4129024, 64516, 254, 1)
    n = offsets // 4129024
    rem = offsets % 4129024
    c = rem // 64516
    rem2 = rem % 64516
    h = rem2 // 254
    w = rem2 % 254

    # Compute input flat index using input strides (4129024, 1, 16256, 64)
    in_idx = n * 4129024 + c * 1 + h * 16256 + w * 64

    tmp0 = tl.load(in_ptr0 + in_idx, mask=mask)
    tmp1 = tl.load(in_ptr1 + c, mask=mask)
    tmp3 = tl.load(in_ptr2 + c, mask=mask)

    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3

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

    # Output is contiguous with strides (4129024, 64516, 254, 1), so flat index = offsets
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
