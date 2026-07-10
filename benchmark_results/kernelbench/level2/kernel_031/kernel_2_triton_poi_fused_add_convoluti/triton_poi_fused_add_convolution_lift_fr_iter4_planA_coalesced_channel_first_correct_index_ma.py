import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_convolution_lift_fresh_minimum_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    xnumel = 128
    ynumel = 2031744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    c = xindex
    hw = yindex % 15876
    b = yindex // 15876
    hw_row = hw // 126
    hw_col = hw % 126
    ptr0_addr = c + hw_col * 128 + hw_row * 16128 + b * 2032128
    tmp0 = tl.load(in_ptr0 + ptr0_addr, mask=ymask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr1 + c, None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + c, None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0.5, tl.float32)
    tmp4 = tl.minimum(tmp2, tmp3)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 2.0, tl.float32)
    tmp8 = tmp6 * tmp7
    out_addr = hw_col + hw_row * 126 + c * 15876 + b * 2032128
    tl.store(out_ptr0 + out_addr, tmp8, mask=ymask, eviction_policy='evict_first')

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 16384
    xnumel = 15876
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_add_convolution_lift_fresh_minimum_mul_2[grid](arg_0, arg_1, arg_2, arg_3, 16384, 15876, YBLOCK=8, XBLOCK=128)
    return arg_3

def get_inputs():
    arg_0 = rand_strided((128, 128, 126, 126), (2032128, 1, 16128, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 1, 1), (1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg_3 = rand_strided((128, 128, 126, 126), (2032128, 15876, 126, 1), device='cuda:0', dtype=torch.float32)
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
