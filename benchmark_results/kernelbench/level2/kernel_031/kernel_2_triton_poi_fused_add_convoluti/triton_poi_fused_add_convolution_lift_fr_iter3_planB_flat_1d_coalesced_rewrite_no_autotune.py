import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_add_convolution_lift_fresh_minimum_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D kernel: treat the entire output as a 1D array of numel=259981312 elements
    # XBLOCK is used as BLOCK_SIZE (set to 1024 at launch)
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    numel = 259981312
    mask = offsets < numel
    # Decompose flat output index into (n, c, h, w)
    # out_ptr0 strides: (2032128, 15876, 126, 1)
    w        = offsets % 126
    tmp_hw   = offsets // 126
    h        = tmp_hw % 126
    tmp_nchw = tmp_hw // 126
    c        = tmp_nchw % 128
    n        = tmp_nchw // 128
    # in_ptr0 strides: (2032128, 1, 16128, 128)
    in0_idx = n * 2032128 + c + h * 16128 + w * 128
    tmp0 = tl.load(in_ptr0 + in0_idx, mask=mask)
    tmp1 = tl.load(in_ptr1 + c, mask=mask)
    tmp5 = tl.load(in_ptr2 + c, mask=mask)
    tmp2 = tmp0 + tmp1
    tmp4 = tl.minimum(tmp2, tl.full(tmp2.shape, 0.5, tl.float32))
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 * tl.full(tmp6.shape, 2.0, tl.float32)
    tl.store(out_ptr0 + offsets, tmp8, mask=mask)

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
