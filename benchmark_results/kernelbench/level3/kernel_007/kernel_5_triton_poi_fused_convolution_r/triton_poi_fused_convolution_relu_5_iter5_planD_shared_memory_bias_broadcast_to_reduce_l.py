import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.jit
def triton_poi_fused_convolution_relu_5(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 191102976
    # Stage 1: load entire 256-element bias into register array (one vectorized load per block)
    bias_mask = tl.full([256], True, tl.int1)
    bias_vec = tl.load(in_ptr0 + tl.arange(0, 256), mask=bias_mask)  # 256-elem float32 register array, 8x LDG.128
    # Stage 2: main pointwise computation
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = tl.full([XBLOCK], True, tl.int1)[:]
    x2 = tl.max_contiguous(tl.multiple_of(xindex, XBLOCK), XBLOCK)
    x0 = (xindex % 256).to(tl.int32)  # channel index into bias register array
    tmp0 = tl.load(in_out_ptr0 + (x2), mask=xmask)
    tmp1 = tl.gather(bias_vec, x0, 0)  # register-speed lookup, zero L1TEX stall
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1], 0, tl.int32)
    tmp4 = tl.maximum(tmp3, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp4, mask=xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    xnumel = 191102976
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_convolution_relu_5[grid](arg_0, arg_1, 191102976, XBLOCK=128)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((1024, 256, 27, 27), (186624, 1, 6912, 256), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((256,), (1,), device='cuda:0', dtype=torch.float32)
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
