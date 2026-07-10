import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 8192, 'YBLOCK': 1}, num_warps=8, num_stages=2),
], key=['xnumel'])
@triton.jit
def triton_poi_fused_convolution_mish_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat 1D kernel over all N*C*H*W = 528,515,072 output elements
    # in_ptr0: NHWC strides (8258048, 1, 32512, 128), N=64,C=128,H=254,W=254
    # out_ptr0: NCHW contiguous strides (8258048, 64516, 254, 1)
    numel = 528515072  # 64 * 128 * 254 * 254
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < numel
    # Decompose flat NCHW output index into (n, c, h, w)
    HW = 64516        # 254 * 254
    CHW = 8258048     # 128 * 64516
    n = offsets // CHW
    rem = offsets % CHW
    c = rem // HW
    rem2 = rem % HW
    h = rem2 // 254
    w = rem2 % 254
    # Load input using NHWC strides: n*8258048 + c*1 + h*32512 + w*128
    in_idx = n * 8258048 + c + h * 32512 + w * 128
    tmp0 = tl.load(in_ptr0 + in_idx, mask=mask)
    tmp1 = tl.load(in_ptr1 + c, mask=mask)  # bias indexed by channel
    tmp2 = tmp0 + tmp1
    # First Mish: x * tanh(softplus(x))
    tmp3 = tl.full([1], 20.0, tl.float32)
    tmp4 = tmp2 > tmp3
    tmp5 = tl.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tmp2 * tmp8
    # Second Mish: x * tanh(softplus(x))
    tmp10 = tmp9 > tmp3
    tmp11 = tl.exp(tmp9)
    tmp12 = libdevice.log1p(tmp11)
    tmp13 = tl.where(tmp10, tmp9, tmp12)
    tmp14 = libdevice.tanh(tmp13)
    tmp15 = tmp9 * tmp14
    # Store to contiguous NCHW output (flat index = offsets)
    tl.store(out_ptr0 + offsets, tmp15, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 4129024
    xnumel = 128
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_mish_2[grid](arg_0, arg_1, arg_2, 4129024, 128)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((64, 128, 254, 254), (8258048, 1, 32512, 128), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((128,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((64, 128, 254, 254), (8258048, 64516, 254, 1), device='cuda:0', dtype=torch.float32)
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
