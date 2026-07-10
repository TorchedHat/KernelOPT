import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4, num_stages=2),
    triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8, num_stages=2),
    triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8, num_stages=3),
    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4, num_stages=2),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_mish_mul_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK: tl.constexpr, XBLOCK: tl.constexpr):
    # Flat 1D kernel: treat all 536870912 elements as a single linear index
    # in_ptr0: (128,64,256,256) strides (4194304, 1, 16384, 64)
    # in_ptr1: (64,) strides (1,)
    # out_ptr0: (128,64,256,256) strides (4194304, 65536, 256, 1)
    n_elements = 536870912
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    mask = offsets < n_elements
    # Decompose flat index into (n, c, h, w) for shape (128,64,256,256)
    w   = offsets % 256
    tmp0_ = offsets // 256
    h   = tmp0_ % 256
    tmp1_ = tmp0_ // 256
    c   = tmp1_ % 64
    n   = tmp1_ // 64
    # in_ptr0 index using its strides (4194304, 1, 16384, 64)
    in0_idx = n * 4194304 + c * 1 + h * 16384 + w * 64
    # in_ptr1 index: bias indexed by channel c only
    in1_idx = c
    # out_ptr0 index using its strides (4194304, 65536, 256, 1)
    out_idx = n * 4194304 + c * 65536 + h * 256 + w
    tmp0 = tl.load(in_ptr0 + in0_idx, mask=mask)
    tmp1 = tl.load(in_ptr1 + in1_idx, mask=mask)
    tmp2 = tmp0 + tmp1
    # Mish: x * tanh(softplus(x)), numerically stable
    tmp4 = tmp2 > 20.0
    tmp5 = tl.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tmp2 * tmp8
    # Hardtanh: clamp(mish(x)+0.5, -1, 1) * 2
    tmp11 = tmp9 + 0.5
    tmp13 = tl.maximum(tmp11, -1.0)
    tmp15 = tl.minimum(tmp13,  1.0)
    tmp17 = tmp15 * 2.0
    tl.store(out_ptr0 + out_idx, tmp17, mask=mask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 8388608
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_add_convolution_hardtanh_mish_mul_2[grid](arg_0, arg_1, arg_2, 8388608, 64)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 64, 256, 256), (4194304, 1, 16384, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 64, 256, 256), (4194304, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
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
