import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4),
    triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8),
    triton.Config({'XBLOCK': 512,  'YBLOCK': 1}, num_warps=4),
    triton.Config({'XBLOCK': 4096, 'YBLOCK': 1}, num_warps=8),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_add_convolution_hardtanh_mish_mul_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    # Flat 1D indexing: treat all 536870912 elements as a 1D array
    # numel = ynumel * xnumel = 8388608 * 64 = 536870912
    # XBLOCK is used as BLOCK_SIZE for the flat 1D grid
    pid = tl.program_id(0)
    offsets = pid * XBLOCK + tl.arange(0, XBLOCK)
    numel = ynumel * xnumel
    mask = offsets < numel
    x2 = offsets % 64
    y3 = offsets // 64
    y0 = y3 % 65536
    y1 = y3 // 65536
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), mask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + x2, mask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([XBLOCK], 20.0, tl.float32)
    tmp4 = tmp2 > tmp3
    tmp5 = tl.exp(tmp2)
    tmp6 = libdevice.log1p(tmp5)
    tmp7 = tl.where(tmp4, tmp2, tmp6)
    tmp8 = libdevice.tanh(tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tl.full([XBLOCK], 0.5, tl.float32)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full([XBLOCK], -1.0, tl.float32)
    tmp13 = tl.maximum(tmp11, tmp12)
    tmp14 = tl.full([XBLOCK], 1.0, tl.float32)
    tmp15 = tl.minimum(tmp13, tmp14)
    tmp16 = tl.full([XBLOCK], 2.0, tl.float32)
    tmp17 = tmp15 * tmp16
    out_idx = y0 + 65536*x2 + 4194304*y1
    tl.store(out_ptr0 + out_idx, tmp17, mask)

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
