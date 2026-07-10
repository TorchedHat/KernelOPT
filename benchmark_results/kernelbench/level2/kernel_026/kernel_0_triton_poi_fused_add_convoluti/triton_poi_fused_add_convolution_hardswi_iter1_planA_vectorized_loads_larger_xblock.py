import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 2048}, num_warps=16),
        triton.Config({'XBLOCK': 1024}, num_warps=8),
        triton.Config({'XBLOCK': 512},  num_warps=4),
    ],
    key=['xnumel'],
)
@triton.jit
def triton_poi_fused_add_convolution_hardswish_mul_0(in_out_ptr0, in_ptr0, in_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 268435456
    xoffset = tl.program_id(0) * XBLOCK
    xoffset = tl.multiple_of(xoffset, XBLOCK)
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = tl.max_contiguous(xindex, XBLOCK)
    x1 = ((xindex // 32768) % 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), xmask)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1], 3.0, tl.float32)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1], 0.0, tl.float32)
    tmp8 = tl.maximum(tmp6, tmp7)
    tmp9 = tl.full([1], 6.0, tl.float32)
    tmp10 = tl.minimum(tmp8, tmp9)
    tmp11 = tmp4 * tmp10
    tmp12 = tl.full([1], 0.16666666666666666, tl.float32)
    tmp13 = tmp11 * tmp12
    tmp14 = tmp4 * tmp13
    tl.store(in_out_ptr0 + (x3), tmp14, xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    xnumel = 268435456
    grid = (triton.cdiv(xnumel, 128),)
    triton_poi_fused_add_convolution_hardswish_mul_0[grid](arg_0, arg_1, arg_2, 268435456)
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
