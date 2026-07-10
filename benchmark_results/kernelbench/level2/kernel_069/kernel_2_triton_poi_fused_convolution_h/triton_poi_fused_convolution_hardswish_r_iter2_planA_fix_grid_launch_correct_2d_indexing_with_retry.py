import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 1024, 'XBLOCK': 64}, num_warps=8),
        triton.Config({'YBLOCK': 512,  'XBLOCK': 64}, num_warps=4),
        triton.Config({'YBLOCK': 256,  'XBLOCK': 64}, num_warps=4),
        triton.Config({'YBLOCK': 128,  'XBLOCK': 64}, num_warps=4),
        triton.Config({'YBLOCK': 64,   'XBLOCK': 64}, num_warps=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_hardswish_relu_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2032128
    xnumel = 64
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 15876)
    y1 = yindex // 15876
    tmp0 = tl.load(in_ptr0 + (x2 + 64*y3), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 3.0, tl.float32)
    tmp4 = tmp2 + tmp3
    tmp5 = tl.full([1, 1], 0.0, tl.float32)
    tmp6 = tl.maximum(tmp4, tmp5)
    tmp7 = tl.full([1, 1], 6.0, tl.float32)
    tmp8 = tl.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tl.full([1, 1], 0.16666666666666666, tl.float32)
    tmp11 = tmp9 * tmp10
    tmp12 = tl.full([1, 1], 0, tl.int32)
    tmp13 = tl.maximum(tmp12, tmp11)
    tl.store(out_ptr0 + (y0 + 15876*x2 + 1016064*y1), tmp13, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 2032128
    xnumel = 64
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_hardswish_relu_2[grid](arg_0, arg_1, arg_2, 2032128, 64)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((128, 64, 126, 126), (1016064, 1, 8064, 64), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((64,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((128, 64, 126, 126), (1016064, 15876, 126, 1), device='cuda:0', dtype=torch.float32)
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
