import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 8},  num_warps=4),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 4},  num_warps=8),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4},  num_warps=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_add_convolution_lift_fresh_minimum_mul_2(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 15876
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    # Apply vectorization hints on 1D arange before broadcasting
    xindex = tl.max_contiguous(tl.multiple_of(xoffset, XBLOCK) + tl.arange(0, XBLOCK), XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 128)
    y1 = yindex // 128
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 128*x2 + 2032128*y1), mask=xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (y0), mask=ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.full([1, 1], 0.5, tl.float32)
    tmp4 = tl.minimum(tmp2, tmp3)
    tmp6 = tmp4 + tmp5
    tmp7 = tl.full([1, 1], 2.0, tl.float32)
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2 + 15876*y3), tmp8, mask=xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    arg_3 = inputs[3]
    ynumel = 16384
    xnumel = 15876
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_add_convolution_lift_fresh_minimum_mul_2[grid](arg_0, arg_1, arg_2, arg_3, 16384, 15876)
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
