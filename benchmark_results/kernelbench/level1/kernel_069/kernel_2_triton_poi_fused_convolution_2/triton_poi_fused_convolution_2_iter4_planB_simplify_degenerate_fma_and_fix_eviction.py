import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'YBLOCK': 32,  'XBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 32},  num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 32,  'XBLOCK': 64},  num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 64,  'XBLOCK': 32},  num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 32,  'XBLOCK': 64},  num_warps=8, num_stages=3),
        triton.Config({'YBLOCK': 16,  'XBLOCK': 64},  num_warps=4, num_stages=3),
        triton.Config({'YBLOCK': 16,  'XBLOCK': 128}, num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 8,   'XBLOCK': 128}, num_warps=4, num_stages=2),
        triton.Config({'YBLOCK': 128, 'XBLOCK': 32},  num_warps=8, num_stages=2),
        triton.Config({'YBLOCK': 4,   'XBLOCK': 256}, num_warps=4, num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 3025
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 96)
    y0 = tl.max_contiguous(tl.multiple_of(y0, [96, 1]), [96, 1])  # hint: channel axis is 96-aligned and contiguous
    y1 = yindex // 96
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), xmask & ymask, eviction_policy='evict_first')
    tmp1 = tl.load(in_ptr1 + (y0), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    arg_2 = inputs[2]
    ynumel = 24576
    xnumel = 3025
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_2[grid](arg_0, arg_1, arg_2, 24576, 3025)
    return arg_2

def get_inputs():
    arg_0 = rand_strided((256, 96, 55, 55), (290400, 1, 5280, 96), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((96,), (1,), device='cuda:0', dtype=torch.float32)
    arg_2 = rand_strided((256, 96, 55, 55), (290400, 3025, 55, 1), device='cuda:0', dtype=torch.float32)
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
