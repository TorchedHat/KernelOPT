import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # num_stages=1 (original baseline, no pipelining)
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=4, num_stages=1),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 1}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 2}, num_warps=2, num_stages=1),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=1),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8, num_stages=1),
        # num_stages=2 (one prefetch buffer - hides ~47-cycle L2 latency)
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=4, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 1}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 2}, num_warps=2, num_stages=2),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=2),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8, num_stages=2),
        # num_stages=3 (two prefetch buffers - deeper pipelining for H200 TMA)
        triton.Config({'XBLOCK': 128, 'YBLOCK': 1}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 2}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 4}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 8}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 1}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 2}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 4}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 1}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 32,  'YBLOCK': 2}, num_warps=2, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 1}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 256, 'YBLOCK': 2}, num_warps=8, num_stages=3),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_2(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 3025
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.program_id(0) * XBLOCK
    # KEY CHANGE: contiguity hints enable 128-bit vectorized memory instructions
    xindex = xoffset + tl.max_contiguous(tl.multiple_of(tl.arange(0, XBLOCK), XBLOCK), XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = (yindex % 96)
    y1 = yindex // 96
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + 96*x2 + 290400*y1), mask=xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), mask=ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x2 + 3025*y3), tmp2, mask=xmask & ymask)

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
