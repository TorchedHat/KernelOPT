import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=16, num_stages=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4,  num_stages=4),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4,  num_stages=4),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4,  num_stages=4),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=8,  num_stages=4),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 256,  'YBLOCK': 4}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4,  num_stages=3),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=8,  num_stages=3),
        triton.Config({'XBLOCK': 2048, 'YBLOCK': 1}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 1024, 'YBLOCK': 1}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 512,  'YBLOCK': 2}, num_warps=8,  num_stages=2),
        triton.Config({'XBLOCK': 128,  'YBLOCK': 8}, num_warps=4,  num_stages=2),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_0(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    _xrange = tl.arange(0, XBLOCK)
    _xrange = tl.max_contiguous(tl.multiple_of(_xrange + xoffset, XBLOCK), XBLOCK)
    xindex = _xrange[None, :]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = (yindex % 3)
    y1 = yindex // 3
    tmp0 = tl.load(in_ptr0 + (x2 + 50176*y3), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + 3*x2 + 150528*y1), tmp0, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 30
    xnumel = 50176
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_0[grid](arg_0, arg_1, 30, 50176)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((10, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((10, 3, 224, 224), (150528, 1, 672, 3), device='cuda:0', dtype=torch.float32)
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
