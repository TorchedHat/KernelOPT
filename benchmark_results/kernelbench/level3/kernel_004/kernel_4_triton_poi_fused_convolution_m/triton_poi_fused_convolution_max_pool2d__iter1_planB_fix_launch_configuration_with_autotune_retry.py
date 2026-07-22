import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        triton.Config({'XBLOCK': 16, 'YBLOCK': 64},  num_warps=4),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 256}, num_warps=8),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 128}, num_warps=4),
        triton.Config({'XBLOCK': 8,  'YBLOCK': 256}, num_warps=8),
        triton.Config({'XBLOCK': 4,  'YBLOCK': 512}, num_warps=8),
        triton.Config({'XBLOCK': 16, 'YBLOCK': 512}, num_warps=8),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 102400
    xnumel = 16
    yoffset = (tl.program_id(1) + tl.program_id(2) * tl.num_programs(1)) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = (yindex % 5)
    y4 = yindex // 5
    y2 = yindex // 25
    y5 = (yindex % 25)
    tmp0 = tl.load(in_ptr0 + (x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (16 + x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (160 + x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (176 + x3 + 32*y0 + 320*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y5 + 25*x3 + 400*y2), tmp6, xmask & ymask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 102400
    xnumel = 16
    grid = (triton.cdiv(xnumel, 8),)
    triton_poi_fused_convolution_max_pool2d_with_indices_relu_4[grid](arg_0, arg_1, 102400, 16)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((4096, 16, 10, 10), (1600, 1, 160, 16), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 16, 5, 5), (400, 25, 5, 1), device='cuda:0', dtype=torch.float32)
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
