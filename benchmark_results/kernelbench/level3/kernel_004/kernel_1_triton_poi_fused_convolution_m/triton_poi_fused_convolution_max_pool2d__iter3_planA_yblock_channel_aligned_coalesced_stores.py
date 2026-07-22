import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(
    configs=[
        # YBLOCK=16: power-of-2, 16/6 gives partial channel groups but more threads per block
        triton.Config({'XBLOCK': 32,  'YBLOCK': 16}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 16}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 16}, num_warps=8, num_stages=3),
        # YBLOCK=32: power-of-2, larger Y tile for more ILP
        triton.Config({'XBLOCK': 32,  'YBLOCK': 32}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=4, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 32}, num_warps=8, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 32}, num_warps=8, num_stages=3),
        # YBLOCK=64: maximum Y reuse, power-of-2
        triton.Config({'XBLOCK': 32,  'YBLOCK': 64}, num_warps=4, num_stages=4),
        triton.Config({'XBLOCK': 64,  'YBLOCK': 64}, num_warps=8, num_stages=3),
        triton.Config({'XBLOCK': 128, 'YBLOCK': 64}, num_warps=8, num_stages=4),
    ],
    key=['ynumel', 'xnumel'],
)
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 196
    yoffset = tl.multiple_of(tl.program_id(1) * YBLOCK, YBLOCK)
    yindex = tl.max_contiguous(yoffset + tl.arange(0, YBLOCK), YBLOCK)[:, None]
    ymask = tl.full([YBLOCK], True, tl.int1)[:, None]
    xoffset = tl.multiple_of(tl.program_id(0) * XBLOCK, XBLOCK)
    xindex = tl.max_contiguous(xoffset + tl.arange(0, XBLOCK), XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 14)
    x3 = xindex // 14
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 6)
    y1 = yindex // 6
    tmp0 = tl.load(in_ptr0 + (2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + 2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + 2*x2 + 56*x3 + 784*y4), mask=ymask & xmask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 6*x5 + 1176*y1), tmp6, mask=ymask & xmask)

def kernel_function(*inputs):
    arg_0 = inputs[0]
    arg_1 = inputs[1]
    ynumel = 24576
    xnumel = 196
    grid = (triton.cdiv(xnumel, 128), triton.cdiv(ynumel, 8))
    triton_poi_fused_convolution_max_pool2d_with_indices_relu_1[grid](arg_0, arg_1, 24576, 196)
    return arg_1

def get_inputs():
    arg_0 = rand_strided((4096, 6, 28, 28), (4704, 784, 28, 1), device='cuda:0', dtype=torch.float32)
    arg_1 = rand_strided((4096, 6, 14, 14), (1176, 1, 84, 6), device='cuda:0', dtype=torch.float32)
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
