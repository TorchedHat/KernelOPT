import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice
from torch._dynamo.testing import rand_strided
from torch._inductor.runtime import triton_helpers


@triton.autotune(configs=[
    triton.Config({'YBLOCK': 128, 'XBLOCK': 16}, num_warps=4),
    triton.Config({'YBLOCK': 256, 'XBLOCK': 8}, num_warps=4),
    triton.Config({'YBLOCK': 64, 'XBLOCK': 16}, num_warps=2),
    triton.Config({'YBLOCK': 128, 'XBLOCK': 8}, num_warps=4),
    triton.Config({'YBLOCK': 512, 'XBLOCK': 4}, num_warps=8),
    triton.Config({'YBLOCK': 32, 'XBLOCK': 32}, num_warps=4),
], key=['ynumel', 'xnumel'])
@triton.jit
def triton_poi_fused_convolution_max_pool2d_with_indices_relu_1(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 196
    yoffset = tl.program_id(0) * YBLOCK
    _yindex = tl.max_contiguous(tl.multiple_of(yoffset + tl.arange(0, YBLOCK), YBLOCK), YBLOCK)
    yindex = _yindex[:, None]
    ymask = yindex < ynumel
    xoffset = tl.program_id(1) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[None, :]
    xmask = xindex < xnumel
    x2 = (xindex % 14)
    x3 = xindex // 14
    y4 = yindex
    x5 = xindex
    y0 = (yindex % 6)
    y1 = yindex // 6
    tmp0 = tl.load(in_ptr0 + (2*x2 + 56*x3 + 784*y4), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + 2*x2 + 56*x3 + 784*y4), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + 2*x2 + 56*x3 + 784*y4), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + 2*x2 + 56*x3 + 784*y4), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.maximum(tmp0, tmp1)
    tmp4 = tl.maximum(tmp2, tmp3)
    tmp6 = tl.maximum(tmp4, tmp5)
    tl.store(out_ptr0 + (y0 + 6*x5 + 1176*y1), tmp6, xmask & ymask)

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
